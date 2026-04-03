# copies from the omnijet alpha repo

import logging
import time
from pathlib import Path
from typing import Tuple
import awkward as ak
from collections import defaultdict



import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vector
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist

from tqdm import tqdm

from src.models.optimizers import configure_optimizers_base
from src.models.positional_encoding import DetectorPosEnc



# vqtorch can be installed from https://github.com/minyoungg/vqtorch
try:
    from vqtorch.nn import VectorQuant  # type: ignore
except ImportError as e:
    raise ImportError("vqtorch is not installed. Please install it to use this module.") from e

from src.utils.arrays import (
    ak_pad,
    ak_select_and_preprocess,
    ak_to_np_stack,
    np_to_ak,
)

vector.register_awkward()

logger = logging.getLogger(__name__)



class NormformerBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # define the MultiheadAttention layer with layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(input_dim)

        # define the MLP with layer normalization
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),  # Add layer normalization
            nn.Linear(input_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(mlp_dim, input_dim),
        )

        # initialize weights of mlp[-1] and layer norm after attn block to 0
        # such that the residual connection is the identity when the block is
        # initialized
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.norm1.weight)

    def forward(self, x, mask=None, return_attn_weights=False):
        # x: (B, S, F)
        # mask: (B, S)
        x = x * mask.unsqueeze(-1)

        # calculate self-attention
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask != 1)
        # Add residual connection and permute back to (B, S, F)
        attn_res = self.norm2(attn_output) + x

        output = self.mlp(attn_res) + attn_res

        if return_attn_weights:
            return output, attn_weights

        # output shape: (B, S, F)
        return output


class Transformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        skip_out_proj=False,
    ):
        super().__init__()

        self.project_in = nn.Linear(input_dim, hidden_dim)

        self.num_blocks = num_blocks
        self.skip_out_proj = skip_out_proj
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.blocks = nn.ModuleList(
            [NormformerBlock(input_dim=hidden_dim, mlp_dim=hidden_dim, num_heads=num_heads) for _ in range(num_blocks)]
        )
        self.project_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask):
        x = self.project_in(x)
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask)
        if self.skip_out_proj:
            return x * mask.unsqueeze(-1)
        x = self.project_out(x) * mask.unsqueeze(-1)
        return x


class NormformerStack(torch.nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        skip_out_proj=False,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.skip_out_proj = skip_out_proj
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.blocks = nn.ModuleList(
            [
                NormformerBlock(
                    input_dim=self.hidden_dim,
                    mlp_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, mask):
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask)
        return x * mask.unsqueeze(-1)




class VQVAENormFormer(torch.nn.Module):
    """This is basically just a re-factor of the VQVAETransformer class, but with more modular
    model components, making it easier to use some components in other models."""

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        vq_kwargs={},
        vit_kwargs={},
        **kwargs,
    ):
        super().__init__()

        self.loss_history = []
        self.lr_history = []

        self.vq_kwargs = vq_kwargs
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # Model components:
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder_normformer = NormformerStack(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
        )
        self.latent_projection_in = nn.Linear(self.hidden_dim, self.latent_dim)
        self.vqlayer = VectorQuant(feature_size=self.latent_dim, **vq_kwargs)
        self.latent_projection_out = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_normformer = NormformerStack(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
        )
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        # ViT components

        # each patch size needs its own linear encoder
        self.linear_projection_encoders, self.linear_projection_decoders = torch.nn.ModuleDict(), torch.nn.ModuleDict()
        for key in vit_kwargs["unique_patch_sizes_dict"].keys():
            num_bins_in_patch = np.prod([k for k in key])
            self.linear_projection_encoders[str(key)] = torch.nn.Linear(num_bins_in_patch, vit_kwargs["D_EMBEDDING"])
            self.linear_projection_decoders[str(key)] = torch.nn.Linear(vit_kwargs["D_EMBEDDING"], num_bins_in_patch)
        
       
        self.positional_encoding = DetectorPosEnc(
            phi_max_per_r = vit_kwargs["n_phi_per_ring"],
            n_z =  vit_kwargs["n_bins_z"],
            d_latent = vit_kwargs["D_EMBEDDING"],
        )        


    def forward(self, batch):

        embeddings = []
        global_patch_ids = []
        local_patch_ids = []
        mask = []
        patch_group_sizes = []  # track how many patches per group, for splitting later
    
        # 1. encode each patch group
        for key in sorted(batch.keys()):
            key_str = str(key)

         

            emb = self.linear_projection_encoders[key_str](batch[key]["flat_tensor"])  # (B, P_k, D) P_k = num. patches per key. should have sum P_k = P
            P_k = emb.shape[1]
            embeddings.append(emb)
            global_patch_ids.append(batch[key]["global_patch_ids"])
            local_patch_ids.append(batch[key]["local_patch_ids"])
            mask.append(batch[key]["mask"])
            patch_group_sizes.append(P_k)

        # 2. concatenate all patches
        embeddings       = torch.cat(embeddings,       dim=1)  # (B, P_total, D)
        global_patch_ids = torch.cat(global_patch_ids, dim=1)  # (B, P_total)
        local_patch_ids  = torch.cat(local_patch_ids,  dim=1)  # (B, P_total, 3)
        mask             = torch.cat(mask,             dim=1)  # (B, P_total)
    
        # 3. reorder all tensors by global patch id
        order   = torch.argsort(global_patch_ids, dim=1)
        order_D = order.unsqueeze(-1).expand_as(embeddings)
        order_3 = order.unsqueeze(-1).expand_as(local_patch_ids)
    
        embeddings      = torch.gather(embeddings,      dim=1, index=order_D)
        local_patch_ids = torch.gather(local_patch_ids, dim=1, index=order_3)
        mask            = torch.gather(mask,            dim=1, index=order)
    
        # 4. add positional encoding
        r_idx, phi_idx, z_idx = local_patch_ids[..., 0], local_patch_ids[..., 1], local_patch_ids[..., 2]

        e = embeddings + self.positional_encoding(r_idx, phi_idx, z_idx)  # (B, P_total, D)

    
        # 5. encode → quantize → decode
        e       = self.input_projection(e)
        e       = self.encoder_normformer(e, mask=mask)
        z_embed = self.latent_projection_in(e) * mask.unsqueeze(-1)




            
        z, vq_out = self.vqlayer(z_embed)

        e_reco  = self.latent_projection_out(z) * mask.unsqueeze(-1)
        e_reco  = self.decoder_normformer(e_reco, mask=mask)
        e_reco  = self.output_projection(e_reco) * mask.unsqueeze(-1)
    
        # 6. undo the sort so patches line up with their original key groupings
        # argsort of argsort gives the inverse permutation
        inv_order = torch.argsort(order, dim=1)
        e_reco_unordered = torch.gather(e_reco, dim=1, index=inv_order.unsqueeze(-1).expand_as(e_reco))
    
        # 7. split back by patch group and decode each with its own linear decoder
        x_reco_chunks = {}
        start = 0
        for key, P_k in zip(sorted(batch.keys()), patch_group_sizes):
            chunk = e_reco_unordered[:, start:start + P_k, :]      # (B, P_k, D)
            x_reco_chunks[key] = self.linear_projection_decoders[str(key)](chunk)  # (B, P_k, bins_k)
            start += P_k

        return e, e_reco, {key:batch[key]["flat_tensor"] for key in batch.keys()}, x_reco_chunks, vq_out



class VQVAELightning(L.LightningModule):
    """PyTorch Lightning module for training a VQ-VAE."""

    def __init__(
        self,
        optimizer_kwargs={},
        lr_scheduler_kwargs = {"use_scheduler":False},
        model_kwargs={},
        vit_kwargs={},
        model_type="Transformer",
        num_train_events=0,
        batch_size_per_gpu=0,
        plot_dir_name="",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        if model_type == "MLP":
            self.model = VQVAEMLP(**model_kwargs)
        elif model_type == "Transformer":
            self.model = VQVAETransformer(**model_kwargs)
        elif model_type == "VQVAENormFormer":
            self.model = VQVAENormFormer(**model_kwargs, vit_kwargs=vit_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.train_loss_history = []
        self.val_loss_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        # loss function (not used atm, since we calc MSE manually)
        self.criterion = torch.nn.MSELoss()

        # for tracking best so far validation accuracy
        self.val_x_original = []
        self.val_x_reco = []
        self.val_mask = []

        self.num_train_events = num_train_events
        self.batch_size_per_gpu = batch_size_per_gpu

        self.plot_dir_name = plot_dir_name

        self.vit_kwargs = vit_kwargs

        
        

    def configure_optimizers(self):
        return configure_optimizers_base(self)



    def forward(self, batch):

        
        embedding_hit, embedding_hit_reco, patches_chunked, patches_chunked_reco, vq_out = self.model(batch)

        return embedding_hit, embedding_hit_reco, patches_chunked, patches_chunked_reco, vq_out

    def model_step(self, batch, return_x=False):
        """Perform a single model step on a batch of data."""


        embedding_hit, embedding_hit_reco, patches_chunked, patches_chunked_reco, vq_out = self.forward(batch)

            

        reco_loss = torch.stack([
            ((patches_chunked[key] - patches_chunked_reco[key]) ** 2).mean()
            for key in patches_chunked.keys()
        ]).mean()

        
        alpha = self.hparams["model_kwargs"]["alpha"]
        cmt_loss = vq_out["loss"]
        loss = reco_loss + alpha * cmt_loss

        if return_x:
            return loss, reco_loss, cmt_loss, embedding_hit, embedding_hit_reco, batch, patches_chunked_reco, vq_out

        return loss, reco_loss, cmt_loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss, reco_loss, cmt_loss = self.model_step(batch)

        self.train_loss_history.append(loss.detach().cpu().numpy())
        self.log(
                "train/total_loss",
                loss,                # <-- pass the tensor, not loss.item()
                on_step=True,
                on_epoch=True,       # optional if you also want epoch avg
                prog_bar=True,
                sync_dist=True       # sync across GPUs
            )
        self.log(
                "train/reco_loss",
                reco_loss,                # <-- pass the tensor, not loss.item()
                on_step=True,
                on_epoch=True,       # optional if you also want epoch avg
                prog_bar=True,
                sync_dist=True       # sync across GPUs
            )
        self.log(
                "train/cmt_loss",
                cmt_loss,                # <-- pass the tensor, not loss.item()
                on_step=True,
                on_epoch=True,       # optional if you also want epoch avg
                prog_bar=True,
                sync_dist=True       # sync across GPUs
            )

        return loss


    def on_train_epoch_start(self):
        logger.info(f"Epoch {self.trainer.current_epoch} starting.")
        self.epoch_train_start_time = time.time()  # start timing the epoch

    def on_train_epoch_end(self):
        self.epoch_train_end_time = time.time()
        self.epoch_train_duration_minutes = (self.epoch_train_end_time - self.epoch_train_start_time) / 60
        self.log(
            "epoch_train_duration_minutes",
            self.epoch_train_duration_minutes,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        logger.info(
            f"Epoch {self.trainer.current_epoch} finished in" f" {self.epoch_train_duration_minutes:.1f} minutes."
        )

    def on_train_end(self):
        pass

    def on_validation_epoch_start(self) -> None:

        self.val_x_original = []
        self.val_x_reco = []
        self.val_mask = []
        self.val_labels = []
        self.val_code_idx = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, reco_loss, cmt_loss, embedding_hit, embedding_hit_reco, batch, patches_chunked_reco, vq_out = self.model_step(batch, return_x=True)

 
        # save the original and reconstructed data
        # self.val_x_original.append(x_original.detach().cpu().numpy())
        # self.val_x_reco.append(x_reco.detach().cpu().numpy())
        # self.val_mask.append(mask.detach().cpu().numpy())
        # self.val_labels.append(labels.detach().cpu().numpy())
        # self.val_code_idx.append(vq_out["q"].detach().cpu().numpy())

        self.log("val/total_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("val/reco_loss", reco_loss.item(), on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("val/cmt_loss", cmt_loss.item(), on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)

        # for the first validation step, plot the model
        if batch_idx == 0:
            # get loggers
            comet_logger = None
            for logger in self.trainer.loggers:
                if isinstance(logger, L.pytorch.loggers.CometLogger):
                    comet_logger = logger.experiment

            curr_epoch, curr_step = self.trainer.current_epoch, self.trainer.global_step

            plot_dir = Path(self.trainer.default_root_dir + f"/plots/{self.plot_dir_name}/")
            plot_dir.mkdir(exist_ok=True)
            plot_filename = f"{plot_dir}/epoch{curr_epoch}_gstep{curr_step}"
            # log the plot
            plot_model(
                batch=batch,
                patches_chunked_reco=patches_chunked_reco,
                vq_out=vq_out,
                num_codes=self.model.vq_kwargs["num_codes"],
                device=self.device,
                vit_kwargs=self.vit_kwargs,
                saveas=plot_filename,
            )
            if comet_logger is not None:
                comet_logger.log_image(plot_filename, name=plot_filename.split("/")[-1], step=curr_step)

        return loss

    def on_test_epoch_start(self) -> None:
        self.test_x_original = []
        self.test_x_reco = []
        self.test_mask = []
        self.test_labels = []
        self.test_code_idx = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, x_original, x_reco, mask, labels, vq_out = self.model_step(batch, return_x=True)

        # save the original and reconstructed data
        self.test_x_original.append(x_original.detach().cpu().numpy())
        self.test_x_reco.append(x_reco.detach().cpu().numpy())
        self.test_mask.append(mask.detach().cpu().numpy())
        self.test_labels.append(labels.detach().cpu().numpy())
        self.test_code_idx.append(vq_out["q"].detach().cpu().numpy())

        self.log("test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)

    def tokenize_dataloader(self, dataloader, hide_pbar=False, pad_length=15000, add_start_end_tokens=False):
        """Tokenize a dataloader of calo hit events.

        Parameters
        ----------
        ak_arr : ak.Array
            Awkward array of jets, shape (N_jets, <var>, N_features).
        pp_dict : dict
            Dictionary with preprocessing information.
        batch_size : int, optional
            Batch size for the evaluation loop. The default is 256.
        pad_length : int, optional
            Length to which the tokens are padded. The default is 128.
        hide_pbar : bool, optional
            Whether to hide the progress bar. The default is False.

        Returns
        -------
        ak.Array
            Awkward array of tokens, shape (N_jets, <var>).
        """

        ak_output = ak.Array([])



        with torch.no_grad():
            if not hide_pbar:
                pbar = tqdm(dataloader)
            else:
                pbar = dataloader

       
            for i, x_batch in enumerate(pbar):

                # move to device
                features_batch = x_batch["calo_hit_features"].to(self.device)
                mask_batch = x_batch["mask"].to(self.device)
                x_particle_reco, vq_out = self.forward(features_batch, mask_batch)
                code = vq_out["q"]

                code = code.squeeze(-1).detach().cpu().numpy()
                mask_batch = mask_batch.squeeze(-1).detach().cpu().numpy().astype(int)

                for row in range(code.shape[0]):

                    row_codes = code[row][mask_batch[row] == 1]
                    if add_start_end_tokens:
                        
                        n_tokens = self.model.vqlayer.num_codes
                        row_codes = np.concatenate([[0], row_codes + 1, [n_tokens + 1]])

                    ak_output = ak.concatenate([ak_output, ak.Array([row_codes])], axis = 0)
       
        
        return ak_output

    def reconstruct_ak_tokens(self, tokens_dataloader, hide_pbar=False):
        """Reconstruct tokenized awkward array.

        Parameters
        ----------
        tokens_ak : ak.Array
            Awkward array of tokens, shape (N_jets, <var>).
        pp_dict : dict
            Dictionary with preprocessing information.
        batch_size : int, optional
            Batch size for the evaluation loop. The default is 256.
        pad_length : int, optional
            Length to which the tokens are padded. The default is 128.
        hide_pbar : bool, optional
            Whether to hide the progress bar. The default is False.

        Returns
        -------
        ak.Array
            Awkward array of reconstructed jets, shape (N_jets, <var>, N_features).
        """

        self.model.eval()


        x_reco = ak.Array([])

        codebook = self.model.vqlayer.codebook.weight


        # if the codebook has an affine transform, apply it
        # before using it to reconstruct the data
        # see https://github.com/minyoungg/vqtorch/blob/main/vqtorch/nn/vq.py#L102-L104
        if hasattr(self.model.vqlayer, "affine_transform"):
            codebook = self.model.vqlayer.affine_transform(codebook)

        last_batch = None

        with torch.no_grad():
            if not hide_pbar:
                pbar = tqdm(tokens_dataloader)
            else:
                pbar = tokens_dataloader
            for i, (batch) in enumerate(pbar):
                # move to device
                tokens_batch = batch["token_features"].to(self.device).int().squeeze(2) # extra dimension for collater
                mask_batch = batch["mask"].to(self.device) # shape: batch_size, num_tokens / hits
                #try:
                z_q = F.embedding(tokens_batch, codebook) # shape: batch_size, num_tokens, d_latent_space
                #except Exception as e:  # noqa: E722
                 #   print(f"Error in embedding: {e}")
                 #   print("batch shape", tokens_batch.shape)
                 #   print("batch max", tokens_batch.max())
                  #  print("batch min", tokens_batch.min())

                if last_batch is not None:
                    break

                if hasattr(self.model, "latent_projection_out"):
                    x_reco_batch = self.model.latent_projection_out(z_q) * mask_batch.unsqueeze(-1)
                    x_reco_batch = self.model.decoder_normformer(x_reco_batch, mask=mask_batch)
                    x_reco_batch = self.model.output_projection(x_reco_batch) * mask_batch.unsqueeze(-1)
                elif hasattr(self.model, "decoder"):
                    x_reco_batch = self.model.decoder(z_q)
                else:
                    raise ValueError("Unknown model structure. Cannot reconstruct.")

                x_reco = ak.concatenate([x_reco, x_reco_batch.detach().cpu().numpy()], axis = 0)

    
        return x_reco

