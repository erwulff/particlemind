# copies from the omnijet alpha repo

import logging
import time
from pathlib import Path
from typing import Tuple
import awkward as ak


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
from src.datasets.augmentations import standardize_calo_hit_features_rphiz, augment_data



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


class VQVAEMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=2,
        latent_dim=2,
        encoder_layers=None,
        decoder_layers=None,
        vq_kwargs={},
        **kwargs,
    ):
        """Initializes the VQ-VAE model.

        Parameters
        ----------
        codebook_size : int, optional
            The size of the codebook. The default is 8.
        embed_dim : int, optional
            The dimension of the embedding space. The default is 2.
        input_dim : int, optional
            The dimension of the input data. The default is 2.
        encoder_layers : list, optional
            List of integers representing the number of units in each encoder layer.
            If None, a default encoder with a single linear layer is used. The default is None.
        decoder_layers : list, optional
            List of integers representing the number of units in each decoder layer.
            If None, a default decoder with a single linear layer is used. The default is None.
        """

        super().__init__()
        self.vq_kwargs = vq_kwargs
        self.embed_dim = latent_dim
        self.input_dim = input_dim  # for jet constituents, eta and phi

        # --- Encoder --- #
        if encoder_layers is None:
            self.encoder = torch.nn.Linear(self.input_dim, self.embed_dim)
        else:
            enc_layers = []
            enc_layers.append(torch.nn.Linear(self.input_dim, encoder_layers[0]))
            enc_layers.append(torch.nn.ReLU())

            for i in range(len(encoder_layers) - 1):
                enc_layers.append(torch.nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
                enc_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.Linear(encoder_layers[-1], self.embed_dim))

            self.encoder = torch.nn.Sequential(*enc_layers)

        # --- Vector-quantization layer --- #
        self.vqlayer = VectorQuant(feature_size=self.embed_dim, **vq_kwargs)

        # --- Decoder --- #
        if decoder_layers is None:
            self.decoder = torch.nn.Linear(self.embed_dim, self.input_dim)
        else:
            dec_layers = []
            dec_layers.append(torch.nn.Linear(self.embed_dim, decoder_layers[0]))
            dec_layers.append(torch.nn.ReLU())

            for i in range(len(decoder_layers) - 1):
                dec_layers.append(torch.nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
                dec_layers.append(torch.nn.ReLU())
            dec_layers.append(torch.nn.Linear(decoder_layers[-1], self.input_dim))

            self.decoder = torch.nn.Sequential(*dec_layers)

        self.loss_history = []
        self.lr_history = []

    def forward(self, samples, mask=None):
        # mask is there for compatibility with the transformer model
        # encode
        z_embed = self.encoder(samples)
        # quantize
        z_q2, vq_out = self.vqlayer(z_embed)
        # decode
        x_reco = self.decoder(z_q2)
        return x_reco, vq_out


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


class VQVAETransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        vq_kwargs={},
        **kwargs,
    ):
        super().__init__()

        self.vq_kwargs = vq_kwargs
        self.latent_dim = latent_dim

        self.encoder = Transformer(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )
        self.vqlayer = VectorQuant(feature_size=latent_dim, **vq_kwargs)
        self.decoder = Transformer(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )
        self.loss_history = []
        self.lr_history = []

    def forward(self, x, mask):
        # encode
        x = self.encoder(x, mask=mask)
        z_embed = x * mask.unsqueeze(-1)
        # quantize
        z, vq_out = self.vqlayer(z_embed)
        # decode
        x_reco = self.decoder(z, mask=mask)
        return x_reco, vq_out


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

    def forward(self, x, mask):
        # encode
        x = self.input_projection(x) # BS, num hits, hidden_dim
        x = self.encoder_normformer(x, mask=mask) # BS, num hits, hidden_dim
        z_embed = self.latent_projection_in(x) * mask.unsqueeze(-1)  # BS, num hits, latent_dim
        # quantize
        z, vq_out = self.vqlayer(z_embed) # BS, num hits, latent_dim
        # decode
        x_reco = self.latent_projection_out(z) * mask.unsqueeze(-1) # BS, num hits, hidden_dim
        x_reco = self.decoder_normformer(x_reco, mask=mask) # BS, num hits, hidden_dim
        x_reco = self.output_projection(x_reco) * mask.unsqueeze(-1) # BS, num hits, input_dim
        return x_reco, vq_out


class VQVAELightning(L.LightningModule):
    """PyTorch Lightning module for training a VQ-VAE."""

    def __init__(
        self,
        optimizer_kwargs={},
        lr_scheduler_kwargs = {"use_scheduler":False},
        model_kwargs={},
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
            self.model = VQVAENormFormer(**model_kwargs)
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

        

    def configure_optimizers(self):
        return configure_optimizers_base(self)



    def forward(self, x_particle, mask_particle):
        x_particle_reco, vq_out = self.model(x_particle, mask=mask_particle)
        return x_particle_reco, vq_out



    def contrastive_loss(self, z1, z2, temperature=0.1, alpha=1):

        def pool(z):
            # z: (B, N, 1, D)
            z = z.squeeze(2)  # (B, N, D)
            return z.mean(dim=1)  # or sum / attention pooling

        z1 = pool(z1)
        z2 = pool(z2)
        # inputs have shape (B, latent_dim)

        # SimCLR loss

        batch_size = z1.shape[0]
        z1 = F.normalize( z1, dim=1 )
        z2 = F.normalize( z2, dim=1 )
        z   = torch.cat( [z1, z2], dim=0 )
        similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
        sim_ij = torch.diag( similarity_matrix,  batch_size ) # batch_size above the main diagonal -- x_i x_i'
        sim_ji = torch.diag( similarity_matrix, -batch_size ) # below the main diagonal -- x_i' x_i
        positives = torch.cat( [sim_ij, sim_ji], dim=0 )
        nominator = torch.exp( positives / temperature )
        negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float().to(z1.device)
        denominator = negatives_mask * torch.exp( similarity_matrix / temperature )
        loss_partial = -torch.log( nominator / (torch.sum( denominator, dim=1 )).pow(exponent=alpha) )
        loss = torch.sum( loss_partial )/( 2*batch_size )

        return loss

    def model_step(self, batch, return_x=False):
        """Perform a single model step on a batch of data."""

        alpha = self.hparams["model_kwargs"]["alpha"]
        beta = self.hparams["model_kwargs"]["beta"]

        # x_particle, mask_particle, labels = batch
        x_particle = batch["calo_hit_features"]
        mask_particle = batch["mask"]
        labels = batch["hit_labels"]   
        
        if beta != 0:
            # augment data
            x_particle_augmented = self.augment_data(x_particle)
            x_particle_augmented = standardize_calo_hit_features_rphiz(x_particle_augmented)
            x_particle_augmented_reco, vq_out_augmented = self.forward(x_particle_augmented, mask_particle)
            ssl_loss = self.contrastive_loss(vq_out["z"], vq_out_augmented["z"])
        else:
            ssl_loss = 0

        x_particle = standardize_calo_hit_features_rphiz(x_particle)

        print(x_particle)
        print(x_particle_augmented)
        exit()
        x_particle_reco, vq_out = self.forward(x_particle, mask_particle)

        reco_loss = ((x_particle_reco - x_particle) ** 2).mean()
        
        cmt_loss = vq_out["loss"]
        code_idx = vq_out["q"]
        
        loss = reco_loss + alpha * cmt_loss + beta * ssl_loss

        if return_x:
            return loss, reco_loss, cmt_loss, ssl_loss, x_particle, x_particle_reco, mask_particle, labels, code_idx

        return loss, reco_loss, cmt_loss, ssl_loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss, reco_loss, cmt_loss, ssl_loss = self.model_step(batch)

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
        self.log(
                "train/ssl_loss",
                loss,                # <-- pass the tensor, not loss.item()
                on_step=True,
                on_epoch=True,       # optional if you also want epoch avg
                prog_bar=True,
                sync_dist=True       # sync across GPUs
            )

        return loss

    """
    def on_train_start(self) -> None:
        self.preprocessing_dict = (
            self.trainer.datamodule.hparams.dataset_kwargs_common.feature_dict
        )
    """

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
        loss, reco_loss, cmt_loss, ssl_loss, x_original, x_reco, mask, labels, code_idx = self.model_step(batch, return_x=True)

 
        # save the original and reconstructed data
        self.val_x_original.append(x_original.detach().cpu().numpy())
        self.val_x_reco.append(x_reco.detach().cpu().numpy())
        self.val_mask.append(mask.detach().cpu().numpy())
        self.val_labels.append(labels.detach().cpu().numpy())
        self.val_code_idx.append(code_idx.detach().cpu().numpy())

        self.log("val/total_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("val/reco_loss", reco_loss.item(), on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("val/cmt_loss", cmt_loss.item(), on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("val/ssl_loss", ssl_loss.item(), on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)

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
                self.model,
                input_data=batch["calo_hit_features"],
                masks=batch["mask"],
                labels=batch["hit_labels"],
                device=self.device,
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
        loss, x_original, x_reco, mask, labels, code_idx = self.model_step(batch, return_x=True)

        # save the original and reconstructed data
        self.test_x_original.append(x_original.detach().cpu().numpy())
        self.test_x_reco.append(x_reco.detach().cpu().numpy())
        self.test_mask.append(mask.detach().cpu().numpy())
        self.test_labels.append(labels.detach().cpu().numpy())
        self.test_code_idx.append(code_idx.detach().cpu().numpy())

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


        
    """
    def on_validation_epoch_end(self) -> None:
        # Lightning hook that is called when a validation epoch ends.

        self.val_x_original_concat = np.concatenate(self.val_x_original)
        self.val_x_reco_concat = np.concatenate(self.val_x_reco)
        self.val_mask_concat = np.concatenate(self.val_mask)
        self.val_labels_concat = np.concatenate(self.val_labels)
        self.val_code_idx_concat = np.concatenate(self.val_code_idx)
    """

    """
    def on_test_epoch_end(self):
        self.test_x_original_concat = np.concatenate(self.test_x_original)
        self.test_x_reco_concat = np.concatenate(self.test_x_reco)
        self.test_mask_concat = np.concatenate(self.test_mask)
        self.test_labels_concat = np.concatenate(self.test_labels)
        self.test_code_idx_concat = np.concatenate(self.test_code_idx)
    """

