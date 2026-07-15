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
from src.models.plotting import plot_model_hit, plot_model_patch
from src.models.base_components import VQVAEMLP, VQVAENormFormer, reco_loss_function, mean_knn_distance

from src.models.contrastive_losses import CLIP_loss

# # vqtorch can be installed from https://github.com/minyoungg/vqtorch
# try:
#     from vqtorch.nn import VectorQuant  # type: ignore
# except ImportError as e:
#     raise ImportError("vqtorch is not installed. Please install it to use this module.") from e

from src.utils.arrays import (
    ak_pad,
    ak_select_and_preprocess,
    ak_to_np_stack,
    np_to_ak,
)

vector.register_awkward()

logger = logging.getLogger(__name__)



class VQVAELightningDouble(L.LightningModule):
    """PyTorch Lightning module for training a VQ-VAE."""

    def __init__(
        self,
        data_type, # patch or hits
        optimizer_kwargs={},
        lr_scheduler_kwargs = {"use_scheduler":False},
        model_kwargs={},
        loss_type="mse",
        vit_kwargs=None,
        num_train_events=0,
        batch_size_per_gpu=0,
        plot_dir_name="",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)


        if model_kwargs["model_type"] == "mlp":
            self.model_HCAL = VQVAEMLP(**model_kwargs, vit_kwargs=vit_kwargs, data_type=data_type,)
            self.model_ECAL = VQVAEMLP(**model_kwargs, vit_kwargs=vit_kwargs, data_type=data_type,)
        elif model_kwargs["model_type"] == "vae":
            self.model_HCAL = VQVAENormFormer(**model_kwargs, vit_kwargs=vit_kwargs, data_type=data_type,)
            self.model_ECAL = VQVAENormFormer(**model_kwargs, vit_kwargs=vit_kwargs, data_type=data_type,)
        
        self.train_loss_history = []
        self.val_loss_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        # for tracking best so far validation accuracy
        self.val_x_original = []
        self.val_x_reco = []
        self.val_mask = []

        self.num_train_events = num_train_events
        self.batch_size_per_gpu = batch_size_per_gpu

        self.plot_dir_name = plot_dir_name
        self.data_type = data_type

        self.use_vq = model_kwargs["vq_kwargs"]
        self.loss_type = loss_type

    def configure_optimizers(self):
        return configure_optimizers_base(self)
    
    def model_step(self, batch, return_x=False):
        """Perform a single model step on a batch of data."""
    
        alpha = self.hparams["model_kwargs"]["alpha"]
        shared_weight = self.hparams["model_kwargs"]["shared_weight"]
        aug_weight = self.hparams["model_kwargs"]["aug_weight"]
    
        if self.data_type == "hit":
    
            x_hit_ECAL = batch["calo_hit_features_ECAL"]
            mask_ECAL = batch["mask_ECAL"]
            x_hit_HCAL = batch["calo_hit_features_HCAL"]
            mask_HCAL = batch["mask_HCAL"]
            
            loss = 0

            if aug_weight != 0:
                x_hit_ECAL_augmented = batch["calo_hit_features_ECAL_augmented"]
                mask_ECAL_augmented = batch["mask_ECAL_augmented"]
                x_hit_HCAL_augmented = batch["calo_hit_features_HCAL_augmented"]
                mask_HCAL_augmented = batch["mask_HCAL_augmented"]
                _, _, z_embed_ECAL_augmented = self.model_ECAL(None, x_hit_ECAL_augmented, mask_ECAL_augmented)
                _, _, z_embed_HCAL_augmented = self.model_HCAL(None, x_hit_HCAL_augmented, mask_HCAL_augmented)
              

            x_hit_ECAL_reco, vq_out, z_embed_ECAL = self.model_ECAL(None, x_hit_ECAL, mask_ECAL) # batch not used
            x_hit_HCAL_reco, vq_out, z_embed_HCAL = self.model_HCAL(None, x_hit_HCAL, mask_HCAL) # batch not used

    
            reco_loss_ECAL = reco_loss_function(x_hit_ECAL, x_hit_ECAL_reco, mask_ECAL)
            loss += reco_loss_ECAL
            
            x1 = F.normalize(x_hit_ECAL, dim=-1, eps=1e-8)       # [2, # hits, 4]
            x2 = F.normalize(x_hit_ECAL_reco, dim=-1, eps=1e-8)  # [2, # hits, 4]
            cos_sim_ECAL = (
                ((x1 * x2).sum(dim=-1) * mask_ECAL).sum()
                / mask_ECAL.sum().clamp(min=1)
            )

            reco_loss_HCAL = reco_loss_function(x_hit_HCAL, x_hit_HCAL_reco, mask_HCAL)
            loss += reco_loss_HCAL
            x1 = F.normalize(x_hit_HCAL, dim=-1, eps=1e-8)       # [2, # hits, 4]
            x2 = F.normalize(x_hit_HCAL_reco, dim=-1, eps=1e-8)  # [2, # hits, 4]
            cos_sim_HCAL = (
                ((x1 * x2).sum(dim=-1) * mask_HCAL).sum()
                / mask_HCAL.sum().clamp(min=1)
            )

            loss_dict = {
                "reco_loss_ECAL": reco_loss_ECAL,
                "reco_loss_HCAL": reco_loss_HCAL,
                "cosine_similarity_ECAL": cos_sim_ECAL,
                "cosine_similarity_HCAL": cos_sim_HCAL,
            }

            # shared loss
            shared_loss = CLIP_loss(z_embed_ECAL, z_embed_HCAL, mask_ECAL, mask_HCAL)
            loss += shared_weight * shared_loss
            loss_dict["shared_loss"] = shared_loss

            truth_knn_ECAL = mean_knn_distance(x_hit_ECAL, mask_ECAL)
            reco_knn_ECAL = mean_knn_distance(x_hit_ECAL_reco, mask_ECAL)
            truth_knn_HCAL = mean_knn_distance(x_hit_HCAL, mask_HCAL)
            reco_knn_HCAL = mean_knn_distance(x_hit_HCAL_reco, mask_HCAL)

            knn_error_ECAL = torch.abs(truth_knn_ECAL - reco_knn_ECAL)
            knn_error_HCAL = torch.abs(truth_knn_HCAL - reco_knn_HCAL)

            loss_dict["knn_error_ECAL"] = knn_error_ECAL
            loss_dict["knn_error_HCAL"] = knn_error_HCAL
  
            if aug_weight != 0:
                ECAL_aug_loss = CLIP_loss(z_embed_ECAL, z_embed_ECAL_augmented, mask_ECAL, mask_ECAL_augmented)
                HCAL_aug_loss = CLIP_loss(z_embed_HCAL, z_embed_HCAL_augmented, mask_HCAL, mask_HCAL_augmented)
                loss += aug_weight * (ECAL_aug_loss + HCAL_aug_loss)
                
                loss_dict["aug_loss_ECAL"] = ECAL_aug_loss
                loss_dict["aug_loss_HCAL"] = HCAL_aug_loss
                
            if self.use_vq:
                cmt_loss = vq_out["loss"]
                code_idx = vq_out["q"]
                loss +=  alpha * cmt_loss
                loss_dict["cmt_loss"] = cmt_loss
            else:
                code_idx = None


            x_hit = torch.cat([
                x_hit_ECAL,
                x_hit_HCAL
            ], dim=1)
            
            x_hit_reco = torch.cat([
                x_hit_ECAL_reco,
                x_hit_HCAL_reco
            ], dim=1)
            
            mask_particle = torch.cat([
                mask_ECAL,
                mask_HCAL
            ], dim=1)
            
            labels = torch.cat([
                batch["labels_ECAL"],
                batch["labels_HCAL"]
            ], dim=1)
            

            loss_dict["total_loss"] = loss

            if return_x:
                return loss_dict, x_hit, x_hit_reco, mask_particle, labels, code_idx
    
            return loss_dict
    
        elif self.data_type == "patch":
    
            embedding_hit, embedding_hit_reco, patches_chunked, patches_chunked_reco, vq_out = self.forward(batch, None, None) # x, mask not used
            
            
            losses = []
            for key in patches_chunked.keys():

                mask = batch[key]["mask"].unsqueeze(-1)
                if mask.sum() > 0:
                    diff = (patches_chunked[key] - patches_chunked_reco[key]) ** 2
                    losses.append((diff * mask).sum() / mask.sum())

            reco_loss = torch.stack(losses).mean()

            loss = reco_loss

            loss_dict = {"reco_loss": reco_loss}

            if self.use_vq:
                cmt_loss = vq_out["loss"]
                loss += alpha * cmt_loss
                loss_dict["cmt_loss"] = cmt_loss

            loss_dict["total_loss"] = loss
                
    
            if return_x:
                return loss_dict, embedding_hit, embedding_hit_reco, batch, patches_chunked_reco, vq_out
    
            return loss_dict

  

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss_dict = self.model_step(batch)

        self.train_loss_history.append(loss_dict["total_loss"].detach().cpu().numpy())
        for loss_type in loss_dict.keys():
            self.log(
                    f"train/{loss_type}",
                    loss_dict[loss_type],               # <-- pass the tensor, not loss.item()
                    on_step=True,
                    on_epoch=True,       # optional if you also want epoch avg
                    prog_bar=True,
                    sync_dist=True       # sync across GPUs
                )

        return loss_dict["total_loss"]

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

    

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:

        if self.data_type == "patch":
            loss_dict, embedding_hit, embedding_hit_reco, batch, patches_chunked_reco, vq_out = self.model_step(batch, return_x=True)

        elif self.data_type == "hit":
            loss_dict, x_original, x_reco, mask, labels, code_idx = self.model_step(batch, return_x=True)


        
        for loss_type in loss_dict.keys():
            self.log(
                f"val/{loss_type}",
                loss_dict[loss_type],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True
            )


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

            if self.data_type == "patch":
                # log the plot
                plot_model_patch(
                    batch=batch,
                    patches_chunked_reco=patches_chunked_reco,
                    vq_out=vq_out,
                    num_codes=self.model.vq_kwargs["num_codes"] if vq_out is not None else None,
                    device=self.device,
                    saveas=plot_filename,
                )
            
            elif self.data_type == "hit":
                try:
                     plot_model_hit(
                         input_data=x_original, 
                         reco=x_reco,
                         labels=labels, 
                         masks=mask,
                         device=self.device,
                         saveas=plot_filename
                         )
                except Exception as e:
                    print(f"Skipping plots: {e}")

            
            if comet_logger is not None:
                comet_logger.log_image(plot_filename, name=plot_filename.split("/")[-1], step=curr_step)

        return loss_dict["total_loss"]

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
                x_hit_reco, vq_out = self.forward(features_batch, mask_batch)
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


