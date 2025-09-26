import os
from argparse import ArgumentParser

import torch
torch.cuda.empty_cache()
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.fabric.utilities.rank_zero import rank_zero_only

from src.datasets.CLDHits import CLDHits
from src.datasets.utils import Collater
from src.models.vae import VAELightning, SSLLightning
from src.models.vqvae import VQVAELightning


@rank_zero_only
def log_config(logger, args):
    if hasattr(logger, "experiment") and hasattr(logger.experiment, "config"):
        logger.experiment.config.update(vars(args))


def main(args):

    seed_everything(0)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WANDB_CACHE_DIR"] = "/pscratch/sd/r/rmastand/"

    if args.logger == "wandb":
        logger = WandbLogger(
            name=args.name, project=args.project, save_dir=f"{args.save_dir}/{args.project}/", log_model="all"
        )
        log_config(logger, args)
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger(args.data_dir, name=args.name)

    if args.train_embedder:
        filename = f"embedder_{args.name}_val_loss_" + "{epoch:02d}"
    else:
        filename = f"projector_{args.name}_val_loss_" + "{epoch:02d}"
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_loss = ModelCheckpoint(
        dirpath=f"{args.save_dir}/{args.project}/best_models/",
        filename=filename,
        monitor="val_loss_epoch",
        mode="min",
        verbose=1,
        auto_insert_metric_name=True,
    )
    callbacks = [checkpoint_loss, lr_monitor]

    trainer = Trainer(
        logger=logger,
        devices=1,
        accelerator="cuda",
        strategy="ddp_find_unused_parameters_true",
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        precision=args.precision,
        default_root_dir=f"{args.save_dir}/{args.project}/",
        limit_train_batches=3000,
        limit_val_batches=300,

    )

    # DATA
    train_dataset = CLDHits(args.data_dir, "train", nfiles=args.num_files, by_event=True, shuffle_files=True)
    val_dataset = CLDHits(args.data_dir, "val", nfiles=args.num_files, by_event=True, shuffle_files=False)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=Collater("all"), num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=Collater("all"), num_workers=2)

    # MODEL
    if args.train_embedder:
        model = VQVAELightning(
            optimizer_kwargs={"lr": args.learning_rate, "weight_decay": args.weight_decay},
            lr_scheduler_kwargs = {"use_scheduler": True, "warmup_frac": 0.01},
            model_kwargs={
                "input_dim": 4,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_blocks": args.num_blocks,
                "alpha": args.alpha,
                "vq_kwargs": {
                    "num_codes": args.num_codes,
                    "beta": args.beta,
                    "kmeans_init": args.kmeans_init,
                   # "norm": "null",
                   #  "cb_norm": "null",
                    "affine_lr": args.affine_lr,
                    "sync_nu": args.sync_nu,
                    "replace_freq": args.replace_freq,
                    "dim": -1,
                },
             
            },
            model_type="VQVAENormFormer",
        )

    else:

        #load in pretrained embedder
        embedder = VQVAELightning.load_from_checkpoint(
            checkpoint_path="/pscratch/sd/r/rmastand/particlemind/vqvae_training/best_models/embedder_test_val_loss_epoch=03.ckpt",
        )
        
        model = SSLLightning(
            embedding_model = embedder.model,
            optimizer_kwargs={"lr": args.learning_rate, "weight_decay": args.weight_decay},
            lr_scheduler_kwargs = {"use_scheduler": True, "warmup_frac": 0.01},

            projector_kwargs={
                "activation": "relu",
                "nodes":[32, 32, 18],
               }
                     )

    


    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM ARGS
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32])

    parser.add_argument("--save_dir", type=str, default="/pscratch/sd/r/rmastand/particlemind/")
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--project", type=str, default="vqvae_training")
    parser.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb"])

    # DATA ARGS
    parser.add_argument(
        "--data_dir", type=str, default="/pscratch/sd/r/rmastand/particlemind/data/p8_ee_tt_ecm365_parquetfiles"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulate_grad_batches", type=int, default=128)
    parser.add_argument("--num_files", type=int, default=30)

    # TRAINER ARGS
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--train_embedder", action="store_true", default=False) # else train projector

    # MODEL args
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--num_codes", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--affine_lr", type=float, default=0.0)
    parser.add_argument("--sync_nu", type=int, default=2)
    parser.add_argument("--replace_freq", type=int, default=20)

    args = parser.parse_args()
    main(args)
