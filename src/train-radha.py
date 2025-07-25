import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger


from src.datasets.CLDHits import CLDHits
from src.datasets.utils import Collater
from src.models.vqvae import VQVAELightning

def main(args):

    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["WANDB_CACHE_DIR"] = "/pscratch/sd/r/rmastand/"


    if args.logger == "wandb":
        logger = WandbLogger(name=args.name, project=args.project, save_dir = f"{args.save_dir}/{args.project}/", log_model="all")
        logger.experiment.config.update(args)
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger(args.data_dir, name=args.name)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_loss = ModelCheckpoint(dirpath = f"{args.save_dir}/{args.project}/best_models/", filename=f"{args.name}_val_loss_"+"{epoch:02d}", monitor="loss/val", mode="min", verbose=1, auto_insert_metric_name=True)
    callbacks = [checkpoint_loss, lr_monitor]



    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        logger=logger if not bool(args.dev + args.test_phase) else None,
        devices="auto",
        accelerator="cuda",
        deterministic=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        callbacks = callbacks,
        precision=args.precision,
        default_root_dir=f"{args.save_dir}/{args.project}/"
    )

    ### DATA
    train_dataset = CLDHits(args.data_dir, "train")
    val_dataset = CLDHits(args.data_dir, "val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=Collater("all"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=Collater("all"))



    ### MODEL
    model = VQVAELightning(
        optimizer = torch.optim.AdamW,
        scheduler = None,
        model_kwargs={
                    "input_dim":4,
                    "latent_dim":3,
                    "hidden_dim":128,
                    "num_heads":1,
                    "num_blocks":2,
                    "vq_kwargs":{
                        "num_codes": 512,
                    "beta": 0.9,
                    "kmeans_init": True,
                 #   "norm": "null",
                   # "cb_norm": "null",
                    "affine_lr": 2,
                    "sync_nu": 1,
                    "replace_freq": 100,
                    "dim": -1
                        }
                   },
        model_type="VQVAENormFormer",
       
    )

    trainer.fit(model, train_loader, val_loader)
   

if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/pscratch/sd/r/rmastand/particlemind/data/p8_ee_tt_ecm365_parquetfiles")
    parser.add_argument("--save_dir", type=str, default="/global/cfs/cdirs/m3246/rmastand/polymathic/")
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--project", type=str, default="test")
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument( "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18", choices=["resnet18", "densenet1d"])
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)



    args = parser.parse_args()
    main(args)