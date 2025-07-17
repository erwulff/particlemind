import os
from argparse import ArgumentParser

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from data.iterable_dataset_hitclass import IterableDatamodule
from models.vqvae import VQVAELightning

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

    if args.train_classifier:
        checkpoint_acc = ModelCheckpoint(dirpath = f"{args.save_dir}/{args.project}/best_models/", filename=f"{args.name}_val_acc_"+"{epoch:02d}", monitor="acc/val", mode="max", verbose=1, auto_insert_metric_name=True)
        callbacks = [checkpoint_acc, checkpoint_loss, lr_monitor]
    
    


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
    data_dir = "/pscratch/sd/r/rmastand/particlemind/data/p8_ee_tt_ecm365_rootfiles/"
    feature_dict = {
        "type": {},
        "energy": {},
        "position.x": {},
        "position.y": {},
        "position.z": {},
    }
    data_indices_train = [62323, 62333, 62343]
    data_indices_val = [62353, 62363]
    data_indices_test = [62373]

    dataset_kwargs_train = {
        "files_list": [f"{data_dir}/reco_p8_ee_tt_ecm365_{ind}.root" for ind in data_indices_train],
        }
    dataset_kwargs_val = {
        "files_list": [f"{data_dir}/reco_p8_ee_tt_ecm365_{ind}.root" for ind in data_indices_val],
        }
    dataset_kwargs_test = {
        "files_list": [f"{data_dir}/reco_p8_ee_tt_ecm365_{ind}.root" for ind in data_indices_test],
    }
    
    dataset_kwargs_common = {
        "pad_length": 20000,
        "feature_dict": feature_dict,
       }
    

    dataModule = IterableDatamodule( 
                            dataset_kwargs_train = dataset_kwargs_train, 
                            dataset_kwargs_val = dataset_kwargs_val, 
                            dataset_kwargs_test = dataset_kwargs_test, 
                            dataset_kwargs_common = dataset_kwargs_common,
                            batch_size= 256,
                            )
    dataModule.setup(stage="fit")
    print(dataModule.train_dataloader)
    

    ### MODEL
    model = VQVAELightning(
        optimizer = torch.optim.AdamW,
        scheduler = None,
        model_kwargs={
                    "input_dim":5,
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
                        }
                   },
        model_type="VQVAENormFormer",
       
    )

    if bool(args.test_phase):
        trainer.test(model, data.test_dataloader())
    else:
        trainer.fit(model, dataModule)
        if args.train_classifier:
            trainer.test(model, data.test_dataloader())


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/global/cfs/cdirs/m3246/rmastand/polymathic/cifar10/")
    parser.add_argument("--save_dir", type=str, default="/global/cfs/cdirs/m3246/rmastand/polymathic/")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument( "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18", choices=["resnet18", "densenet1d"])
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # OTHER ARGS
    parser.add_argument("--project", type=str)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--train_classifier", action="store_true")
    parser.add_argument("--use_embedding_space", action="store_true")
    parser.add_argument("--path_to_embedding_network", type=str, default="/global/cfs/cdirs/m3246/rmastand/polymathic/extractor/best_models/val_loss-v2.ckpt")

    parser.add_argument("--mlp", type=str, default="4096-4096-4096")
    parser.add_argument("--sim-coeff", type=float, default=1.0, help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=1.0, help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0, help='Covariance regularization loss coefficient')


    args = parser.parse_args()
    main(args)
