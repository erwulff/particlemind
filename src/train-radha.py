import os
from argparse import ArgumentParser

import torch
import yaml

torch.cuda.empty_cache()
from lightning import Trainer, seed_everything
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from src.datasets.CLDHits import CLDHits, CLDHitsSingleFile
from src.datasets.Tokens import Tokens
from src.datasets.utils import Collater
from src.models.backbone import BackboneNextTokenPredictionLightning

# from src.models.vae import VAELightning, SSLLightning
from src.models.vqvae import VQVAELightning
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


@rank_zero_only
def log_config(logger, args):
    if hasattr(logger, "experiment") and hasattr(logger.experiment, "config"):
        logger.experiment.config.update(vars(args))


def main(args):
    if args.train_embedder:
        project = "vqvae_training"
        with open(f"configs/{args.config_embedder}.yaml", "r") as file:
            configs = yaml.safe_load(file)
            print(configs)
            filename = f"embedder_{args.name}_val_loss_" + "{epoch:02d}"

    elif args.generate_tokenized_dataset:
        with open(f"configs/{args.config_tokenizer}.yaml", "r") as file:
            configs = yaml.safe_load(file)
            print(configs)

    elif args.train_tokenizer:
        project = "tokenizer_training"
        with open(f"configs/{args.config_gpt}.yaml", "r") as file:
            configs = yaml.safe_load(file)
            print(configs)
            filename = f"tokenizer_{args.name}_val_loss_" + "{epoch:02d}"

    elif args.generate_samples:
        with open(f"configs/{args.config_tokenizer}.yaml", "r") as file:
            configs = yaml.safe_load(file)
            print(configs)

    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = configs["trainer_kwargs"]["visible_devices"]
    os.environ["WANDB_CACHE_DIR"] = "/pscratch/sd/r/rmastand/"

    #torch.set_float32_matmul_precision("medium")



    if not args.generate_tokenized_dataset:

        if args.logger == "wandb":
            logger = WandbLogger(
                name=args.name,
                project=project,
                save_dir=f"{args.save_dir}/{project}/",
                log_model="all",
            )
            log_config(logger, args)
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger(args.data_dir, name=args.name)
    
            
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_loss = ModelCheckpoint(
            dirpath=f"{args.save_dir}/{project}/best_models/",
            filename=filename,
            monitor="val_loss_epoch",
            mode="min",
            verbose=1,
            auto_insert_metric_name=True,
        )
        callbacks = [checkpoint_loss, lr_monitor]
    
        trainer = Trainer(
            logger=logger,
            devices=len(configs["trainer_kwargs"]["visible_devices"].split(",")),
            accelerator="cuda",
            strategy="ddp_find_unused_parameters_true",
            accumulate_grad_batches=configs["trainer_kwargs"]["accumulate_grad_batches"],
            deterministic=True,
            enable_model_summary=True,
            log_every_n_steps=1,
            max_epochs=configs["trainer_kwargs"]["max_epochs"],
            callbacks=callbacks,
            precision=configs["trainer_kwargs"]["precision"],
            default_root_dir=f"{args.save_dir}/{project}/",
            limit_train_batches=configs["trainer_kwargs"]["limit_train_batches"],
            limit_val_batches=configs["trainer_kwargs"]["limit_val_batches"],
        )

     # MODEL
    if args.train_embedder:

        # DATA
        train_dataset = CLDHits(
            configs["data_kwargs"]["data_dir"],
            "train",
            nfiles=configs["data_kwargs"]["num_files"],
            by_event=True,
            shuffle_files=True,
            train_fraction=configs["data_kwargs"]["train_fraction"],
        )
        val_dataset = CLDHits(
            configs["data_kwargs"]["data_dir"],
            "val",
            nfiles=configs["data_kwargs"]["num_files"],
            by_event=True,
            shuffle_files=False,
            train_fraction=configs["data_kwargs"]["train_fraction"],
        )
    
        train_loader = DataLoader(
            train_dataset,
            batch_size=configs["data_kwargs"]["batch_size"],
            collate_fn=Collater(empty_key="calo_hit_features", variable_size_keys="all"),
            num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=configs["data_kwargs"]["batch_size"],
            collate_fn=Collater(empty_key="calo_hit_features", variable_size_keys="all"),
            num_workers=2,
        )

   
        model = VQVAELightning(
            optimizer_kwargs=configs["optimizer_kwargs"],
            lr_scheduler_kwargs=configs["lr_scheduler_kwargs"],
            model_kwargs=configs["model_kwargs"],
            model_type="VQVAENormFormer",
        )

        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, val_loader)

    if args.generate_tokenized_dataset:

        from pathlib import Path

        import awkward as ak
        import numpy as np

        # load in pretrained embedder
        embedder = VQVAELightning.load_from_checkpoint(
            checkpoint_path=configs["model_kwargs"]["checkpoint_path"],
        )

        # TODO clean this up, it shouldn't just be in the main file

        # get the files
        parquet_files = list(Path(configs["data_kwargs"]["data_dir"]).glob("*.parquet"))

        for file in parquet_files[configs["data_kwargs"]["start_files"]:configs["data_kwargs"]["stop_files"]]:

            print("Analyzing file", file.name)

            file_dataset = CLDHitsSingleFile(file, by_event=True)
            file_loader = DataLoader(
                file_dataset,
                batch_size=configs["data_kwargs"]["batch_size"],
                collate_fn=Collater(empty_key="calo_hit_features", variable_size_keys="all"),
                num_workers=2,
            )

            codes = embedder.tokenize_dataloader(file_loader, add_start_end_tokens=True)

            # Save
            ak.to_parquet(codes, args.codes_dir + "/" + file.name)

            print("Saved out to", args.codes_dir + "/" + file.name)

    if args.train_tokenizer:
        # TODO: DEFINE DATALOADERS
        train_dataset = Tokens(
            configs["data_kwargs"]["data_dir"],
            "train",
            nfiles=configs["data_kwargs"]["num_files"],
            by_event=True,
            shuffle_files=True,
            train_fraction=configs["data_kwargs"]["train_fraction"],
        )
        val_dataset = Tokens(
            configs["data_kwargs"]["data_dir"],
            "val",
            nfilesconfigs["data_kwargs"]["num_files"],
            by_event=True,
            shuffle_files=False,
            train_fraction=configs["data_kwargs"]["train_fraction"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=configs["data_kwargs"]["batch_size"],
            collate_fn=Collater(empty_key="token_features", variable_size_keys="all"),
            num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=configs["data_kwargs"]["batch_size"],
            collate_fn=Collater(empty_key="token_features", variable_size_keys="all"),
            num_workers=2,
        )

        # train the generative model backbone
        model = BackboneNextTokenPredictionLightning(
            optimizer_kwargs=configs["optimizer_kwargs"],
            lr_scheduler_kwargs=configs["lr_scheduler_kwargs"],
            model_kwargs=configs["model_kwargs"],
        )

        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, val_loader)

    if args.generate_samples:

        # load in pretrained tokenizer
        tokenizer = BackboneNextTokenPredictionLightning.load_from_checkpoint(
            checkpoint_path=configs["model_kwargs"]["tokenizer_checkpoint_path"],
        )

        # load in pretrained embedder
        embedder = VQVAELightning.load_from_checkpoint(
            checkpoint_path=configs["model_kwargs"]["checkpoint_path"],
        )

        samples_token = tokenizer.generate_n_jets_batched(configs["data_kwargs"]["n_events"], configs["data_kwargs"]["batch_size"])
        # PRINT, CHECK IF START, STOP TOKCNES ARE THERE
        print(samples_token)
        # save tokens as well
        exit()

        # remove start, stop tokens

        samples_x = embedder.reconstruct_ak_tokens(samples_token, pp_dict, batch_size=configs["data_kwargs"]["n_events"], pad_length=128, hide_pbar=False)

        # TODO REVERSE PREPROCESS
        # save out

    


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM ARGS
    # parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument(
        "--save_dir", type=str, default="/pscratch/sd/r/rmastand/particlemind/"
    )
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument(
        "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
    )

    parser.add_argument("--codes_dir", type=str, default="")

    # VQVAE args
    parser.add_argument("--train_embedder", action="store_true", default=False)
    parser.add_argument("--config_embedder", type=str, default="vqvae")

    parser.add_argument(
        "--generate_tokenized_dataset", action="store_true", default=False
    )
    parser.add_argument("--config_tokenizer", type=str, default="tokenize_dataset")

    # GPT args
    parser.add_argument("--train_tokenizer", action="store_true", default=False)
    parser.add_argument("--config_gpt", type=str, default="gpt")

    parser.add_argument(
        "--generate_samples", action="store_true", default=False
    )
    parser.add_argument("--config_generation", type=str, default="generate_samples")
    

    args = parser.parse_args()
    main(args)
