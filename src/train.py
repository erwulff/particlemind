import argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import logging


from src.models.self_attention_transformer import SelfAttentionTransformer
from models.losses import transformer_instance_loss, discriminative_loss
from datasets.CLDHits import CLDHits
from datasets.utils import Collater


class TransformerLightningModule(pl.LightningModule):
    def __init__(self, model, loss_function, lr):
        super(TransformerLightningModule, self).__init__()
        self.model = model
        self.learning_rate = lr
        self.loss_function = loss_function

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        x = batch["calo_hit_features"]
        y = batch["hit_labels"]

        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        # Logging
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["calo_hit_features"]
        y = batch["hit_labels"]

        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        # Logging
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x = batch["calo_hit_features"]
        y = batch["hit_labels"]

        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        # Logging
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer]  # , [scheduler]


def get_loss_func(name):
    if name == "instance":
        loss_function = transformer_instance_loss
    elif name == "discriminative":
        loss_function = discriminative_loss
    return loss_function


def train(
    input_size=4,
    output_size=10,
    embed_dim=64,
    num_layers=4,
    num_heads=8,
    ff_dim=128,
    dropout=0.1,
    lr=1e-3,
    batch_size=32,
    num_epochs=20,
    data_dir="data/CLDHits",
    gpus=0,
    loss_function="discriminative",
    ntrain=1.0,
    nval=1.0,
):

    wandb_logger = pl.loggers.WandbLogger(
        project="test_project",
        name=f"transformer_{input_size}_{output_size}_{embed_dim}_{num_layers}_{num_heads}_{ff_dim}",
        log_model=False,
        config={
            "input_size": input_size,
            "output_size": output_size,
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "data_dir": data_dir,
            "gpus": gpus,
            "loss_function": loss_function,
        },
    )

    # Data loaders
    train_dataset = CLDHits(data_dir, "train")
    val_dataset = CLDHits(data_dir, "val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=Collater("all"), num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=Collater("all"), num_workers=8)

    model = SelfAttentionTransformer(
        input_size=input_size,
        output_size=output_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
    )

    # Count the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of trainable parameters: {num_params}")

    loss_function = get_loss_func(loss_function)
    model = TransformerLightningModule(model, loss_function, lr)

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        num_nodes=1,
        devices=gpus,
        limit_train_batches=ntrain,
        limit_val_batches=nval,
        accelerator="gpu" if gpus > 0 else "cpu",
        strategy="ddp" if gpus > 1 else None,
        logger=wandb_logger,
    )

    logging.info("Trainer initialized with the following configuration: %s", trainer.__dict__)

    logging.info("Starting training... Global rank: %s", trainer.global_rank)
    trainer.fit(model, train_loader, val_loader)
    logging.info("Training complete. Global rank: %s", trainer.global_rank)

    logging.info("Testing the model... Global rank: %s", trainer.global_rank)
    trainer.test(model, val_loader)  # TODO: Use a separate test dataset if available
    logging.info("Testing complete. Global rank: %s", trainer.global_rank)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Model")
    parser.add_argument("--input_size", type=int, default=4, help="Input size of the model")
    parser.add_argument("--output_size", type=int, default=10, help="Output size of the model")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=128, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--data_dir", type=str, default="data/CLDHits", help="Directory for the dataset")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use")
    parser.add_argument("--ntrain", type=float, default=1.0, help="Fraction of training data to use (1.0 for all)")
    parser.add_argument("--nval", type=float, default=1.0, help="Fraction of validation data to use (1.0 for all)")
    return parser.parse_args()


def main():
    # configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = parse_args()

    if args.ntrain > 1.0:  # If ntrain is greater than 1, treat it as an absolute number of samples
        args.ntrain = int(args.ntrain)
    if args.nval > 1.0:
        args.nval = int(args.nval)

    train(**vars(args))


if __name__ == "__main__":
    main()
