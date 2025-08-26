import lightning.pytorch as pl
from torch.utils.data import DataLoader

from src import models as src_models

from src.datasets.CLDHits import CLDHits
from src.datasets.utils import Collater
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def get_ml_loggers(ml_logger):
    """
    Initialize and return a list of PyTorch Lightning loggers based on the configuration.
    Args:
        ml_logger (dict): A dictionary containing logger configurations.
    """
    loggers = []
    for key in ml_logger.keys():
        if key not in ["wandb", "tensorboard", "csv"]:
            raise ValueError(f"Unsupported logger type: {key}")
        if key == "wandb":
            loggers.append(pl.loggers.WandbLogger(**ml_logger[key]))
        elif key == "tensorboard":
            loggers.append(pl.loggers.TensorBoardLogger(**ml_logger[key]))
        elif key == "csv":
            loggers.append(pl.loggers.CSVLogger(**ml_logger[key]))
    return loggers


def train(cfg):
    loggers = get_ml_loggers(cfg.ml_logger)

    data_dir = cfg.data.data_dir
    batch_size = cfg.data.batch_size

    # Data loaders
    train_dataset = CLDHits(data_dir, "train")
    val_dataset = CLDHits(data_dir, "val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=Collater("all"), num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=Collater("all"), num_workers=8)

    try:
        model_class_name = cfg.model.model_name
        lightning_module_class = getattr(src_models, model_class_name)
    except AttributeError as e:
        log.error(f"Could not find {model_class_name} in src.models")
        raise e

    model = lightning_module_class(
        optimizer_kwargs=cfg.model.optimizer_kwargs,
        model_kwargs=cfg.model.model_kwargs,
        model_type=cfg.model.model_type,
    )
    log.info("Model initialized with the following parameters: %s", model.model.__dict__)

    # Count the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Number of trainable parameters: {num_params}")

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=loggers,
    )

    log.info("Trainer initialized with the following configuration: %s", trainer.__dict__)

    log.info("Starting training... Global rank: %s", trainer.global_rank)
    trainer.fit(model, train_loader, val_loader)
    log.info("Training complete. Global rank: %s", trainer.global_rank)

    log.info("Testing the model... Global rank: %s", trainer.global_rank)
    trainer.test(model, val_loader)  # TODO: Use a separate test dataset if available
    log.info("Testing complete. Global rank: %s", trainer.global_rank)
