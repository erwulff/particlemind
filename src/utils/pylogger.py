import logging
import os

import colorlog

from lightning.pytorch.utilities import rank_zero_only


def get_pylogger_with_rank(name=__name__, rank=None) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Default is __name__.
    rank : int, optional
        Rank of the current process. If not provided, it will be retrieved from
        torch.distributed.get_rank().

    Returns
    -------
    logging.Logger
        Logger object.
    """
    if rank is None:
        rank = "unknown"
    rank_string = f"rank:{rank}"

    hostname = os.getenv("HOSTNAME", default="unknown-host")

    # logging.basicConfig(level="INFO", datefmt="[%X]"])

    logger = logging.getLogger(f"{hostname}|{rank_string}|{name}")

    # reset the logging basic config to avoid duplicated logs
    # logging.basicConfig(level="INFO", datefmt="[%X]"])

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def configure_root_logger():
    """
    Configures the root logger for the application.
    This function sets the logging level to INFO and adds a stream handler
    that outputs log messages to the console. The log messages are formatted
    with colors for better visibility, including timestamps, logger names,
    and log levels.
    The log colors are defined as follows:
        - DEBUG: cyan
        - INFO: green
        - WARNING: yellow
        - ERROR: red
        - CRITICAL: bold red
    Example:
        >>> configure_root_logger()
    """

    root_logger = logging.getLogger("root")
    root_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )

    root_logger.addHandler(handler)


def get_pylogger(name=__name__, rank_zero_only=False) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Default is __name__.
    rank_zero_only : bool, optional
        If True, only the rank zero process will log messages. Default is False.
    Returns
    -------
    logging.Logger
        Logger object.
    """
    hostname = os.getenv("HOSTNAME", default="unknown-host")

    logger = logging.getLogger(f"{hostname}|{name}")
    logger.setLevel(logging.INFO)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    if rank_zero_only:
        logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
        for level in logging_levels:
            setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
