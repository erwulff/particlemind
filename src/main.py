import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.pylogger import configure_root_logger


@hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
def main_hydra_decorated(cfg: DictConfig):
    configure_root_logger()
    # lazy import to speed up initial loading e.g. when doing python -m src.main --help
    from src.train import train

    train(cfg)


def main():
    """
    Main function to run the training script with Hydra configuration.
    This function is a workaround to use Hydra's configuration management
    without wrapping the main training code in the hydra.main decorator.
    It allows for command-line overrides and configuration loading.
    """
    configure_root_logger()

    cfg_holder = []

    @hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
    def parse(cfg):
        OmegaConf.resolve(cfg)  # If you need resolving, it needs to be done here
        cfg_holder.append(cfg)

    parse()
    cfg = cfg_holder[0]

    # Run main code
    from src.train import train  # lazy import to speed up initial loading

    train(cfg)


if __name__ == "__main__":
    main_hydra_decorated()  # replace with main() if you want to run without @hydra.main
