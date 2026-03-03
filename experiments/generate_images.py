import hydra
import torch
from omegaconf import DictConfig
from loguru import logger

from utils.datasets_utils import get_dataset
from tools.run_infinity import *
from utils.experiment_utils import (
    ModelSteerer,
)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):

    torch.set_grad_enabled(False)

    logger.info("[Loading dataset]")
    _, test_dataset = get_dataset(config=cfg.experiment)
    logger.info("<Finished loading dataset>")
    logger.info("[Loading the model]")
    steerer: ModelSteerer = hydra.utils.instantiate(cfg.model)
    steerer.setup_model()
    logger.info("<Finished loading the model>")

    logger.info("[Generating images]")
    steerer.generate_no_steering(
        dataset=test_dataset,
        batch_size=cfg.experiment.BATCH_SIZE,
        save_folder=cfg.save_folder,
    )
    logger.info("<Finished generating images>")

    with open(f"{cfg.save_folder}/prompts_used.txt", "w") as f:
        f.write("\n".join(test_dataset))


if __name__ == "__main__":
    main()
