import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from loguru import logger

from utils.datasets_utils import get_dataset
from utils.experiment_utils import (
    create_steered_model,
    create_steering_registry,
    select_steering_type,
)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    logger.info("[Loading dataset]")
    train_dataset, test_dataset = get_dataset(config=cfg.experiment)

    steering_type = select_steering_type(cfg)

    steering_registry = create_steering_registry(cfg)

    logger.info("[Loading the model]")
    steerer = create_steered_model(cfg, train_dataset, steering_registry, steering_type)

    with open(f"{cfg.save_folder}/prompts_used.txt", "w") as f:
        f.write("\n".join(test_dataset))

    with torch.no_grad():
        for strength in tqdm(cfg.steering.STRENGTHS, desc="Generating steered images"):
            if cfg.steering.STEPS_VISION is not None:
                all_steps = cfg.steering.STEPS_VISION
                for steps in all_steps:
                    save_steps = str(steps).strip("[]").replace(", ", "-")
                    steerer.generate_steered_images(
                        dataset=test_dataset,
                        registry=steering_registry,
                        steering_type=steering_type,
                        strength=strength,
                        steps=steps,
                        text_layers=cfg.steering.LAYERS_TEXT,
                        vision_layers=cfg.steering.LAYERS_VISION,
                        batch_size=cfg.experiment.BATCH_SIZE,
                        save_folder=f"{cfg.save_folder}/steering_strength_{strength}_steps_{save_steps}",
                    )
            else:
                steerer.generate_steered_images(
                    dataset=test_dataset,
                    registry=steering_registry,
                    steering_type=steering_type,
                    strength=strength,
                    steps=[],
                    text_layers=cfg.steering.LAYERS_TEXT,
                    vision_layers=cfg.steering.LAYERS_VISION,
                    batch_size=cfg.experiment.BATCH_SIZE,
                    save_folder=f"{cfg.save_folder}/steering_strength_{strength}",
                )


if __name__ == "__main__":
    main()
