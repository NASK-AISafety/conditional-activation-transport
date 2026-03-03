import os
import psutil
from enum import Enum
from typing import Tuple
from abc import ABC
from omegaconf import DictConfig

import torch
import hydra
import pandas as pd

from steering.methods import (
    LinearTransportSteering,
    MeanSteering,
    SteeringRegistry,
)
from steering.training import (
    MLPTransportSteering,
    AffineTransportSteering,
)

STEERING_METHODS = {
    "MEAN": MeanSteering,
    "LINEAR_TRANSPORT": LinearTransportSteering,
    "MLP_TRANSPORT": MLPTransportSteering,
    "AFFINE_TRANSPORT": AffineTransportSteering,
}


class Steering(Enum):
    TEXT = 1
    VISUAL = 2
    BOTH = 3

    def __str__(self) -> str:
        return self.name


Location = Tuple[int, int]


class ModelSteerer(ABC):
    def generate_steered_images(self):
        raise NotImplementedError

    def get_text_activations(self):
        raise NotImplementedError

    def get_visual_activations(self):
        raise NotImplementedError


def print_vram() -> None:
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Allocated: {allocated:.1f} MB")
    print(f"Reserved:  {reserved:.1f} MB")
    print(f"Max alloc: {max_alloc:.1f} MB")


def log_rss(tag: str = "") -> None:
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss
    print(f"[RSS]{tag} {rss/1024**2:.1f}MB")


def create_steered_model(
    cfg: DictConfig,
    train_dataset: pd.DataFrame,
    steering_registry: SteeringRegistry,
    steering_type: Steering,
) -> ModelSteerer:
    steerer: ModelSteerer = hydra.utils.instantiate(cfg.model)
    steerer.setup_model()

    if steering_type in (Steering.TEXT, Steering.BOTH):
        with torch.no_grad():
            steerer.get_text_activations_cached(
                contrastive_dataset=train_dataset,
                out_dir=cfg.cache,
                layers_to_capture=cfg.steering.LAYERS_TEXT,
                safe_class=cfg.experiment.SAFE_CLASS,
                batch_size=cfg.experiment.BATCH_SIZE,
                remove_files_if_exists=cfg.clear_cache,
            )
        steering_registry.build_text(cfg.steering.LAYERS_TEXT)
        (
            steering_registry.train_text(**cfg.steering.TRAIN_KWARGS)
            if cfg.steering.TRAIN_KWARGS
            else steering_registry.train_text()
        )

    if steering_type in (Steering.VISUAL, Steering.BOTH):
        with torch.no_grad():
            steerer.get_model().configure_activation_capture()
            steerer.get_visual_activations_cached(
                contrastive_dataset=train_dataset,
                out_dir=cfg.cache,
                layers_to_capture=cfg.steering.LAYERS_VISION,
                batch_size=cfg.experiment.BATCH_SIZE,
                safe_class=cfg.experiment.SAFE_CLASS,
                remove_files_if_exists=cfg.clear_cache,
            )
        steering_registry.build_vision(
            steerer.params.num_steps, cfg.steering.LAYERS_VISION
        )
        (
            steering_registry.train_vision(
                steerer.params.num_steps,
                cfg.steering.LAYERS_VISION,
                **cfg.steering.TRAIN_KWARGS,
            )
            if cfg.steering.TRAIN_KWARGS
            else steering_registry.train_vision(
                steerer.params.num_steps,
                cfg.steering.LAYERS_VISION,
            )
        )
        steerer.get_model().reset_activation_capture()

    return steerer


def create_steering_registry(cfg: DictConfig) -> SteeringRegistry:
    steeringClass = STEERING_METHODS[cfg.steering.METHOD]
    return SteeringRegistry(
        steering_method=steeringClass,
        steering_condition=cfg.conditioning.CONDITIONING,
        steering_mode=cfg.steering.MODE,
        cache_path=cfg.cache,
        categories=cfg.experiment.TRAIN_CATEGORIES,
        conditioner_kwargs=cfg.conditioning.CONDITIONER_KWARGS,
    )


def select_steering_type(cfg: DictConfig) -> Steering:
    if cfg.steering.LAYERS_TEXT is not None:
        if cfg.steering.LAYERS_VISION is not None:
            steering_type = Steering.BOTH
        else:
            steering_type = Steering.TEXT
    else:
        if cfg.steering.LAYERS_VISION is None:
            raise ValueError("At least one steering type must be specified")
        steering_type = Steering.VISUAL
    return steering_type
