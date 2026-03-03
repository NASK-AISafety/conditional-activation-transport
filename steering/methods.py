from abc import ABC
from enum import Enum
from typing import Optional
from typing import Tuple, Dict, List
from pathlib import Path

import torch
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import torch.nn as nn


class SteeringMode(Enum):
    CONDITIONAL = 1
    UNCONDITIONAL = 2
    BOTH = 3

    def __str__(self) -> str:
        return self.name


def get_bayes_precision_estimate(covariance_matrix: torch.Tensor, n: int, d: int):
    covariance_matrix = covariance_matrix.to(torch.float32)
    inv_cov = d * torch.inverse(
        (n - 1) * covariance_matrix
        + torch.trace(covariance_matrix) * torch.eye(d).to(covariance_matrix.device)
    )
    return inv_cov


def get_gda_params(means: torch.Tensor, inv_cov: torch.Tensor):
    priori = torch.log(torch.tensor(1.0 / means.size(0))).to(means.device)
    means = means.to(torch.float32)
    W = means @ inv_cov
    b = priori - 0.5 * torch.einsum("nd, dc, nc -> n", means, inv_cov, means)

    return W, b


def get_maha_distance(x: torch.Tensor, mean: torch.Tensor, inv_cov: torch.Tensor):
    x = x.to(torch.float32)
    mean = mean.to(device=x.device, dtype=torch.float32)
    inv_cov = inv_cov.to(device=x.device, dtype=torch.float32)
    diff = x - mean
    return (diff @ inv_cov * diff).sum(dim=1)


def get_means_and_covariance(
    from_activations: torch.Tensor, to_activations: Optional[torch.Tensor] = None
):
    mean_neg = from_activations.mean(dim=0)
    centered_neg = from_activations - mean_neg
    if to_activations is None:
        all_centered = centered_neg
    else:
        mean_pos = to_activations.mean(dim=0)
        centered_pos = to_activations - mean_pos
        all_centered = torch.cat([centered_pos, centered_neg], dim=0)
    covariance_matrix = (all_centered.T @ all_centered) / (all_centered.size(0) - 1)
    if to_activations is None:
        return mean_neg, covariance_matrix
    return mean_pos, mean_neg, covariance_matrix


def get_gda_pred(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, return_probs=True):
    W = W.to(x.device)
    b = b.to(x.device)
    logits = x @ W.T + b
    if return_probs:
        return torch.softmax(logits, dim=1)[:, 1]
    return logits


class SteeringMethod(ABC):
    def steer(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class ConditioningMethod(ABC):
    def condition(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class MahalanobisConditioning(ConditioningMethod):
    def __init__(self, threshold: float = 0.0, **kwargs):
        self.inv_cov_matrix = None
        self.mean_vector = None
        self.threshold = threshold
        self.W = None
        self.b = None

    def condition(self, x, threshold: float = 0.0):
        x = x.mean(dim=1)
        self.W = self.W.to(x.dtype)
        self.b = self.b.to(x.dtype)
        pred = get_gda_pred(x, self.W, self.b, return_probs=True)
        mask = pred >= threshold
        return mask

    def train(self, from_activations, to_activations):
        mean_pos, mean_neg, covariance_matrix = get_means_and_covariance(
            from_activations=from_activations, to_activations=to_activations
        )
        W, b = get_gda_params(
            torch.stack([mean_pos, mean_neg], dim=0),
            get_bayes_precision_estimate(
                covariance_matrix,
                to_activations.size(0) + from_activations.size(0),
                to_activations.size(1),
            ),
        )
        self.W = W
        self.b = b


class OODMahalanobisConditioning(ConditioningMethod):
    def __init__(self, quantile: float = 0.95, **kwargs):
        self.inv_cov_matrix = None
        self.mean_vector = None
        self.quantile = quantile
        self.threshold = None

    def condition(self, x):
        x = x.mean(dim=1).to(torch.float32)
        self.mean_vector = self.mean_vector.to(device=x.device, dtype=torch.float32)
        self.inv_cov_matrix = self.inv_cov_matrix.to(
            device=x.device, dtype=torch.float32
        )

        mahalanobis_dist = get_maha_distance(x, self.mean_vector, self.inv_cov_matrix)
        mask = mahalanobis_dist <= self.threshold
        return mask

    def train(self, from_activations, *args, **kwargs):
        mean, covariance_matrix = get_means_and_covariance(from_activations)
        self.inv_cov_matrix = get_bayes_precision_estimate(
            covariance_matrix,
            from_activations.size(0),
            from_activations.size(1),
        )
        self.mean_vector = mean

        distances = get_maha_distance(
            from_activations, self.mean_vector, self.inv_cov_matrix
        ).to(torch.float32)
        self.threshold = torch.quantile(distances, self.quantile)


class MinMaxConditioning(ConditioningMethod):
    def __init__(self, quantile: float = 0.0, **kwargs):
        self.min_vector = None
        self.max_vector = None
        self.quantile = quantile

    def condition(self, x, **kwargs):
        x = x.mean(dim=1)
        self.min_vector = self.min_vector.to(x.device)
        self.max_vector = self.max_vector.to(x.device)
        mask = torch.all(x >= self.min_vector, dim=1) & torch.all(
            x <= self.max_vector, dim=1
        )
        return mask

    def train(self, from_activations, *args, **kwargs):
        from_mean = from_activations.mean(dim=0, dtype=torch.float32)
        self.min_vector = torch.quantile(from_mean, self.quantile, dim=0)
        self.max_vector = torch.quantile(from_mean, 1.0 - self.quantile, dim=0)


CONDITIONING_METHODS = {
    "MAHALANOBIS": MahalanobisConditioning,
    "OODMAHALANOBIS": OODMahalanobisConditioning,
    "MIN_MAX": MinMaxConditioning,
}


class MeanSteering(SteeringMethod):
    def __init__(
        self,
        steering_mode: SteeringMode,
        conditioning_type: ConditioningMethod | str = "none",
        **conditioner_kwargs,
    ):
        self.steering_mode = steering_mode
        self.strength = None
        self.conditioning_type = conditioning_type
        if conditioning_type == "none":
            self.conditioner = None
        else:
            self.conditioner = CONDITIONING_METHODS[conditioning_type](
                **conditioner_kwargs
            )

    def steer(self, x, strength, **kwargs):
        self.mean_direction = self.mean_direction.to(x.device).to(x.dtype)
        if self.conditioner is not None:
            mask = self.conditioner.condition(x).to(x.device)
            steered_vector = x[mask]
            x[mask] = steered_vector + strength * self.mean_direction
            return x
        else:
            return x + strength * self.mean_direction

    def train(self, from_activations, to_activations):
        self.mean_direction = to_activations.mean(dim=0) - from_activations.mean(dim=0)
        if self.conditioner is not None:
            self.conditioner.train(from_activations, to_activations)


class LinearTransportSteering(SteeringMethod):
    def __init__(
        self,
        steering_mode: SteeringMode,
        conditioning_type: ConditioningMethod | str = "none",
        **conditioner_kwargs,
    ):
        self.steering_mode = steering_mode
        self.scaling = None
        self.steering_vector = None
        self.conditioning_type = conditioning_type
        if conditioning_type == "none":
            self.conditioner = None
        else:
            self.conditioner = CONDITIONING_METHODS[conditioning_type](
                **conditioner_kwargs
            )

    def steer(self, x, strength, **kwargs):
        if self.conditioner is not None:
            mask = self.conditioner.condition(x)
        else:
            mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        self.steering_vector = self.steering_vector.to(x.device).to(x.dtype)
        transported_x = (x[mask] * self.scaling) + self.steering_vector
        x[mask] = (1 - strength) * x[mask] + strength * transported_x
        return x

    def train(self, from_activations, to_activations, input_dims: int = 2):
        from_mean = from_activations.mean(dim=0)
        to_mean = to_activations.mean(dim=0)

        centered_from = (from_activations - from_mean).float()
        centered_to = (to_activations - to_mean).float()
        self.scaling = (centered_from * centered_to).sum() / (centered_to**2).sum()
        self.steering_vector = (to_mean) - (self.scaling * from_mean)
        if self.conditioner is not None:
            self.conditioner.train(from_activations, to_activations)


Location = Tuple[int, int]
Strengths = Tuple[int, int]


class SteeringRegistry:
    def __init__(
        self,
        steering_method: SteeringMethod,
        steering_condition: ConditioningMethod,
        steering_mode: SteeringMode,
        cache_path: str | Path,
        categories: List[str],
        conditioner_kwargs,
    ):
        self._steering_method = steering_method
        self._steering_condition = steering_condition
        self._steering_mode = steering_mode
        self._conditioner_kwargs = conditioner_kwargs
        self._vision_steerers: Dict[Location, SteeringMethod] = {}
        self._text_steerers: Dict[Location, SteeringMethod] = {}
        self.cache_path = Path(cache_path)
        self.categories = pa.array(categories)

    def build_vision(self, num_steps, layers):
        for step in range(num_steps):
            for layer in layers:
                self._vision_steerers[(step, layer)] = self._steering_method(
                    conditioning_type=self._steering_condition,
                    steering_mode=self._steering_mode,
                    conditioner_kwargs=self._conditioner_kwargs,
                )

    def build_text(self, layers):
        for layer in layers:
            self._text_steerers[layer] = self._steering_method(
                conditioning_type=self._steering_condition,
                steering_mode=self._steering_mode,
                conditioner_kwargs=self._conditioner_kwargs,
            )

    def train_text(self, **train_kwargs):
        for layer, steerer in self._text_steerers.items():
            unsafe_table = self.read_layer(layer, split="unsafe")
            safe_table = self.read_layer(layer, split="safe")
            unsafe = self.ppyarrow_to_torch(unsafe_table)
            safe = self.ppyarrow_to_torch(safe_table)
            steerer.train(from_activations=unsafe, to_activations=safe, **train_kwargs)

    def train_vision(self, num_steps, layers, **train_kwargs):
        for step in range(num_steps):
            table_unsafe = self.read_scale(step, split="unsafe")
            table_safe = self.read_scale(step, split="safe")
            for layer in layers:
                location = (step, layer)
                steerer = self._vision_steerers[location]
                unsafe_table = table_unsafe.filter(
                    pc.equal(table_unsafe["layer_id"], layer)
                )
                safe_table = table_safe.filter(pc.equal(table_safe["layer_id"], layer))
                unsafe = self.ppyarrow_to_torch(unsafe_table)
                safe = self.ppyarrow_to_torch(safe_table)
                steerer.train(
                    from_activations=unsafe, to_activations=safe, **train_kwargs
                )

    def read_layer(
        self,
        layer: int,
        split: str,
    ):
        root = self.cache_path
        dir = Path("text") / Path(f"{split}_layer{layer}.parquet")
        path = root / dir

        table = pq.read_table(path, columns=["prompt_id", "category", "hidden"])
        table = table.filter(pc.is_in(table["category"], value_set=self.categories))

        return table

    def read_scale(
        self,
        step: int,
        split: str,
    ):
        root = self.cache_path
        dir = Path("vision") / Path(f"{split}_scale{step}.parquet")
        path = root / dir

        table = pq.read_table(
            path, columns=["prompt_id", "category", "layer_id", "hidden"]
        )
        table = table.filter(pc.is_in(table["category"], value_set=self.categories))
        return table

    def ppyarrow_to_torch(
        self,
        table: pa.lib.Table,
        device: str | torch.device = "cpu",
    ):
        hidden_list = table["hidden"].to_pylist()
        hidden = np.stack(hidden_list, axis=0)
        return torch.from_numpy(hidden).to(device)


class SteeringWrapper(nn.Module):
    def __init__(
        self,
        original_layer: nn.Module,
        registry: SteeringRegistry,
        model_root: nn.Module,
        layer_id: int,
        location: str,
        schedule_by_scale: dict,
        mode: SteeringMode,
        double_batch: bool = True,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.registry = registry
        self.root = model_root
        self.layer_id = layer_id
        self.schedule_by_scale = schedule_by_scale["scales"]
        self.mode = mode
        self.double_batch = double_batch

    def forward(self, *args, **kwargs):
        out = self.original_layer(*args, **kwargs)

        si = getattr(self.root, "_current_step", 0)
        schedule = self.schedule_by_scale.get(str(si))
        if not schedule or (self.layer_id not in set(schedule.get("layers", []))):
            return out

        strength_cond = float(schedule.get("strength_cond", 0.0))
        strength_uncond = float(schedule.get("strength_uncond", 0.0))
        if strength_cond == 0.0 and strength_uncond == 0.0:
            return out

        steerer = self.registry._vision_steerers.get((si, self.layer_id))
        if steerer is None:
            return out

        if self.double_batch:
            B = out.shape[0] // 2
            if self.mode in (SteeringMode.CONDITIONAL, SteeringMode.BOTH):
                x = out[:B]
                out[:B] = steerer.steer(x, strength_cond)

            if self.mode in (SteeringMode.UNCONDITIONAL, SteeringMode.BOTH):
                x = out[B:]
                out[B:] = steerer.steer(x, strength_uncond)
        else:
            out = steerer.steer(out, strength_uncond)
        return out


class TextWrapper(nn.Module):
    def __init__(
        self,
        original_layer: nn.Module,
        registry: SteeringRegistry,
        model_root: nn.Module,
        strength: float,
        layer_id: int,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.registry = registry
        self.root = model_root
        self.layer_id = layer_id
        self.strength = strength

    def forward(self, *args, **kwargs):
        out = self.original_layer(*args, **kwargs)
        steerer = self.registry._text_steerers.get(self.layer_id)
        if steerer is None:
            return out

        if isinstance(out, tuple):
            hidden = out[0]
            hidden = steerer.steer(hidden, self.strength)
            return (hidden,) + out[1:]
        else:
            return steerer.steer(out, self.strength)


def enable_text_steering(model, selected_layers, registry, strength):
    setattr(model, str("_steering_wrapped"), [])
    for lid, block in enumerate(model.encoder.block):
        if lid not in selected_layers:
            continue
        wrapper = TextWrapper(
            original_layer=block,
            registry=registry,
            model_root=model,
            strength=strength,
            layer_id=lid,
        )
        setattr(model.encoder.block, str(lid), wrapper)
        model._steering_wrapped.append((model.encoder.block, str(lid), block))


def enable_text_steering_qwen(model, selected_layers, registry, strength):
    setattr(model, str("_steering_wrapped"), [])
    for lid, layer in enumerate(model.layers):
        if lid not in selected_layers:
            continue
        wrapper = TextWrapper(
            original_layer=layer,
            registry=registry,
            model_root=model,
            strength=strength,
            layer_id=lid,
        )
        setattr(model.layers, str(lid), wrapper)
        model._steering_wrapped.append((model.layers, str(lid), layer))


def disable_registry_steering(model):
    if hasattr(model, "_steering_wrapped"):
        for parent, attr, original in model._steering_wrapped:
            setattr(parent, attr, original)
        model._steering_wrapped.clear()
