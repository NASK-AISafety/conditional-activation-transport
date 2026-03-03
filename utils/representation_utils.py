from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
from eval.classifier import *
from steering.infinity import *
from steering.methods import *


Location = Tuple[int, int]


def get_infinity_activations(
    contrastive_dataset: List[str],
    batch_size: int,
    infinity_setup: tuple,
    vae_type: int = 16,
    safe_class: int = 0,
) -> Dict[int, Dict[nn.Module, torch.Tensor]]:
    """Collect per-layer activations for a set of prompts."""
    batched_prompts = [
        contrastive_dataset[p : p + batch_size]
        for p in range(0, len(contrastive_dataset), batch_size)
    ]
    infinity, vae, text_tokenizer, text_encoder, scale_schedule, cfg, tau, seed = (
        infinity_setup
    )
    unsafe_activations = {}
    safe_activations = {}
    buckets = {}
    for batch in tqdm(batched_prompts, desc="Collecting infinity activations"):
        img_list, activations = inference(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            batch,
            scale_schedule,
            vae_type=vae_type,
            cfg_list=cfg,
            tau_list=tau,
            seed=seed,
            capture_activations=True,
        )
        # activation = {scale: {layer_name: activation_tensor}}, each tensor has shape (2*B, L, C)
        for scale, activation in activations.items():
            for layer_id, activation_tensor in enumerate(activation.values()):
                location: Location = (scale, layer_id)
                bucket = buckets.setdefault(location, [])
                bucket.append(activation_tensor.detach().mean(dim=1).cpu())

        del img_list, activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for location, data in buckets.items():
        joined_activations = torch.cat(data, dim=0)
        if safe_class == 0:
            unsafe_activations[location] = joined_activations[1::2]
            safe_activations[location] = joined_activations[::2]
        else:
            unsafe_activations[location] = joined_activations[::2]
            safe_activations[location] = joined_activations[1::2]
    return unsafe_activations, safe_activations


def save_infinity_activations(
    contrastive_dataset: List[str],
    batch_size: int,
    infinity_setup: tuple,
    vae_type: int = 16,
    cache_path: Path = None,
) -> Dict[int, Dict[nn.Module, torch.Tensor]]:
    """Collect per-layer activations for a set of prompts."""
    batched_prompts = [
        contrastive_dataset[p : p + batch_size]
        for p in range(0, len(contrastive_dataset), batch_size)
    ]
    infinity, vae, text_tokenizer, text_encoder, scale_schedule, cfg, tau, seed = (
        infinity_setup
    )
    for i, batch in tqdm(
        list(enumerate(batched_prompts)), desc="Collecting infinity activations"
    ):
        img_list, activations = inference(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            batch,
            scale_schedule,
            vae_type=vae_type,
            cfg_list=cfg,
            tau_list=tau,
            seed=seed,
            capture_activations=True,
        )
        container = {}
        # activation = {scale: {layer_name: activation_tensor}}, each tensor has shape (2*B, L, C)
        for scale, activation in activations.items():
            for layer_id, activation_tensor in enumerate(activation.values()):
                location: Location = (scale, layer_id)
                container[location] = (
                    activation_tensor.detach().mean(dim=1).cpu().to(torch.float16)
                )

        filename = Path(f"batch_{i}.pt")
        torch.save(container, cache_path / filename)
        del img_list, activations, container
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_means_difference(
    contrastive_dataset: List[str],
    batch_size: int,
    infinity_setup: tuple,
    vae_type: int = 16,
) -> Dict[int, Dict[nn.Module, torch.Tensor]]:
    """Collect per-layer activations for a set of prompts."""
    batched_prompts = [
        contrastive_dataset[p : p + batch_size]
        for p in range(0, len(contrastive_dataset), batch_size)
    ]
    infinity, vae, text_tokenizer, text_encoder, scale_schedule, cfg, tau, seed = (
        infinity_setup
    )
    means_diff = {}
    buckets = {}
    for batch in tqdm(batched_prompts, desc="Collecting activations"):
        img_list, activations = inference(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            batch,
            scale_schedule,
            vae_type=vae_type,
            cfg_list=cfg,
            tau_list=tau,
            seed=seed,
            capture_activations=True,
        )
        # activation = {scale: {layer_name: activation_tensor}}, each tensor has shape (2*B, L, C)
        for scale, activation in activations.items():
            for layer_id, activation_tensor in enumerate(activation.values()):
                B = activation_tensor.shape[0] // 2
                key = (scale, layer_id)
                bucket = buckets.setdefault(key, {"cond": [], "uncond": []})
                bucket["cond"].append(activation_tensor[:B].detach().cpu())
                bucket["uncond"].append(activation_tensor[B:].detach().cpu())

        del img_list, activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for (scale, layer_id), data in buckets.items():
        H_cond = torch.cat(data["cond"], dim=0)
        H_uncond = torch.cat(data["uncond"], dim=0)
        cond_even = H_cond[::2].mean(dim=1).mean(dim=0)
        cond_odd = H_cond[1::2].mean(dim=1).mean(dim=0)
        uncond_even = H_uncond[::2].mean(dim=1).mean(dim=0)
        uncond_odd = H_uncond[1::2].mean(dim=1).mean(dim=0)
        scale_dict = means_diff.setdefault(scale, {})
        scale_dict[layer_id] = {
            "cond": -(cond_odd - cond_even),
            "uncond": -(uncond_odd - uncond_even),
        }
    return means_diff


def separation_check(activation_dir: Dict[int, Dict[nn.Module, torch.Tensor]]) -> None:
    """Check if the activations are separated by class."""
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    separation_acc = {}
    for scale, layers in activation_dir.items():
        for layer_id in range(len(layers)):
            class_one_activations = list(activation_dir[scale].values())[layer_id][::2]
            class_two_activations = list(activation_dir[scale].values())[layer_id][1::2]
            X = np.concatenate((class_one_activations, class_two_activations), axis=0)
            y = np.array(
                [0] * len(class_one_activations) + [1] * len(class_two_activations)
            )
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]
            train_size = int(len(X) * 0.8)
            classifier.fit(X[:train_size], y[:train_size])
            train_acc = classifier.score(X[:train_size], y[:train_size])
            test_acc = classifier.score(X[train_size:], y[train_size:])
            separation_acc[(scale, layer_id)] = {
                "train_acc": round(train_acc, 2),
                "test_acc": round(test_acc, 2),
            }
    return separation_acc


def get_t5_activations(
    prompts: list[str],
    tokenizer: T5TokenizerFast,
    encoder: T5EncoderModel,
    layers_to_capture: list[int] | None = None,
    batch_size: int = 16,
    safe_class: int = 0,
) -> dict[int, np.ndarray]:
    encoder.eval()
    layers_to_capture = layers_to_capture or list(range(len(encoder.encoder.block)))

    unsafe_buf: Dict[int, List[torch.Tensor]] = {lid: [] for lid in layers_to_capture}
    safe_buf: Dict[int, List[torch.Tensor]] = {lid: [] for lid in layers_to_capture}
    with torch.inference_mode():
        for i in tqdm(
            range(0, len(prompts), batch_size), desc="Collecting t5 activations"
        ):
            toks = tokenizer(
                prompts[i : i + batch_size],
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ).to(encoder.device)

            hidden = encoder(**toks, output_hidden_states=True)
            for lid in layers_to_capture:
                pooled = hidden.hidden_states[lid + 1] * toks.attention_mask.unsqueeze(
                    -1
                )
                pooled = pooled.sum(dim=1) / toks.attention_mask.sum(
                    dim=1, keepdim=True
                )
                if safe_class == 0:
                    unsafe_buf[lid].append(pooled[1::2].cpu().detach())
                    safe_buf[lid].append(pooled[::2].cpu().detach())
                else:
                    unsafe_buf[lid].append(pooled[::2].cpu().detach())
                    safe_buf[lid].append(pooled[1::2].cpu().detach())

        unsafe_activations = {
            lid: torch.cat(tensors, dim=0) for lid, tensors in unsafe_buf.items()
        }
        safe_activations = {
            lid: torch.cat(tensors, dim=0) for lid, tensors in safe_buf.items()
        }
    return unsafe_activations, safe_activations
