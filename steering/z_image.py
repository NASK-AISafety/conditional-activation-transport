import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.zimage.pipeline import generate
from src.utils.helpers import ensure_model_weights
from src.utils.loader import load_from_local_dir
from src.utils.attention import set_attention_backend
from utils.experiment_utils import ModelSteerer, Steering
from steering.methods import (
    SteeringMode,
    SteeringRegistry,
    disable_registry_steering,
    enable_text_steering_qwen,
)


def generate_zimage(
    dataset: List[str],
    components: dict,
    device: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    batch_size: int = 16,
    seed: int = 42,
):
    batched_prompts = [
        dataset[p : p + batch_size] for p in range(0, len(dataset), batch_size)
    ]
    images = []
    for batch in batched_prompts:
        generator = torch.Generator(device).manual_seed(seed)

        img_list = generate(
            prompt=batch,
            **components,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        images.extend(img_list)
        torch.cuda.empty_cache()
    return images


@dataclass
class ZimageParams:
    weights: str
    inference_steps: int
    guidance_scale: float
    out_folder: str
    num_step: Optional[int] | None = None


class ZimageSteerer(ModelSteerer):
    def __init__(self, params: ZimageParams, device: str = "cuda"):
        self.params = params
        self.params.num_steps = params.inference_steps
        self.device = device

    def setup_model(self, dtype=torch.bfloat16, compile: bool = False):
        attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
        set_attention_backend(attn_backend)
        model_path = ensure_model_weights(self.params.weights)
        self.zimage_setup = load_from_local_dir(
            model_path, device=self.device, dtype=dtype, compile=compile
        )

    def get_model(self):
        return self.zimage_setup["transformer"]

    def generate_steered_images(
        self,
        dataset: List[str],
        registry: SteeringRegistry,
        steering_type: Steering,
        strength: float,
        text_layers: List[int],
        vision_layers: List[int],
        steps: List[int],
        steering_mode: SteeringMode = SteeringMode.BOTH,
        batch_size: int = 16,
        save_folder: str = ".",
    ):
        zimage = self.zimage_setup["transformer"]
        text_encoder = self.zimage_setup["text_encoder"]

        if steering_type in (Steering.VISUAL, Steering.BOTH):
            setup = {
                "scales": self.create_timestep_layer_dict(
                    self.params.inference_steps, vision_layers, steps, strength
                )
            }
            zimage.enable_registry_steering(
                registry=registry,
                schedule_by_scale=setup,
                steering_location="ffn",
                steering_mode=steering_mode,
            )
        if steering_type in (Steering.TEXT, Steering.BOTH):
            enable_text_steering_qwen(text_encoder, text_layers, registry, strength)

        steered = generate_zimage(
            dataset,
            device=self.device,
            components=self.zimage_setup,
            batch_size=batch_size,
            num_inference_steps=self.params.inference_steps,
            guidance_scale=self.params.guidance_scale,
        )

        if steering_type in (Steering.TEXT, Steering.BOTH):
            disable_registry_steering(text_encoder)
        if steering_type in (Steering.VISUAL, Steering.BOTH):
            zimage.disable_registry_steering()

        if save_folder is None:
            return steered

        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(steered):
            img.save(save_folder / Path(f"steered_{i}.jpg"))

    def generate_no_steering(
        self,
        dataset: List[str],
        batch_size: int,
        save_folder: Path = Path("./generated_images"),
    ) -> torch.Tensor:
        """Return a list of generated images for all prompts."""
        images = generate_zimage(
            dataset,
            device=self.device,
            components=self.zimage_setup,
            batch_size=batch_size,
            num_inference_steps=self.params.inference_steps,
            guidance_scale=self.params.guidance_scale,
        )

        if save_folder is None:
            return images

        for i, img in enumerate(images):
            img.save(save_folder / Path(f"base_{i}.jpg"))

    def create_timestep_layer_dict(
        self,
        num_timesteps: int,
        layers_list: List[int],
        timesteps_to_steer: List[int],
        steering_strength: float,
    ) -> Dict[int, List[int]]:
        timestep_layer_dict = {}
        for step in range(num_timesteps):
            if step in timesteps_to_steer:
                timestep_layer_dict[f"{step}"] = {
                    "layers": layers_list,
                    "strength_cond": steering_strength,
                    "strength_uncond": steering_strength,
                }
            else:
                timestep_layer_dict[f"{step}"] = {
                    "layers": [],
                    "strength_cond": 0.0,
                    "strength_uncond": 0.0,
                }

        return timestep_layer_dict

    def get_text_activations_cached(
        self,
        contrastive_dataset: pd.DataFrame,
        out_dir: str | Path,
        layers_to_capture: List[int] | None = None,
        batch_size: int = 16,
        max_sequence_length: int = 512,
        safe_class: int = 0,
        remove_files_if_exists: bool = False,
    ):
        out_dir = Path(out_dir) / "text"

        if out_dir.exists() and not remove_files_if_exists:
            return

        out_dir.mkdir(parents=True, exist_ok=True)

        dtype = pa.float32()
        schema = pa.schema(
            [
                ("prompt_id", pa.int64()),
                ("category", pa.string()),
                ("hidden", pa.list_(dtype)),
            ]
        )

        tokenizer, encoder = (
            self.zimage_setup["tokenizer"],
            self.zimage_setup["text_encoder"],
        )
        encoder.eval()

        layers_to_capture = layers_to_capture or list(range(len(encoder.encoder.block)))
        pqwriters: Dict[Tuple[str, int, int], pq.ParquetWriter] = {}

        def get_writer(
            split: str,
            layer: int,
        ) -> pq.ParquetWriter:
            key = (split, layer)
            if key in pqwriters:
                return pqwriters[key]

            fname = out_dir / f"{split}_layer{layer}.parquet"
            if remove_files_if_exists:
                if fname.exists():
                    fname.unlink()

            w = pq.ParquetWriter(fname, schema=schema, compression="zstd")
            pqwriters[key] = w
            return w

        prompt_base = 0
        try:
            for i in tqdm(
                range(0, len(contrastive_dataset), batch_size),
                desc="Collecting qwen activations",
            ):
                formatted_prompts = []
                batch = contrastive_dataset.iloc[i : i + batch_size]
                texts = batch["text"].tolist()
                labels = batch["label"].to_numpy()
                categories = batch["category"].astype(str).to_numpy()

                for p in texts:
                    messages = [{"role": "user", "content": p}]
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                    formatted_prompts.append(formatted_prompt)

                text_inputs = tokenizer(
                    formatted_prompts,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(self.device)
                prompt_masks = text_inputs.attention_mask.to(self.device).bool()

                prompt_embeds = encoder(
                    input_ids=text_input_ids,
                    attention_mask=prompt_masks,
                    output_hidden_states=True,
                )
                prompt_ids = np.arange(
                    prompt_base, prompt_base + len(texts), dtype=np.int64
                )
                for lid in layers_to_capture:
                    pooled = prompt_embeds.hidden_states[
                        lid + 1
                    ] * prompt_masks.unsqueeze(2)
                    pooled = (
                        (pooled.sum(dim=1) / prompt_masks.sum(dim=1, keepdim=True))
                        .to(torch.float32)
                        .cpu()
                        .numpy()
                    )

                    safe_mask = labels == (0 if safe_class == 0 else 1)
                    unsafe_mask = ~safe_mask

                    safe = pooled[safe_mask]
                    unsafe = pooled[unsafe_mask]

                    safe_ids = prompt_ids[safe_mask]
                    unsafe_ids = prompt_ids[unsafe_mask]

                    if safe.shape[0] > 0:
                        safe_categories_col = pa.array(
                            categories[safe_mask].tolist(), type=pa.string()
                        )
                        safe_hidden_col = pa.array(list(safe), type=pa.list_(dtype))
                        safe_table = pa.Table.from_arrays(
                            [
                                pa.array(safe_ids, type=pa.int64()),
                                safe_categories_col,
                                safe_hidden_col,
                            ],
                            schema=schema,
                        )
                        get_writer("safe", lid).write_table(safe_table)

                    if unsafe.shape[0] > 0:
                        unsafe_categories_col = pa.array(
                            categories[unsafe_mask].tolist(), type=pa.string()
                        )
                        unsafe_hidden_col = pa.array(list(unsafe), type=pa.list_(dtype))
                        unsafe_table = pa.Table.from_arrays(
                            [
                                pa.array(unsafe_ids, type=pa.int64()),
                                unsafe_categories_col,
                                unsafe_hidden_col,
                            ],
                            schema=schema,
                        )
                        get_writer("unsafe", lid).write_table(unsafe_table)

                prompt_base += len(texts)
        finally:
            for writer in pqwriters.values():
                writer.close()

    def get_visual_activations_cached(
        self,
        contrastive_dataset: pd.DataFrame,
        batch_size: int,
        out_dir: str | Path,
        layers_to_capture: List[int] = None,
        safe_class: int = 0,
        height: int = 256,
        width: int = 256,
        device: str = "cuda",
        seed: int = 42,
        remove_files_if_exists: bool = False,
    ):
        out_dir = Path(out_dir) / "vision"

        if out_dir.exists() and not remove_files_if_exists:
            return

        out_dir.mkdir(parents=True, exist_ok=True)

        layers_to_capture = layers_to_capture or list(
            range(len(self.params.num_layers))
        )

        dtype = pa.float32()
        schema = pa.schema(
            [
                ("prompt_id", pa.int64()),
                ("category", pa.string()),
                ("layer_id", pa.int32()),
                ("hidden", pa.list_(dtype)),
            ]
        )
        pqwriters: Dict[Tuple[str, int, int], pq.ParquetWriter] = {}

        def get_writer(
            split: str,
            scale: int,
        ) -> pq.ParquetWriter:
            key = (split, scale)
            if key in pqwriters:
                return pqwriters[key]

            fname = out_dir / f"{split}_scale{scale}.parquet"
            if remove_files_if_exists:
                if fname.exists():
                    fname.unlink()

            w = pq.ParquetWriter(fname, schema=schema, compression="zstd")
            pqwriters[key] = w
            return w

        prompt_base = 0
        try:
            for i in tqdm(
                range(0, len(contrastive_dataset), batch_size),
                desc="Collecting z-image activations",
            ):
                batch = contrastive_dataset.iloc[i : i + batch_size]
                texts = batch["text"].tolist()
                labels = batch["label"].to_numpy()
                categories = batch["category"].astype(str).to_numpy()

                generator = torch.Generator(device).manual_seed(seed)
                img_list, activations = generate(
                    prompt=texts,
                    **self.zimage_setup,
                    height=height,
                    width=width,
                    num_inference_steps=self.params.inference_steps,
                    guidance_scale=self.params.guidance_scale,
                    generator=generator,
                    capture_activations=True,
                )
                del img_list
                prompt_ids = np.arange(
                    prompt_base, prompt_base + len(texts), dtype=np.int64
                )

                for time_step, activation in activations.items():
                    safe_ids_all = []
                    safe_layer_all = []
                    safe_hidden_all = []
                    safe_categories_all = []

                    unsafe_ids_all = []
                    unsafe_layer_all = []
                    unsafe_hidden_all = []
                    unsafe_categories_all = []

                    for layer_id, activation_tensor in enumerate(activation.values()):
                        if int(layer_id) not in layers_to_capture:
                            continue
                        pooled = (
                            (activation_tensor.detach().mean(dim=1))
                            .to(torch.float32)
                            .cpu()
                            .numpy()
                        )

                        safe_mask = labels == (0 if safe_class == 0 else 1)
                        unsafe_mask = ~safe_mask

                        if safe_mask.any():
                            safe_ids_all.append(prompt_ids[safe_mask])
                            safe_categories_all.extend(categories[safe_mask].tolist())
                            safe_layer_all.append(
                                np.full(int(safe_mask.sum()), int(layer_id), np.int32)
                            )
                            safe_hidden_all.append(pooled[safe_mask])

                        if unsafe_mask.any():
                            unsafe_ids_all.append(prompt_ids[unsafe_mask])
                            unsafe_categories_all.extend(
                                categories[unsafe_mask].tolist()
                            )
                            unsafe_layer_all.append(
                                np.full(int(unsafe_mask.sum()), int(layer_id), np.int32)
                            )
                            unsafe_hidden_all.append(pooled[unsafe_mask])

                    if safe_ids_all:
                        safe_ids_cat = np.concatenate(safe_ids_all)
                        safe_layer_cat = np.concatenate(safe_layer_all)
                        safe_hidden_cat = np.concatenate(safe_hidden_all, axis=0)

                        safe_ids_col = pa.array(safe_ids_cat, type=pa.int64())
                        safe_categories_col = pa.array(
                            safe_categories_all, type=pa.string()
                        )
                        safe_layer_col = pa.array(
                            safe_layer_cat,
                            type=pa.int32(),
                        )
                        safe_hidden_col = pa.array(
                            list(safe_hidden_cat), type=pa.list_(dtype)
                        )
                        safe_table = pa.Table.from_arrays(
                            [
                                safe_ids_col,
                                safe_categories_col,
                                safe_layer_col,
                                safe_hidden_col,
                            ],
                            schema=schema,
                        )
                        get_writer(
                            "safe",
                            int(time_step),
                        ).write_table(safe_table)

                    if unsafe_hidden_all:
                        unsafe_ids_cat = np.concatenate(unsafe_ids_all)
                        unsafe_layer_cat = np.concatenate(unsafe_layer_all)
                        unsafe_hidden_cat = np.concatenate(unsafe_hidden_all, axis=0)

                        unsafe_ids_col = pa.array(unsafe_ids_cat, type=pa.int64())
                        unsafe_categories_col = pa.array(
                            unsafe_categories_all, type=pa.string()
                        )
                        unsafe_layer_col = pa.array(
                            unsafe_layer_cat,
                            type=pa.int32(),
                        )
                        unsafe_hidden_col = pa.array(
                            list(unsafe_hidden_cat), type=pa.list_(dtype)
                        )
                        unsafe_table = pa.Table.from_arrays(
                            [
                                unsafe_ids_col,
                                unsafe_categories_col,
                                unsafe_layer_col,
                                unsafe_hidden_col,
                            ],
                            schema=schema,
                        )
                        get_writer(
                            "unsafe",
                            int(time_step),
                        ).write_table(unsafe_table)

                prompt_base += len(texts)
                del activations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            for writer in pqwriters.values():
                writer.close()
