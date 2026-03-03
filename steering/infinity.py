import argparse
import sys
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast
from huggingface_hub import hf_hub_download, login

from utils.experiment_utils import (
    ModelSteerer,
    Steering,
)
from steering.methods import (
    SteeringMode,
    SteeringRegistry,
    disable_registry_steering,
    enable_text_steering,
)
from tools.run_infinity import *


def encode_batch(
    captions: List[str], text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel
):
    """Tokenise a list of strings and return compact T5 key-value tensors."""
    tokens = text_tokenizer(
        text=captions,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    with torch.inference_mode():
        text_features = text_encoder(
            input_ids=input_ids, attention_mask=mask, return_dict=False
        )[0]
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    return (kv_compact, lens, cu_seqlens_k, Ltext)


def inference(
    infinity,
    vae,
    text_tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    prompts: List[str],
    scale_schedule: List[tuple[int, int, int]],
    cfg_list: List[int],
    tau_list: List[float],
    seed: int,
    batch_size: int = 32,
    gt_leak: int = 0,
    gt_ls_Bl: int = None,
    cfg_insertion_layer: List[int] = [0],
    vae_type: int = 32,
    sampling_per_bits: int = 1,
    softmax_merge_topk: int = -1,
    top_k: int = 900,
    top_p: float = 0.97,
    cfg_sc: int = 3,
    cfg_exp_k: float = 0.0,
    gumbel: int = 0,
    capture_activations: bool = False,
) -> tuple[torch.Tensor, Dict[int, Dict[nn.Module, torch.Tensor]]]:
    """Run Infinity's autoregressive_infer_cfg with convenience defaults."""
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_batch(prompts, text_tokenizer, text_encoder)
    with (
        torch.inference_mode(),
        torch.amp.autocast(
            device_type="cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True
        ),
    ):
        _, _, img_list, activations = infinity.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple,
            g_seed=seed,
            B=min(batch_size, len(prompts)),
            negative_label_B_or_BLT=None,
            force_gt_Bhw=None,
            cfg_sc=cfg_sc,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=top_k,
            top_p=top_p,
            returns_vemb=1,
            ratio_Bl1=None,
            gumbel=gumbel,
            norm_cfg=False,
            cfg_exp_k=cfg_exp_k,
            cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type,
            softmax_merge_topk=softmax_merge_topk,
            ret_img=True,
            trunk_scale=1000,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            capture_activations=capture_activations,
        )
    return img_list, activations


def generate_infinity(
    dataset: List[str], infinity_setup: Dict, batch_size: int, vae_type: int = 16
) -> torch.Tensor:
    """Return a list of generated images for all prompts."""
    batched_prompts = [
        dataset[p : p + batch_size] for p in range(0, len(dataset), batch_size)
    ]
    images = []
    for batch in batched_prompts:
        img_list, activations = inference(
            infinity=infinity_setup["transformer"],
            vae=infinity_setup["vae"],
            text_tokenizer=infinity_setup["text_tokenizer"],
            text_encoder=infinity_setup["text_encoder"],
            prompts=batch,
            scale_schedule=infinity_setup["scale_schedule"],
            vae_type=vae_type,
            cfg_list=infinity_setup["cfg"],
            tau_list=infinity_setup["tau"],
            seed=infinity_setup["seed"],
        )
        del activations
        img_list = [img.detach().cpu() for img in img_list]
        images.extend(img_list)
        torch.cuda.empty_cache()
    return images


@dataclass
class InfinityParams:
    repository: str
    hf_hub_token: str
    weights: str
    model_name: str
    model_type: str
    vae_name: str
    vae_type: int
    num_layers: int
    num_scales: int
    text_encoder_ckpt: str
    pixel_number: str
    cfg: float
    out_folder: str
    num_steps: Optional[int] = None


class InfinitySteerer(ModelSteerer):
    def __init__(self, params: InfinityParams):
        self.params = params
        self.params.num_steps = params.num_scales

    def setup_model(self):
        infinity_args = self.config_infinity()

        infinity, vae, text_encoder, text_tokenizer = self.load_infinity_weights(
            infinity_args
        )
        self.infinity_setup = self.setup_infinity(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            infinity_args,
        )

    def get_model(self):
        return self.infinity_setup["transformer"]

    def config_infinity(self) -> argparse.Namespace:
        sys.path.insert(0, os.getcwd())
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        weights_dir = self.params.weights
        model_path = f"{weights_dir}/{self.params.model_name}"
        vae_path = f"{weights_dir}/{self.params.vae_name}"
        text_encoder_ckpt = self.params.text_encoder_ckpt
        args = argparse.Namespace(
            pn=self.params.pixel_number,
            model_path=model_path,
            cfg_insertion_layer=0,
            vae_type=self.params.vae_type,
            vae_path=vae_path,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=True,
            model_type=self.params.model_type,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            sampling_per_bits=1,
            text_encoder_ckpt=text_encoder_ckpt,
            text_channels=2048,
            apply_spatial_patchify=0,
            h_div_w_template=1.000,
            use_flex_attn=0,
            cache_dir="./Infinity/tmp",
            checkpoint_type="torch",
            seed=0,
            bf16=1,
            save_file="tmp.jpg",
            enable_model_cache=False,
        )
        return args

    def load_infinity_weights(self, args):
        login(token=self.params.hf_hub_token)

        hf_hub_download(
            repo_id=self.params.repository,
            filename=self.params.vae_name,
            local_dir=self.params.weights,
        )
        hf_hub_download(
            repo_id=self.params.repository,
            filename=self.params.model_name,
            local_dir=self.params.weights,
        )

        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)

        vae = load_visual_tokenizer(args)

        infinity = load_transformer(vae, args)
        return infinity, vae, text_encoder, text_tokenizer

    def setup_infinity(self, infinity, vae, text_tokenizer, text_encoder, args):
        h_div_w = 1 / 1
        h_div_w_template_ = h_div_w_templates[
            np.argmin(np.abs(h_div_w_templates - h_div_w))
        ]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["scales"]
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

        tau = 0.5
        seed = random.randint(0, 10000)
        return {
            "transformer": infinity,
            "vae": vae,
            "text_tokenizer": text_tokenizer,
            "text_encoder": text_encoder,
            "scale_schedule": scale_schedule,
            "cfg": self.params.cfg,
            "tau": tau,
            "seed": seed,
        }

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
    ) -> None:
        infinity = self.infinity_setup["transformer"]
        text_encoder = self.infinity_setup["text_encoder"]

        if steering_type in (Steering.VISUAL, Steering.BOTH):
            scales = steps
            setup = {
                "scales": self.create_scale_layer_dict(
                    self.params.num_scales, vision_layers, scales, strength
                )
            }
            infinity.enable_registry_steering(
                registry=registry,
                schedule_by_scale=setup,
                steering_location="ffn",
                steering_mode=steering_mode,
            )

        if steering_type in (Steering.TEXT, Steering.BOTH):
            enable_text_steering(text_encoder, text_layers, registry, strength)

        steered = generate_infinity(
            dataset,
            self.infinity_setup,
            batch_size=batch_size,
            vae_type=self.params.vae_type,
        )
        steered_cpu = [t.detach().cpu() for t in steered]

        if steering_type in (Steering.TEXT, Steering.BOTH):
            disable_registry_steering(text_encoder)
        if steering_type in (Steering.VISUAL, Steering.BOTH):
            infinity.disable_registry_steering()

        if save_folder is None:
            return steered_cpu

        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(steered_cpu):
            cv2.imwrite(save_folder / Path(f"steered_{i}.jpg"), img.numpy())

    def generate_no_steering(
        self,
        dataset: List[str],
        batch_size: int,
        save_folder: Path = Path("./generated_images"),
    ) -> None:
        """Return a list of generated images for all prompts."""
        images = generate_infinity(
            dataset,
            self.infinity_setup,
            batch_size=batch_size,
            vae_type=self.params.vae_type,
        )

        if save_folder is None:
            return images

        for i, img in enumerate(images):
            cv2.imwrite(save_folder / Path(f"base_{i}.jpg"), img.numpy())

    def create_scale_layer_dict(
        self,
        num_scales: int,
        layers_list: List[int],
        scales_to_steer: List[int],
        steering_strength: float,
    ) -> Dict[int, List[int]]:
        scale_layer_dict = {}
        for scale in range(num_scales):
            if scale in scales_to_steer:
                scale_layer_dict[f"{scale}"] = {
                    "layers": layers_list,
                    "strength_cond": steering_strength,
                    "strength_uncond": steering_strength,
                }
            else:
                scale_layer_dict[f"{scale}"] = {
                    "layers": [],
                    "strength_cond": 0.0,
                    "strength_uncond": 0.0,
                }

        return scale_layer_dict

    def get_text_activations_cached(
        self,
        contrastive_dataset: pd.DataFrame,
        out_dir: str | Path,
        layers_to_capture: List[int] | None = None,
        batch_size: int = 16,
        safe_class: int = 0,
        remove_files_if_exists: bool = False,
    ) -> None:

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
            self.infinity_setup["text_tokenizer"],
            self.infinity_setup["text_encoder"],
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
                desc="Collecting t5 activations",
            ):
                batch = contrastive_dataset.iloc[i : i + batch_size]
                texts = batch["text"].tolist()
                labels = batch["label"].to_numpy()
                categories = batch["category"].astype(str).to_numpy()

                with torch.no_grad():
                    toks = tokenizer(
                        texts,
                        padding="longest",
                        truncation=True,
                        return_tensors="pt",
                    ).to(encoder.device)
                    hidden = encoder(**toks, output_hidden_states=True)

                prompt_ids = np.arange(
                    prompt_base, prompt_base + len(texts), dtype=np.int64
                )

                for lid in layers_to_capture:
                    pooled = hidden.hidden_states[
                        lid + 1
                    ] * toks.attention_mask.unsqueeze(-1)
                    pooled = (
                        (
                            pooled.sum(dim=1)
                            / toks.attention_mask.sum(dim=1, keepdim=True)
                        )
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
                        safe_hidden_col = pa.array(safe.tolist(), type=pa.list_(dtype))
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
                        unsafe_hidden_col = pa.array(
                            unsafe.tolist(), type=pa.list_(dtype)
                        )
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
        remove_files_if_exists: bool = False,
        safe_class: int = 0,
    ) -> None:
        """Collect per-layer activations for a set of prompts."""
        out_dir = Path(out_dir) / "vision"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if out_dir.exists() and not remove_files_if_exists:
            return

        out_dir.mkdir(parents=True, exist_ok=True)

        layers_to_capture = layers_to_capture or list(range(self.params.num_layers))

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
                desc="Collecting infinity activations",
            ):
                batch = contrastive_dataset.iloc[i : i + batch_size]
                texts = batch["text"].tolist()
                labels = batch["label"].to_numpy()
                categories = batch["category"].astype(str).to_numpy()

                img_list, activations = inference(
                    infinity=self.infinity_setup["transformer"],
                    vae=self.infinity_setup["vae"],
                    text_tokenizer=self.infinity_setup["text_tokenizer"],
                    text_encoder=self.infinity_setup["text_encoder"],
                    prompts=texts,
                    scale_schedule=self.infinity_setup["scale_schedule"],
                    vae_type=self.params.vae_type,
                    cfg_list=self.infinity_setup["cfg"],
                    tau_list=self.infinity_setup["tau"],
                    seed=self.infinity_setup["seed"],
                    capture_activations=True,
                )
                del img_list

                prompt_ids = np.arange(
                    prompt_base, prompt_base + len(texts), dtype=np.int64
                )

                # activation = {scale: {layer_name: activation_tensor}}, each tensor has shape (2*B, L, C)
                for scale, activation in activations.items():
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
                        B = activation_tensor.shape[0] // 2
                        pooled = (
                            (
                                activation_tensor[:B]
                                .detach()
                                .to(device)
                                .mean(dim=1)
                                .to(torch.float32)
                            )
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
                            int(scale),
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
                            int(scale),
                        ).write_table(unsafe_table)

                prompt_base += len(texts)
                del activations
        finally:
            for writer in pqwriters.values():
                writer.close()
