import gc
import argparse
import random
from pathlib import Path
from itertools import chain
from collections import defaultdict
from typing import Dict, List

import torch
import hydra
import cv2
import pandas as pd
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from omegaconf import DictConfig
from loguru import logger

from utils.datasets_utils import get_dataset
from utils.experiment_utils import (
    create_steered_model,
    create_steering_registry,
    select_steering_type,
    Steering,
    ModelSteerer,
    print_vram,
)
from steering.methods import SteeringRegistry
from eval.fid import FID
from eval.classifier import Classifier, calculate_class_percentage
from eval.clip_score import CLIP


rng = random.Random(42)


class FolderDataset(Dataset):
    def __init__(self, filepath: Path):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp")
        paths_iterables = [filepath.rglob(pat) for pat in exts]
        self.paths = sorted(
            (
                p
                for p in chain.from_iterable(paths_iterables)
                if p.is_file() and "benign_images" not in p.parts
            ),
            key=lambda x: int(x.stem.lstrip("steered_").lstrip("base_")),
        )
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # C,H,W to H,W,C
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transforms(img)


def update_csv(csv_path: Path, new_df: pd.DataFrame, key: str) -> pd.DataFrame:
    if not csv_path.exists():
        return new_df

    old_df = pd.read_csv(csv_path)

    old = old_df.set_index(key)
    new = new_df.set_index(key)

    missing_cols = [c for c in new.columns if c not in old.columns]
    if missing_cols:
        old[missing_cols] = new[missing_cols]

    return old.reset_index()


def to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()

    if x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = x.permute(1, 2, 0)

    x = x.float()
    min, max = float(x.min()), float(x.max())

    if min >= 0.0 and max <= 1.0:
        x = x * 255.0
    elif min >= -1.0 and max <= 1.0:
        x = (x + 1.0) * 0.5 * 255.0

    return x.clamp(0, 255).to(torch.uint8).numpy()


def to_chw_float01(img) -> torch.Tensor:
    if isinstance(img, Image.Image):
        t = transforms.ToTensor()(img)
        return t[:3]

    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img)
    elif torch.is_tensor(img):
        t = img

    if t.ndim == 3 and t.shape[0] not in (1, 3, 4) and t.shape[-1] in (3, 4):
        t = t.permute(2, 0, 1)

    t = t.float()

    if t.max() > 1.0:
        t = t / 255.0

    return t[:3]


def evaluate_dir(
    directory_name: str,
    eval_name: str,
    images_eval_name: str,
    eval_config: DictConfig,
    steering_config: DictConfig,
) -> None:
    batch_size = eval_config.batch_size
    fid_real_dataset = eval_config.fid_dataset
    clf_selected_policies = eval_config.policies
    clf_custom_policies = dict(eval_config.eval)
    clf_quant = eval_config.quant
    subfolders = eval_config.subfolders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = Path(directory_name)
    classifier = None
    clip_model = None
    fid_calculator = None

    train_dataset, _ = get_dataset(config=steering_config.experiment)
    steering_type = select_steering_type(steering_config)
    benign_dataset = load_dataset(fid_real_dataset, split="train")
    benign_prompts = rng.sample(benign_dataset["caption"], eval_config.clip_samples)
    steering_registry = create_steering_registry(steering_config)
    steerer = create_steered_model(
        steering_config, train_dataset, steering_registry, steering_type
    )

    if eval_config.classifier:
        classifier = Classifier(
            classifier_type="gemma",
            weights=None,
            custom_policies=clf_custom_policies,
            quant=clf_quant,
            device=device,
        )
    if eval_config.clip:
        clip_model = CLIP(device=device)

    if eval_config.fid:
        fid_calculator = FID(fid_real_dataset, device=device)

    directories = [
        dir for dir in list(directory.iterdir()) if dir.name.startswith("steering")
    ]

    steps = (
        steering_config.steering.STEPS_VISION[0]
        if steering_config.steering.STEPS_VISION
        else None
    )
    results = defaultdict(list)
    if subfolders:
        for p in tqdm(directories, desc="Image eval"):
            if p.is_dir():
                evaluate(
                    filepath=p,
                    setup=p.name,
                    results=results,
                    classifier=classifier,
                    clf_policies=clf_selected_policies,
                    images_eval_name=images_eval_name,
                    prompts=benign_prompts,
                    clip_model=clip_model,
                    fid_calculator=fid_calculator,
                    batch_size=batch_size,
                    steering_config=steering_config,
                    steering_type=steering_type,
                    steering_registry=steering_registry,
                    steps=steps,
                    steerer=steerer,
                )
            torch.cuda.empty_cache()
            gc.collect()
    else:
        evaluate(
            filepath=directory,
            setup=directory.name,
            results=results,
            classifier=classifier,
            clf_policies=clf_selected_policies,
            images_eval_name=images_eval_name,
            prompts=benign_prompts,
            clip_model=clip_model,
            fid_calculator=fid_calculator,
            batch_size=batch_size,
            steering_config=steering_config,
            steering_type=None,
            steering_registry=steering_registry,
            steps=steps,
            steerer=steerer,
        )

    df = pd.DataFrame(results)
    summary_path = directory / Path(eval_name)

    df_updated = update_csv(summary_path, df, key="setup")
    df_updated.to_csv(summary_path, index=False, float_format="%.2f")


def extract_strength(name: str) -> float:
    return float(name.split("steering_strength_", 1)[1].split("_", 1)[0])


def evaluate(
    filepath: str | Path,
    setup: str,
    results: Dict,
    classifier: Classifier,
    clf_policies: List[str],
    images_eval_name: str,
    prompts: List[str],
    clip_model: CLIP,
    fid_calculator: FID,
    batch_size: int,
    steering_config: DictConfig,
    steering_type: Steering,
    steering_registry: SteeringRegistry,
    steps: List[int],
    steerer: ModelSteerer,
) -> None:
    filepath = Path(filepath)
    dataset = FolderDataset(filepath)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results["setup"].append(setup)
    if classifier:
        print_vram()
        y_preds = classifier.classify(dataloader, policies=clf_policies)
        y_preds = classifier.calculate_attack_succes_rate(y_preds)
        policies_with_asr = clf_policies + ["ASR"]
        class_percentages = calculate_class_percentage(y_preds, policies_with_asr)
        for policy, class_perc in class_percentages.items():
            results[policy].append(class_perc)
        per_image_df = pd.DataFrame(
            data=y_preds.float().cpu().numpy(),
            columns=policies_with_asr,
        )
        image_ids = [p.name for p in dataset.paths]
        per_image_df.insert(0, "image", image_ids)
        csv_path = filepath / images_eval_name
        updated = update_csv(csv_path, per_image_df, key="image")
        updated.to_csv(csv_path, index=False, float_format="%.4f")

    if clip_model or fid_calculator:
        if steering_type:
            strength = extract_strength(filepath.name)
            images = steerer.generate_steered_images(
                dataset=prompts,
                registry=steering_registry,
                steering_type=steering_type,
                strength=strength,
                steps=steps,
                text_layers=steering_config.steering.LAYERS_TEXT,
                vision_layers=steering_config.steering.LAYERS_VISION,
                batch_size=steering_config.experiment.BATCH_SIZE,
                save_folder=None,
            )
        else:
            images = steerer.generate_no_steering(
                dataset=prompts,
                batch_size=steering_config.experiment.BATCH_SIZE,
                save_folder=None,
            )
        if steering_config.model.params.arch == "zimage":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # C,H,W to H,W,C
                ]
            )
            images = [transform(img) for img in images]

        if clip_model:
            clip_scores = clip_model.compute_clip_scores(images, prompts)
            mean_clip = round(float(np.mean(clip_scores)), 2)
            results["clip_score"].append(mean_clip)

        if fid_calculator:
            imgs = torch.stack(images, dim=0)
            fid_images = TensorDataset(imgs)
            dataloader = DataLoader(fid_images, batch_size=32, shuffle=False)
            mu_fake, cov_fake = fid_calculator.stream_stats_from_loader(dataloader)
            fid_val = round(fid_calculator.fid_from_stats(mu_fake, cov_fake), 2)
            results["fid"].append(fid_val)

        out_dir = filepath / "benign_images"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            if isinstance(img, Image.Image):
                img.save(out_dir / f"benign_{i}.jpg")
            else:
                arr = to_uint8_hwc(img)
                cv2.imwrite(str(out_dir / f"benign_{i}.jpg"), arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="Path to folder with images")
    parser.add_argument(
        "--eval_name", type=str, default="evaluation_fid.csv", help="Name of eval csv"
    )
    parser.add_argument(
        "--per_image_name",
        type=str,
        default="images.csv",
        help="Name of csv with evaluation of every image independently",
    )
    args = parser.parse_args()
    logger.info(f"Evaluting {args.filepath}")

    with hydra.initialize(version_base="1.2", config_path="config"):
        eval_cfg: DictConfig = hydra.compose(config_name="img_eval")

    with hydra.initialize_config_dir(version_base="1.2", config_dir=args.filepath):
        steering_cfg: DictConfig = hydra.compose(config_name="config")

    evaluate_dir(
        args.filepath,
        eval_name=args.eval_name,
        images_eval_name=args.per_image_name,
        eval_config=eval_cfg,
        steering_config=steering_cfg,
    )


if __name__ == "__main__":
    main()
