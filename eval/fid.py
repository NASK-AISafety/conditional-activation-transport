import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from datasets import load_dataset, Image as HFImage
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, Lambda
from PIL import Image


class FID:
    def __init__(self, real_dataset, real_dataset_split="train", device="cuda"):
        self.device = device
        self.real_mu, self.real_cov = self.load_fid_stats(
            real_dataset_name=real_dataset, real_dataset_split=real_dataset_split
        )

    def load_fid_stats(self, real_dataset_name, real_dataset_split):
        real_images_loader = self.load_real_images(
            dataset_name=real_dataset_name, split=real_dataset_split
        )
        mu_real, cov_real = self.stream_stats_from_loader(real_images_loader)

        return mu_real, cov_real

    def load_real_images(
        self,
        dataset_name: str,
        split: str = "train",
        batch_size: int = 64,
    ):
        dataset = load_dataset(
            dataset_name,
            split=split,
        ).cast_column("image", HFImage())
        basic_transforms = Compose(
            [
                Resize((224, 224)),
                Lambda(
                    lambda img: (
                        img.convert("RGB") if isinstance(img, Image.Image) else img
                    )
                ),
                ToTensor(),
            ]
        )
        transform = make_transform(basic_transforms)
        dataset.reset_format()
        dataset.set_transform(transform)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

    @torch.no_grad()
    def stream_stats_from_loader(self, loader: DataLoader, dims=2048):
        net, _ = self._build_inception(dims)
        D = dims
        n = 0
        s = torch.zeros(D, dtype=torch.float64)
        s2 = torch.zeros(D, D, dtype=torch.float64)

        for batch in loader:
            images = batch["image"] if isinstance(batch, dict) else batch
            x = images[0] if isinstance(batch, (tuple, list)) else images
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            if x.ndim == 3:
                x = x.unsqueeze(0)

            if x.ndim == 4 and x.shape[1] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
                x = x.permute(0, 3, 1, 2)

            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            else:
                x = x.float()
                if x.min() < 0.0:
                    x = (x + 1.0) * 0.5
                if x.max() > 1.5:
                    x = x / 255.0
                x = x.clamp(0.0, 1.0)

            x = x.to(self.device, non_blocking=True).float()
            f = net(x)[0].squeeze(-1).squeeze(-1)
            f64 = f.double().cpu()

            n += f64.shape[0]
            s += f64.sum(dim=0)
            s2 += f64.t().mm(f64)

        mu = (s / max(n, 1)).numpy()

        cov = (
            (s2 - n * torch.outer(torch.from_numpy(mu), torch.from_numpy(mu)))
            / max(n - 1, 1)
        ).numpy()
        return mu, cov

    def _build_inception(self, dims: int = 2048):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        net = InceptionV3([block_idx]).to(self.device).eval()
        return net, block_idx

    def fid_from_stats(self, mu_f, sig_f):
        return float(
            calculate_frechet_distance(self.real_mu, self.real_cov, mu_f, sig_f)
        )


def make_transform(basic_transforms):
    def transform(example):
        if isinstance(example["image"], list):
            example["image"] = [
                basic_transforms(img) if isinstance(img, Image.Image) else img
                for img in example["image"]
            ]

        elif isinstance(example["image"], Image.Image):
            example["image"] = basic_transforms(example["image"])
        return example

    return transform
