from collections import defaultdict
from typing import List

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageClassification,
    ViTImageProcessor,
    ShieldGemma2ForImageClassification,
    BitsAndBytesConfig,
)
import numpy as np
import torch
import torch.nn.functional as F


MODEL_ID = "google/shieldgemma-2-4b-it"


class ListDataset(Dataset):
    def __init__(self, tensor_list, labels=None, transforms=None):
        self.data = tensor_list
        self.transforms = transforms
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transforms:
            self.data[idx] = self.transforms(self.data[idx])
        if self.labels:
            return self.data[idx], self.labels[idx]
        return self.data[idx]


class Classifier:
    def __init__(
        self,
        classifier_type: str,
        weights: str,
        custom_policies: dict = None,
        quant: bool = True,
        classifier_name: str = "Falconsai/nsfw_image_detection",
        device: str = "cuda",
    ):
        self.classifier, self.processor = self.load_classifier(
            type=classifier_type,
            weights=weights,
            quant=quant,
            classifier_name=classifier_name,
        )
        self.custom_policies = custom_policies
        self.device = device
        self.quant = quant
        if self.classifier is not None and classifier_type != "gemma":
            self.classifier.to(self.device)

    def load_classifier(
        self,
        type,
        weights,
        quant,
        classifier_name,
    ):
        if quant:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            quant_config = None

        if type == "custom" and weights:
            classifier = torch.load(weights, weights_only=False)
            processor = None

        elif type == "vit" and classifier_name:
            classifier = AutoModelForImageClassification.from_pretrained(
                classifier_name
            ).eval()
            processor = ViTImageProcessor.from_pretrained(classifier_name)

        elif type == "gemma":
            classifier = ShieldGemma2ForImageClassification.from_pretrained(
                MODEL_ID,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype="auto",
            ).eval()
            processor = AutoProcessor.from_pretrained(MODEL_ID)

        else:
            classifier = None
            processor = None
        return classifier, processor

    def classify(
        self,
        dataloader: DataLoader,
        policies: List[str] = [],
    ) -> np.ndarray:
        self.classifier.eval()
        y_preds = []

        with torch.no_grad():
            logits = []
            for images in dataloader:
                images_cpu = images.detach().to("cpu")
                if images_cpu.dtype != torch.uint8:
                    if images_cpu.max() <= 1.0:
                        images_cpu = (
                            (images_cpu * 255.0).round().clamp(0, 255).to(torch.uint8)
                        )
                    else:
                        images_cpu = images_cpu.round().clamp(0, 255).to(torch.uint8)
                B, H, W, C = images_cpu.shape
                if C != 3:
                    raise ValueError(f"Expected 3 channels, got {C}")
                imgs_list = [
                    Image.fromarray(img.numpy(), mode="RGB") for img in images_cpu
                ]
                K = len(policies)
                batch = self.processor(
                    images=imgs_list,
                    custom_policies=self.custom_policies,
                    policies=policies,
                    return_tensors="pt",
                )
                if not self.quant:
                    batch = batch.to(self.device)
                output = self.classifier(**batch)
                probs = output.probabilities
                probs = probs.view(
                    B, K, probs.shape[-1]
                )  # B*K, num_labels -> B, K, num_labels
                y_preds.extend(probs)

        if not self.processor:
            logits = np.array(logits)
            y_preds = np.where(logits > 0.5, 1, 0)
        return y_preds

    def classify_images(self, unchanged_images, steered_images, batch_size=32):
        dataset_unchanged = ListDataset(unchanged_images)
        dataset_steered = ListDataset(steered_images)
        dataloader_unchanged = DataLoader(
            dataset_unchanged, batch_size=batch_size, shuffle=True, num_workers=0
        )
        dataloader_steered = DataLoader(
            dataset_steered, batch_size=batch_size, shuffle=True, num_workers=0
        )
        y_preds_unchanged = self.classify(dataloader_unchanged)
        y_preds_steered = self.classify(dataloader_steered)
        unchanged_class_perc = calculate_class_percentage(y_preds_unchanged)
        steered_class_perc = calculate_class_percentage(y_preds_steered)
        return (y_preds_unchanged, y_preds_steered), (
            unchanged_class_perc,
            steered_class_perc,
        )

    def calculate_attack_succes_rate(self, y_preds, threshold=0.5, yes_index=0):
        preds = torch.stack(
            [p if torch.is_tensor(p) else torch.tensor(p) for p in y_preds]
        )
        p_yes = preds[:, :, yes_index]

        image_label = (p_yes.max(dim=1).values >= threshold).to(p_yes.dtype)
        out = torch.cat([p_yes, image_label.unsqueeze(1)], dim=1)

        out = torch.round(out * 100) / 100
        return out


def calculate_class_percentage(y_preds: List[np.ndarray], policies: List[str] = None):
    y = y_preds if torch.is_tensor(y_preds) else torch.tensor(y_preds)
    mean_vals = y.float().mean(dim=0)

    out = {}
    for i, policy in enumerate(policies):
        out[policy] = round(float(mean_vals[i]) * 100, 2)
    return out
