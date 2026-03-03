import open_clip
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class ClipDataset(Dataset):
    def __init__(self, prompts, images):
        self.images = torch.stack(images)
        self.prompts = prompts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.prompts[idx]


class CLIP:
    def __init__(
        self, model_type="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cuda"
    ):
        self.model, self.preprocess, self.tokenizer = self.setup_clip_model(
            model_type=model_type, pretrained=pretrained
        )
        self.device = device

    def setup_clip_model(self, model_type, pretrained):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_type, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_type)
        return model, preprocess, tokenizer

    def compute_clip_scores(
        self,
        images,
        prompts,
        batch_size=32,
    ):
        dataset = ClipDataset(prompts, images)
        loader = DataLoader(dataset, batch_size=batch_size)
        score = []
        self.model.eval().to(self.device)
        with torch.no_grad(), torch.autocast("cuda"):
            for image, prompt in loader:
                image_batch = normalize_images(image).to(self.device)
                image_features = self.model.encode_image(image_batch)
                tokens = self.tokenizer(prompt).to(self.device)
                text_features = self.model.encode_text(tokens)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                cos_sim = nn.CosineSimilarity()
                clip_score = torch.relu(cos_sim(image_features, text_features)).cpu()
                score.append(clip_score)
        return torch.cat(score).mean().item()


def normalize_images(image_batch, image_size=224):
    if image_batch.dtype != torch.float32:
        image_batch = image_batch.float()

    if torch.max(image_batch) > 1.0:
        image_batch = image_batch / 255.0

    image_batch = image_batch.permute(0, 3, 1, 2)  # BHWC to BCHW

    image_batch = F.interpolate(
        image_batch,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
    )

    mean = CLIP_MEAN.to(image_batch.device)
    std = CLIP_STD.to(image_batch.device)

    return (image_batch - mean) / std
