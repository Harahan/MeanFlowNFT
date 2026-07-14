"""
CLIP-based text-image similarity scorer.

Adapted from DiffusionNFT's CLIP scorer.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoImageProcessor, CLIPProcessor, CLIPModel


def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")


def get_image_transform(processor: AutoImageProcessor):
    config = processor.to_dict()
    resize = T.Resize(get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
    crop = T.CenterCrop(get_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
    normalise = (
        T.Normalize(mean=processor.image_mean, std=processor.image_std) if config.get("do_normalize") else nn.Identity()
    )
    return T.Compose([resize, crop, normalise])


class ClipScorer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tform = get_image_transform(self.processor.image_processor)
        self.eval()

    def _process(self, pixels):
        dtype = pixels.dtype
        pixels = self.tform(pixels)
        pixels = pixels.to(dtype=dtype)
        return pixels

    @torch.no_grad()
    def __call__(self, pixels, prompts, return_img_embedding=False):
        texts = self.processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        pixels = self._process(pixels).to(self.device)
        outputs = self.model(pixel_values=pixels, **texts)
        if return_img_embedding:
            return outputs.logits_per_image.diagonal() / 100, outputs.image_embeds
        return outputs.logits_per_image.diagonal() / 100
