"""SD3.5 VAE decoding utilities used by evaluation and inference."""

from __future__ import annotations

import torch
from PIL import Image


def _config_get(config, name: str, default=None):
    if hasattr(config, name):
        return getattr(config, name)
    if hasattr(config, "get"):
        return config.get(name, default)
    return default


@torch.no_grad()
def decode_latents_to_tensor(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode latent tensors to float NCHW tensor [0, 1] using the VAE.

    Args:
        vae: AutoencoderKL instance.
        latents: Latent tensors [B, C, H, W] on GPU.

    Returns:
        Float tensor [B, C, H, W] in [0, 1] range on GPU.
    """
    device = latents.device
    if next(vae.parameters()).device != device:
        vae.to(device)

    scaling_factor = _config_get(vae.config, "scaling_factor", 1.0)
    shift_factor = _config_get(vae.config, "shift_factor", 0.0)
    scaled_latents = latents / scaling_factor + shift_factor
    images = vae.decode(scaled_latents.to(vae.dtype)).sample

    images = (images / 2 + 0.5).clamp(0, 1)
    return images


@torch.no_grad()
def decode_latents_to_pil(vae, latents: torch.Tensor) -> list[Image.Image]:
    """Decode latent tensors to PIL images using the VAE.

    Args:
        vae: AutoencoderKL instance.
        latents: Latent tensors [B, C, H, W] on GPU.

    Returns:
        List of PIL Images.
    """
    images = decode_latents_to_tensor(vae, latents)
    images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(img) for img in images]
