"""
Shared helpers for video reward scoring.

Self-contained port of GenRL's ``genrl/reward/utils.py`` plus the temp-file
helpers from its hpsv3 / videoalign reward modules. No ``genrl`` dependency:
the only external pieces are the reward packages themselves (``hpsv3`` and the
local ``VideoAlign`` repo), which are imported lazily inside the scorers.

Video tensors flow through MeanFlowNFT as ``[B, F, C, H, W]`` float in ``[0, 1]``
(the layout produced by :func:`meanflownft.models.wan.decode_wan_latents`).
"""

from __future__ import annotations

import contextlib
import os
import tempfile

import numpy as np
import torch
from PIL import Image


def prepare_video_images(images) -> tuple[np.ndarray, bool]:
    """Normalize reward input to a uint8 numpy array + ``is_video`` flag.

    Mirrors GenRL ``prepare_images``:
      - torch.Tensor ``[B, C, H, W]`` (image) -> NHWC uint8, is_video=False
      - torch.Tensor ``[B, F, C, H, W]`` (video) -> NFHWC uint8, is_video=True
      - numpy 4D/5D -> assumed already NHWC / NFHWC, cast to uint8

    Float inputs are assumed in ``[0, 1]`` and scaled to ``[0, 255]``.
    """
    if isinstance(images, torch.Tensor):
        if images.dim() == 4 and images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC
            is_video = False
        elif images.dim() == 5 and images.shape[2] == 3:
            images = images.permute(0, 1, 3, 4, 2)  # NFCHW -> NFHWC
            is_video = True
        else:
            raise ValueError(f"Unsupported tensor shape for reward: {tuple(images.shape)}")
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    else:
        images = np.asarray(images)
        if images.ndim == 4:
            is_video = False
        elif images.ndim == 5:
            is_video = True
        else:
            raise ValueError(f"Unsupported array shape for reward: {images.shape}")
        if images.dtype != np.uint8:
            images = (images * 255).round().clip(0, 255).astype(np.uint8)
    return images, is_video


def to_grayscale(images: np.ndarray) -> np.ndarray:
    """RGB -> grayscale (replicated to 3 channels). NHWC or NFHWC uint8."""
    if images.ndim in (4, 5):
        gray = np.dot(images[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    raise ValueError(f"Unsupported array shape for grayscale: {images.shape}")


def save_frame_to_temp_png(frame: np.ndarray) -> str:
    """Save one HWC uint8 frame to a temp PNG; returns the path."""
    pil = Image.fromarray(frame)
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.close()
    pil.save(f.name)
    return f.name


def save_video_to_temp_mp4(frames: np.ndarray, fps: float = 8.0) -> str:
    """Save FHWC uint8 frames to a temp mp4 (libx264); returns the path."""
    import imageio  # lazy: requires imageio + imageio-ffmpeg

    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    f = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    f.close()
    imageio.mimsave(f.name, frames, fps=fps, codec="libx264", format="FFMPEG")
    return f.name


def cleanup_temp_files(paths) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


def preserve_accelerate_state():
    """Context manager preserving Accelerate global state during reward init.

    Some reward backends (HPSv3 / VideoAlign) construct HF ``TrainingArguments``
    which mutate the global Accelerate state; restore it afterwards. No-op if
    accelerate is unavailable.
    """
    try:
        from accelerate.state import AcceleratorState, PartialState
    except Exception:  # noqa: BLE001
        return contextlib.nullcontext()

    class _StatePreserver:
        def __enter__(self):
            self._acc_state = dict(AcceleratorState._shared_state)
            self._partial_state = dict(PartialState._shared_state)

        def __exit__(self, exc_type, exc, tb):
            AcceleratorState._shared_state.clear()
            AcceleratorState._shared_state.update(self._acc_state)
            PartialState._shared_state.clear()
            PartialState._shared_state.update(self._partial_state)
            return False

    return _StatePreserver()
