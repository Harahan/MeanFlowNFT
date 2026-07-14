"""
Human Preference Score v3 (HPSv3) reward scorer.

This scorer follows DanceGRPO's usage pattern:
    from hpsv3 import HPSv3RewardInferencer
    reward = inferencer.reward(images_or_paths, prompts)

`hpsv3` is not a default MeanFlowNFT dependency. Install it first, e.g.:
    pip install hpsv3 --no-deps
or clone/install from https://github.com/MizzenAI/HPSv3.

Path behavior (aligned with hpsv2 style):
- Prefer local config/checkpoint under `reward_ckpt_path` (CKPT_PATH) if present.
- Optional explicit overrides via env:
    MEANFLOWNFT_HPSV3_CONFIG_PATH
    MEANFLOWNFT_HPSV3_CHECKPOINT_PATH
- Fall back to HPSv3 package defaults if no local files are found.
"""

from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from meanflownft.rewards.reward_ckpt_path import CKPT_PATH


def _patch_transformers_for_hpsv3():
    """Backfill symbols for hpsv3/transformers API drift."""
    try:
        from transformers import image_utils as hf_image_utils
    except Exception:
        return

    if not hasattr(hf_image_utils, "VideoInput"):
        # Older transformers versions do not expose this typing alias.
        # HPSv3 only needs it to exist during import time.
        hf_image_utils.VideoInput = object

    # Newer transformers moved Qwen2VL token embedding access behind
    # get_input_embeddings(), while older hpsv3 code still calls:
    #   self.model.embed_tokens(input_ids)
    # where self.model is Qwen2VLModel. Add a compatibility alias.
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel
    except Exception:
        return

    if not hasattr(Qwen2VLModel, "embed_tokens"):
        Qwen2VLModel.embed_tokens = property(
            lambda self: self.get_input_embeddings()
        )


def _patch_safetensors_loader_for_hpsv3():
    """Fix known HPSv3 checkpoint key mismatch at load time."""
    try:
        import safetensors.torch as safetensors_torch
    except Exception:
        return

    if getattr(safetensors_torch.load_file, "_meanflownft_hpsv3_patched", False):
        return

    original_load_file = safetensors_torch.load_file

    def _patched_load_file(filename, *args, **kwargs):
        state_dict = original_load_file(filename, *args, **kwargs)
        if isinstance(state_dict, dict):
            # Mirror the conversion used by recent transformers Qwen2-VL:
            #   ^visual -> model.visual
            #   ^model(?!\.(language_model|visual)) -> model.language_model
            # This keeps old HPSv3 checkpoints loadable under newer transformers.
            rewritten = {}
            changed = False
            for key, value in state_dict.items():
                new_key = key
                if key.startswith("visual."):
                    new_key = f"model.visual.{key[len('visual.'):]}"
                elif key.startswith("model.") and not key.startswith("model.visual.") and not key.startswith(
                    "model.language_model."
                ):
                    new_key = f"model.language_model.{key[len('model.'):]}"
                if new_key != key:
                    changed = True
                rewritten[new_key] = value
            if changed:
                state_dict = rewritten
        return state_dict

    _patched_load_file._meanflownft_hpsv3_patched = True
    safetensors_torch.load_file = _patched_load_file


def _import_hpsv3():
    _patch_transformers_for_hpsv3()
    _patch_safetensors_loader_for_hpsv3()
    try:
        from hpsv3 import HPSv3RewardInferencer
    except ImportError as e:
        detail = str(e)
        if "VideoInput" in detail and "transformers.image_utils" in detail:
            raise ImportError(
                "HPSv3 import failed due to transformers compatibility "
                f"(missing image_utils.VideoInput). Original error: {detail}"
            ) from e
        raise ImportError(
            "HPSv3 scorer requires a compatible `hpsv3` installation. "
            "Install via `pip install hpsv3 --no-deps` or clone/install "
            "https://github.com/MizzenAI/HPSv3. "
            f"Original error: {detail}"
        ) from e
    return HPSv3RewardInferencer


def _to_pil_list(images) -> list[Image.Image]:
    if isinstance(images, list) and len(images) > 0 and isinstance(images[0], Image.Image):
        return images
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)
    if isinstance(images, np.ndarray):
        return [Image.fromarray(img) for img in images]
    raise TypeError(f"Unsupported image type for HPSv3 scorer: {type(images)}")


def _exists(path: str | Path | None) -> bool:
    return bool(path) and Path(path).expanduser().is_file()


def _pick_first_file(dir_path: Path, names: Sequence[str]) -> str | None:
    for name in names:
        path = dir_path / name
        if path.is_file():
            return str(path)
    return None


def _discover_hpsv3_paths() -> tuple[str | None, str | None]:
    """Discover local HPSv3 config/checkpoint paths from env and CKPT_PATH."""
    env_cfg = os.environ.get("MEANFLOWNFT_HPSV3_CONFIG_PATH")
    env_ckpt = os.environ.get("MEANFLOWNFT_HPSV3_CHECKPOINT_PATH")
    if _exists(env_cfg) or _exists(env_ckpt):
        return (
            str(Path(env_cfg).expanduser()) if _exists(env_cfg) else None,
            str(Path(env_ckpt).expanduser()) if _exists(env_ckpt) else None,
        )

    ckpt_root = Path(CKPT_PATH).expanduser()
    candidate_dirs = [
        ckpt_root / "hpsv3",
        ckpt_root / "HPSv3",
        ckpt_root,
    ]
    config_names = (
        "config.yaml",
        "config.yml",
        "hpsv3.yaml",
        "inferencer_config.yaml",
        "HPSv3_7B.yaml",
    )
    ckpt_names = (
        "HPSv3.safetensors",
        "HPS_v3.1_compressed.pt",
        "HPS_v3_compressed.pt",
        "HPSv3.pt",
        "hpsv3.pt",
        "checkpoint.pt",
        "checkpoint.pth",
        "model.pt",
        "model.pth",
        "model.safetensors",
        "pytorch_model.bin",
    )

    best_cfg = None
    best_ckpt = None
    for directory in candidate_dirs:
        if not directory.is_dir():
            continue
        if best_cfg is None:
            best_cfg = _pick_first_file(directory, config_names)
        if best_ckpt is None:
            best_ckpt = _pick_first_file(directory, ckpt_names)
        if best_cfg is not None and best_ckpt is not None:
            break

    return best_cfg, best_ckpt


def _build_hpsv3_inferencer(HPSv3RewardInferencer, device: torch.device):
    """Initialize inferencer with local paths when supported by the package."""
    config_path, checkpoint_path = _discover_hpsv3_paths()

    # Keep args compatible across different hpsv3 versions.
    sig = inspect.signature(HPSv3RewardInferencer.__init__)
    params = sig.parameters
    kwargs = {}
    if "device" in params:
        kwargs["device"] = str(device)
    if "differentiable" in params:
        kwargs["differentiable"] = False
    if config_path and "config_path" in params:
        kwargs["config_path"] = config_path
    if checkpoint_path and "checkpoint_path" in params:
        kwargs["checkpoint_path"] = checkpoint_path

    has_custom_paths = ("config_path" in kwargs) or ("checkpoint_path" in kwargs)

    try:
        return HPSv3RewardInferencer(**kwargs)
    except Exception:
        # Fallback to package defaults only when custom local paths fail.
        if not has_custom_paths:
            raise
        return HPSv3RewardInferencer(device=str(device))


class HPSv3Scorer(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        del dtype  # HPSv3 inferencer controls precision internally.
        self.device = torch.device(device)
        HPSv3RewardInferencer = _import_hpsv3()
        self._inferencer = _build_hpsv3_inferencer(
            HPSv3RewardInferencer, self.device
        )
        self._sync_inferencer_device(move_model=True)
        self.eval()

    def _sync_inferencer_device(self, move_model: bool):
        device_str = str(self.device)
        try:
            if hasattr(self._inferencer, "device"):
                self._inferencer.device = device_str
            if not move_model:
                return
            if hasattr(self._inferencer, "model"):
                model = self._inferencer.model
                model_device = None
                first_param = next(model.parameters(), None)
                if first_param is not None:
                    model_device = first_param.device
                else:
                    first_buffer = next(model.buffers(), None)
                    if first_buffer is not None:
                        model_device = first_buffer.device
                if model_device != self.device:
                    model.to(self.device)
            elif hasattr(self._inferencer, "to"):
                self._inferencer.to(self.device)
        except Exception:
            pass

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (str, int, torch.device)):
                device = first
        if device is not None:
            self.device = torch.device(device)
            self._sync_inferencer_device(move_model=True)
        return self

    @staticmethod
    def _pil_to_temp_paths(images: Sequence[Image.Image]) -> list[str]:
        paths = []
        for img in images:
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            img.save(path, format="PNG")
            paths.append(path)
        return paths

    @torch.no_grad()
    def __call__(self, images, prompts):
        if len(prompts) == 0:
            return torch.empty(0, dtype=torch.float32)

        # Keep inferencer input placement aligned with scorer.device.
        # Avoid moving the heavy reward model in the hot call path.
        self._sync_inferencer_device(move_model=False)
        pil_images = _to_pil_list(images)

        # Prefer direct PIL path first; if inferencer requires file paths,
        # fallback to temporary png files.
        try:
            rewards = self._inferencer.reward(pil_images, prompts)
        except Exception:
            paths = self._pil_to_temp_paths(pil_images)
            try:
                rewards = self._inferencer.reward(paths, prompts)
            finally:
                for path in paths:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards)

        rewards = rewards.detach().float()
        if rewards.ndim == 2 and rewards.shape[1] >= 1:
            rewards = rewards[:, 0]
        elif rewards.ndim > 1:
            rewards = rewards.reshape(rewards.shape[0], -1)[:, 0]
        return rewards.contiguous()
