"""
VideoMultiScorer: weighted multi-reward evaluation for video models.

Self-contained port of the four GenRL ``longcat`` video rewards (no ``genrl``
dependency):

- ``hpsv3_general``   : HPSv3 per-frame visual quality (prompt "A high-quality
                        image"), mean over frames.
- ``hpsv3_percentile``: HPSv3 per-frame text alignment (the caption), mean of
                        the top-30% frames.
- ``videoalign_mq``   : VideoAlign Motion Quality on a grayscale temp mp4.
- ``videoalign_ta``   : VideoAlign Text-Video Alignment on a color temp mp4.

The reward packages (``hpsv3`` pip package or local ``HPSv3`` submodule, and a
local ``VideoAlign`` repo + checkpoints) are imported lazily inside each scorer,
so this module imports fine before they are installed.

Interface mirrors :class:`meanflownft.rewards.multi_scorer.MultiScorer` exactly so
the NFT trainer's reward path is reused unchanged:

    scorer = VideoMultiScorer(device, {"hpsv3_general": 1.0, ...})
    score_details, _ = scorer(videos[B,F,3,H,W], prompts, metadata=None, only_strict=False)
    # score_details = {"hpsv3_general": [...], ..., "mean": [...]}  (weighted sum)
    scorer.to("cpu")  # offload
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import torch

from meanflownft.utils.fast_init import fast_init
from meanflownft.utils.logging import setup_logging

from .video_utils import (
    cleanup_temp_files,
    prepare_video_images,
    preserve_accelerate_state,
    save_frame_to_temp_png,
    save_video_to_temp_mp4,
    to_grayscale,
)

logger = logging.getLogger(__name__)

# Default locations for the (user-installed) reward repos/checkpoints, mirroring
# GenRL's layout but rooted under meanflownft/rewards/ so the framework is
# self-contained. Override via VideoMultiScorer(... hpsv3_root=, videoalign_ckpt=).
_REWARDS_DIR = Path(__file__).resolve().parent
_DEFAULT_HPSV3_ROOT = _REWARDS_DIR / "HPSv3"
_DEFAULT_VIDEOALIGN_DIR = _REWARDS_DIR / "VideoAlign"

# Shared inferencer caches (mirrors GenRL): the HPSv3 VLM is reused across the
# general/percentile scorers, and the VideoAlign VLM across mq/ta — so each big
# VLM is loaded ONCE, not per reward.
_HPSV3_INFERENCER_CACHE: dict = {}
_VIDEOALIGN_INFERENCER_CACHE: dict = {}


def _transformers_renamed_qwen2vl() -> bool:
    """True if the installed transformers (>=4.52) uses the NEW Qwen2-VL key
    layout, where the vision tower and text model were moved under ``model`` and
    the text model was renamed to ``language_model``. The reward checkpoints
    (HPSv3, VideoReward) were saved with the OLD (<4.52) layout."""
    try:
        import transformers

        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        return (major, minor) >= (4, 52)
    except Exception:  # noqa: BLE001
        return True


def convert_qwen2vl_old_to_new_keys(state_dict: dict) -> dict:
    """Remap pre-4.52 Qwen2-VL ``state_dict`` keys to the transformers>=4.52 layout.

    transformers 4.52 restructured Qwen2-VL, applying this official
    ``_checkpoint_conversion_mapping`` inside ``from_pretrained``::

        ^visual                              -> model.visual
        ^model(?!\\.(language_model|visual))  -> model.language_model

    The reward models instead load their checkpoints via a *manual*
    ``load_state_dict`` (which bypasses that auto-conversion), so on transformers
    >=4.52 the OLD-naming checkpoints (``visual.*`` / ``model.layers.*``) do not
    match the in-memory model (``model.visual.*`` / ``model.language_model.*``).
    Applying the same mapping here makes the load strict-clean, so ALL trained
    weights (LM + reward head) load correctly.

    A leading PEFT ``base_model.model.`` prefix (VideoReward's PeftModel) is
    preserved and the mapping is applied to the remainder. The transform is
    idempotent on keys that are already in the new layout.
    """
    import re

    out: dict = {}
    for k, v in state_dict.items():
        prefix, rest = "", k
        if rest.startswith("base_model.model."):
            prefix, rest = "base_model.model.", rest[len("base_model.model.") :]
        new = re.sub(r"^visual", "model.visual", rest)
        if new == rest:  # the two rules are mutually exclusive
            new = re.sub(r"^model(?!\.(language_model|visual))", "model.language_model", rest)
        out[prefix + new] = v
    return out


def _alias_qwen2vl_embed_tokens(model) -> None:
    """Re-expose ``self.model.embed_tokens`` for transformers>=4.52.

    The reward models' ``forward`` (written for transformers<4.52) calls
    ``self.model.embed_tokens(input_ids)``, but 4.52 moved that module to
    ``self.model.language_model.embed_tokens``. Everything else in their forward
    stays valid on 4.52: ``self.visual`` still works (4.52 keeps a BC property),
    and the combined ``Qwen2VLModel.forward`` is numerically equivalent here
    (with no grid passed it derives standard positions from the attention mask,
    then runs the text model). So we only need to alias ``embed_tokens`` back onto
    the combined ``Qwen2VLModel`` — no edits to the reward packages, no change to
    the numerics. Works for both the bare (HPSv3) and PEFT-wrapped (VideoReward)
    models by walking submodules. No-op on transformers<4.52.
    """
    if not _transformers_renamed_qwen2vl():
        return
    for mod in model.modules():
        if type(mod).__name__ == "Qwen2VLModel" and "embed_tokens" not in mod._modules:
            lm = getattr(mod, "language_model", None)
            if lm is not None and hasattr(lm, "embed_tokens"):
                mod.embed_tokens = lm.embed_tokens


def _as_torch_device(device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _get_hpsv3_inferencer(
    device: torch.device,
    hpsv3_root: str | None = None,
    config_path: str | None = None,
    checkpoint_path: str | None = None,
):
    """Build-or-fetch a cached HPSv3RewardInferencer (shared general/percentile).

    ``config_path`` / ``checkpoint_path`` are forwarded to
    ``HPSv3RewardInferencer``. When both are None it auto-downloads
    ``MizzenAI/HPSv3`` and uses the packaged ``HPSv3_7B.yaml`` (base Qwen2-VL-7B).
    """
    import sys as _sys
    root = Path(hpsv3_root) if hpsv3_root else _DEFAULT_HPSV3_ROOT
    if root.exists() and str(root) not in _sys.path:
        _sys.path.insert(0, str(root))
    # transformers>=4.49 moved ``VideoInput`` from ``image_utils`` to
    # ``video_utils``, but hpsv3 (pinned to transformers==4.45.2) still does
    # ``from transformers.image_utils import VideoInput`` in its
    # differentiable_image_processor. Alias it back so hpsv3 imports cleanly on
    # the env's newer transformers (no env/site-packages change).
    try:
        import transformers.image_utils as _iu
        if not hasattr(_iu, "VideoInput"):
            from transformers.video_utils import VideoInput as _VI
            _iu.VideoInput = _VI
    except Exception:  # noqa: BLE001
        pass
    from hpsv3 import HPSv3RewardInferencer  # lazy

    key = (str(device), config_path, checkpoint_path)
    if key not in _HPSV3_INFERENCER_CACHE:
        # hpsv3 loads its reward checkpoint via ``safetensors.torch.load_file`` +
        # ``model.load_state_dict(..., strict=True)`` (inference.py). On
        # transformers>=4.52 the Qwen2-VL keys were renamed, so the OLD-naming
        # ``HPSv3.safetensors`` won't match the in-memory model. Scope-patch
        # ``load_file`` to remap ONLY that file's keys (the base Qwen2-VL shards,
        # loaded by ``from_pretrained``, keep their own naming and are untouched).
        import safetensors.torch as _stt

        _orig_load_file = _stt.load_file
        _do_remap = _transformers_renamed_qwen2vl()

        def _load_file_remap(filename, *args, **kwargs):
            sd = _orig_load_file(filename, *args, **kwargs)
            try:
                if _do_remap and os.path.basename(str(filename)) == "HPSv3.safetensors":
                    sd = convert_qwen2vl_old_to_new_keys(sd)
            except Exception:  # noqa: BLE001
                pass
            return sd

        _stt.load_file = _load_file_remap
        try:
            with preserve_accelerate_state(), fast_init(device, init_weights=False):
                _HPSV3_INFERENCER_CACHE[key] = HPSv3RewardInferencer(
                    config_path=config_path, checkpoint_path=checkpoint_path, device=device,
                )
        finally:
            _stt.load_file = _orig_load_file
        # transformers>=4.52: re-expose ``self.model.embed_tokens`` used by hpsv3's forward.
        _alias_qwen2vl_embed_tokens(_HPSV3_INFERENCER_CACHE[key].model)
    return _HPSV3_INFERENCER_CACHE[key]


def _get_videoalign_inferencer(device: torch.device, checkpoint_path: str, videoalign_dir: str | None):
    """Build-or-fetch a cached VideoVLMRewardInference (shared mq/ta)."""
    import importlib.util
    import sys as _sys

    va_dir = Path(videoalign_dir) if videoalign_dir else _DEFAULT_VIDEOALIGN_DIR
    inference_path = va_dir / "inference.py"
    if not inference_path.is_file():
        raise FileNotFoundError(
            f"VideoAlign inference.py not found: {inference_path}"
        )
    if va_dir.exists() and str(va_dir) not in _sys.path:
        _sys.path.insert(0, str(va_dir))
    module_name = "_meanflownft_videoalign_inference"
    module = _sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name,
            inference_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load VideoAlign from {inference_path}")
        module = importlib.util.module_from_spec(spec)
        _sys.modules[module_name] = module
        spec.loader.exec_module(module)
        original_loader = module.load_model_from_checkpoint

        def _compatible_checkpoint_loader(model, checkpoint_dir, checkpoint_step):
            try:
                return original_loader(model, checkpoint_dir, checkpoint_step)
            except RuntimeError:
                import glob

                checkpoint_paths = glob.glob(
                    os.path.join(checkpoint_dir, "checkpoint-*")
                )
                checkpoint_paths.sort(
                    key=lambda item: int(item.rsplit("-", 1)[-1]),
                    reverse=True,
                )
                if not checkpoint_paths:
                    raise
                if checkpoint_step not in (None, -1):
                    requested = os.path.join(
                        checkpoint_dir,
                        f"checkpoint-{checkpoint_step}",
                    )
                    selected = (
                        requested
                        if requested in checkpoint_paths
                        else checkpoint_paths[0]
                    )
                else:
                    selected = checkpoint_paths[0]
                state = torch.load(
                    os.path.join(selected, "model.pth"),
                    map_location="cpu",
                    weights_only=False,
                )
                model.load_state_dict(
                    convert_qwen2vl_old_to_new_keys(state),
                    strict=True,
                )
                return model, selected.rsplit("checkpoint-", 1)[-1]

        module.load_model_from_checkpoint = _compatible_checkpoint_loader
    VideoVLMRewardInference = module.VideoVLMRewardInference

    key = (str(va_dir.resolve()), os.path.abspath(checkpoint_path), str(device))
    if key not in _VIDEOALIGN_INFERENCER_CACHE:
        with preserve_accelerate_state(), fast_init(device, init_weights=False):
            _VIDEOALIGN_INFERENCER_CACHE[key] = VideoVLMRewardInference(
                load_from_pretrained=os.path.abspath(checkpoint_path),
                device=str(device), dtype=torch.bfloat16,
            )
        # transformers>=4.52: re-expose ``self.model.embed_tokens`` used by VideoAlign's forward.
        _alias_qwen2vl_embed_tokens(_VIDEOALIGN_INFERENCER_CACHE[key].model)
    return _VIDEOALIGN_INFERENCER_CACHE[key]


def _extract_reward_scalar(reward) -> float:
    if isinstance(reward, (list, tuple)) and len(reward) > 0:
        return float(reward[0])
    if isinstance(reward, torch.Tensor):
        return 0.0 if reward.numel() == 0 else float(reward.flatten()[0].item())
    if hasattr(reward, "item"):
        return float(reward.item())
    return float(reward)


# ---------------------------------------------------------------------------
# HPSv3 (per-frame) scorer — general / percentile
# ---------------------------------------------------------------------------

class HPSv3VideoScorer:
    """HPSv3 per-frame scorer. ``mode`` is ``"general"`` or ``"percentile"``."""

    GENERAL_PROMPT = "A high-quality image"

    def __init__(self, device, mode: str, hpsv3_root: str | None = None,
                 config_path: str | None = None, checkpoint_path: str | None = None):
        assert mode in ("general", "percentile"), mode
        self.mode = mode
        self.device = _as_torch_device(device)
        # Shared across general/percentile (loaded once).
        self.inferencer = _get_hpsv3_inferencer(
            self.device, hpsv3_root, config_path=config_path, checkpoint_path=checkpoint_path,
        )

    def to(self, device):
        device = _as_torch_device(device)
        if getattr(self.inferencer, "device", None) != device:
            self.inferencer.device = device
            self.inferencer.model.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def __call__(self, videos, prompts, metadata=None) -> np.ndarray:
        images_np, is_video = prepare_video_images(videos)
        dev_type = self.device.type
        rewards: list[float] = []
        tmp: list[str] = []
        try:
            for i in range(images_np.shape[0]):
                frames = images_np[i] if is_video else images_np[i : i + 1]
                paths = [save_frame_to_temp_png(f) for f in frames]
                tmp.extend(paths)
                prompt = self.GENERAL_PROMPT if self.mode == "general" else str(prompts[i])
                frame_prompts = [prompt] * len(paths)
                with torch.amp.autocast(
                    device_type=dev_type,
                    dtype=torch.bfloat16,
                    enabled=dev_type == "cuda",
                ):
                    raw = self.inferencer.reward(image_paths=paths, prompts=frame_prompts)
                scores = [_extract_reward_scalar(r) for r in raw]
                if not scores:
                    rewards.append(0.0)
                elif self.mode == "general":
                    rewards.append(float(np.mean(scores)))
                else:
                    scores_sorted = sorted(scores, reverse=True)
                    k = max(1, int(len(scores_sorted) * 0.3))
                    rewards.append(float(np.mean(scores_sorted[:k])))
        finally:
            cleanup_temp_files(tmp)
        return np.asarray(rewards, dtype=np.float32)


# ---------------------------------------------------------------------------
# VideoAlign (whole-video) scorer — MQ / TA
# ---------------------------------------------------------------------------

class VideoAlignScorer:
    """VideoAlign scorer. ``mode`` is ``"mq"`` (grayscale) or ``"ta"`` (color)."""

    def __init__(self, device, mode: str, checkpoint_path: str | None = None,
                 videoalign_dir: str | None = None, fps: float = 8.0):
        assert mode in ("mq", "ta"), mode
        self.mode = mode
        self.key = "MQ" if mode == "mq" else "TA"
        self.fps = fps
        self.device = _as_torch_device(device)
        va_dir = Path(videoalign_dir) if videoalign_dir else _DEFAULT_VIDEOALIGN_DIR
        ckpt = checkpoint_path or str(va_dir / "checkpoints")
        # Shared across mq/ta (loaded once per checkpoint/device).
        self.inferencer = _get_videoalign_inferencer(self.device, ckpt, videoalign_dir)

    def to(self, device):
        device = _as_torch_device(device)
        if str(getattr(self.inferencer, "device", "")) != str(device):
            self.inferencer.device = str(device)
            self.inferencer.model.to(str(device))
        self.device = device
        return self

    @torch.no_grad()
    def __call__(self, videos, prompts, metadata=None) -> np.ndarray:
        images_np, is_video = prepare_video_images(videos)
        if self.mode == "mq":
            images_np = to_grayscale(images_np)
        dev_type = self.device.type
        rewards: list[float] = []
        tmp: list[str] = []
        try:
            for i in range(images_np.shape[0]):
                frames = images_np[i] if is_video else images_np[i : i + 1]
                path = save_video_to_temp_mp4(frames, fps=self.fps)
                tmp.append(path)
                with torch.amp.autocast(
                    device_type=dev_type,
                    dtype=torch.bfloat16,
                    enabled=dev_type == "cuda",
                ):
                    out = self.inferencer.reward(
                        video_paths=[path], prompts=[str(prompts[i])], use_norm=True,
                    )
                rewards.append(float(out[0][self.key]))
        finally:
            cleanup_temp_files(tmp)
        return np.asarray(rewards, dtype=np.float32)


_VIDEO_SCORER_FACTORIES = {
    "hpsv3_general": lambda device, **kw: HPSv3VideoScorer(
        device, "general", hpsv3_root=kw.get("hpsv3_root"),
        config_path=kw.get("hpsv3_config_path"), checkpoint_path=kw.get("hpsv3_checkpoint_path"),
    ),
    "hpsv3_percentile": lambda device, **kw: HPSv3VideoScorer(
        device, "percentile", hpsv3_root=kw.get("hpsv3_root"),
        config_path=kw.get("hpsv3_config_path"), checkpoint_path=kw.get("hpsv3_checkpoint_path"),
    ),
    "videoalign_mq": lambda device, **kw: VideoAlignScorer(
        device, "mq", checkpoint_path=kw.get("videoalign_ckpt"), videoalign_dir=kw.get("videoalign_dir"),
    ),
    "videoalign_ta": lambda device, **kw: VideoAlignScorer(
        device, "ta", checkpoint_path=kw.get("videoalign_ckpt"), videoalign_dir=kw.get("videoalign_dir"),
    ),
}

VIDEO_REWARD_NAMES = tuple(_VIDEO_SCORER_FACTORIES.keys())


class VideoMultiScorer:
    """Weighted multi-reward wrapper for video models (drop-in for MultiScorer)."""

    def __init__(self, device, score_dict, allow_unavailable: bool = False, **scorer_kwargs):
        self.device = _as_torch_device(device)
        self.requested_score_dict = dict(score_dict)
        self.score_dict: dict[str, float] = {}
        self.score_fns: dict[str, object] = {}
        self._unavailable: dict[str, str] = {}

        for name, weight in score_dict.items():
            if name not in _VIDEO_SCORER_FACTORIES:
                raise KeyError(
                    f"Unsupported video reward {name!r}. "
                    f"Supported: {list(_VIDEO_SCORER_FACTORIES.keys())}"
                )
            try:
                with preserve_accelerate_state():
                    fn = _VIDEO_SCORER_FACTORIES[name](self.device, **scorer_kwargs)
            except Exception as e:  # noqa: BLE001
                if not allow_unavailable:
                    raise
                self._unavailable[name] = f"{type(e).__name__}: {e}"
                warnings.warn(
                    f"[VideoMultiScorer] Skip unavailable reward {name!r}: {type(e).__name__}: {e}",
                    stacklevel=2,
                )
                continue
            finally:
                # Reward backends can mutate root logger on import; restore.
                setup_logging()
            self.score_dict[name] = weight
            self.score_fns[name] = fn

        self.active_reward_names = list(self.score_dict.keys())
        self.unavailable_rewards = dict(self._unavailable)

    def __call__(self, videos, prompts, metadata=None, only_strict: bool = True):
        del metadata, only_strict
        score_details: dict[str, object] = {}
        total: np.ndarray | None = None
        for name, weight in self.score_dict.items():
            scores = self.score_fns[name](videos, prompts)
            scores = np.asarray(scores, dtype=np.float32)
            score_details[name] = scores
            weighted = float(weight) * scores
            total = weighted if total is None else total + weighted
        if total is None:
            total = np.zeros(len(prompts), dtype=np.float32)
        # Match MultiScorer: "mean" is the weighted SUM of active rewards.
        score_details["mean"] = total
        return score_details, {}

    def to(self, target_device):
        for fn in self.score_fns.values():
            if hasattr(fn, "to"):
                fn.to(target_device)
        self.device = _as_torch_device(target_device)
        return self
