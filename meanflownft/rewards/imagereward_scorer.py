"""
ImageReward v1.0 scorer.

Adapted from DiffusionNFT's ImageReward scorer.
"""

import importlib
import logging
import os
import sys
import torch
from meanflownft.rewards.reward_ckpt_path import CKPT_PATH

logger = logging.getLogger(__name__)


def _ensure_pkg_resources(retries: int = 8, base_delay: float = 0.5) -> bool:
    """Make ``import pkg_resources`` succeed before ImageReward/clip need it.

    Root cause (observed on this cluster): ImageReward imports ``clip`` which
    does ``from pkg_resources import packaging``. Under the multi-process
    torchrun launch, the shared-CephFS site-packages directory listing comes
    back stale/incomplete for some ranks, so Python's PathFinder fails to find
    the (installed) ``pkg_resources`` package -> ``ModuleNotFoundError: No
    module named 'pkg_resources'`` -- even though a standalone ``import
    pkg_resources`` works fine. It is NOT a missing/poisoned package.

    Fix: drop any poisoned ``None`` cache entry, then retry the import while
    calling :func:`importlib.invalidate_caches` between attempts to force a
    fresh directory re-scan (the standard remedy for "file is on disk but the
    finder did not see it"). Returns True once importable.
    """
    # Clear a poisoned ``= None`` entry (different failure mode; harmless here).
    if sys.modules.get("pkg_resources", False) is None:
        del sys.modules["pkg_resources"]
    if "pkg_resources" in sys.modules:
        return True

    import time

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            import pkg_resources  # type: ignore  # noqa: F401 - cache the real module
            if attempt > 0:
                logger.info(
                    "[ImageReward] pkg_resources became importable after %d "
                    "retries (stale CephFS dir listing).", attempt,
                )
            return True
        except Exception as e:  # noqa: BLE001 - retry stale-finder failures
            last_err = e
            importlib.invalidate_caches()
            time.sleep(base_delay * (attempt + 1))

    logger.warning(
        "[ImageReward] pkg_resources still not importable after %d retries: "
        "%r", retries, last_err,
    )
    return False


# Backwards-compatible alias (older call sites).
def _repair_pkg_resources() -> None:
    _ensure_pkg_resources()


def _load_imagereward_module():
    """Load ImageReward with compatibility patch for new transformers versions."""
    _ensure_pkg_resources()
    try:
        from transformers import modeling_utils as hf_modeling_utils
    except Exception:
        hf_modeling_utils = None

    if hf_modeling_utils is not None:
        try:
            from transformers import pytorch_utils as hf_pytorch_utils
        except Exception:
            hf_pytorch_utils = None
        if hf_pytorch_utils is not None:
            for symbol_name in (
                "apply_chunking_to_forward",
                "find_pruneable_heads_and_indices",
                "prune_linear_layer",
            ):
                if not hasattr(hf_modeling_utils, symbol_name) and hasattr(hf_pytorch_utils, symbol_name):
                    # ImageReward imports these symbols from the old module path.
                    setattr(hf_modeling_utils, symbol_name, getattr(hf_pytorch_utils, symbol_name))

    try:
        return importlib.import_module("ImageReward")
    except Exception:
        # Surface the FULL traceback once (the MultiScorer wrapper otherwise
        # collapses this to just "ModuleNotFoundError: ... pkg_resources",
        # hiding which import actually fails). Diagnostic aid; re-raise so
        # the unavailable-reward handling stays unchanged.
        import traceback
        logger.error(
            "[ImageReward] import failed; pkg_resources in sys.modules=%r\n%s",
            sys.modules.get("pkg_resources", "ABSENT"),
            traceback.format_exc(),
        )
        raise


class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        rm_module = _load_imagereward_module()
        self.model = (
            rm_module.load(
                "ImageReward-v1.0",
                device=str(self.device),
                # Keep ImageReward cache/checkpoints under unified reward_ckpt_path.
                download_root=os.path.join(os.path.expanduser(CKPT_PATH), "ImageReward"),
            )
            .eval()
            .to(dtype=dtype)
        )
        self.model.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (str, int, torch.device)):
                device = first
        if device is not None:
            self.device = torch.device(device)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "device"):
                # ImageReward.inference_rank relies on this attribute to place inputs.
                self.model.device = str(self.device)
        return self

    @torch.no_grad()
    def __call__(self, prompts, images):
        if hasattr(self.model, "device"):
            # Keep inference input placement aligned with model weights.
            self.model.device = str(self.device)
        _, rewards = self.model.inference_rank(prompts, images)
        raw = torch.as_tensor(rewards, device=self.device)
        n = len(prompts)
        try:
            rewards = torch.diagonal(raw.reshape(n, n), 0).contiguous()
        except Exception as e:  # noqa: BLE001 - debug aid for the "all -10" issue
            print(
                f"[ImageRewardDebug] reshape FAILED: n_prompts={n} "
                f"n_images={len(images)} raw_numel={raw.numel()} "
                f"raw_shape={tuple(raw.shape)} err={type(e).__name__}: {e}",
                flush=True,
            )
            raise
        # --- TEMP DEBUG: diagnose why imagereward becomes all -10 in some runs ---
        # Prints on rank 0 always, and on ANY rank that emits the -10 sentinel
        # (so rank-specific failures are visible). Remove once root-caused.
        _rank = int(os.environ.get("RANK", 0))
        _flat = rewards.detach().float().flatten()
        _n_neg10 = int((_flat == -10).sum().item())
        if _rank == 0 or _n_neg10 > 0:
            print(
                f"[ImageRewardDebug][rank{_rank}] n_prompts={n} n_images={len(images)} "
                f"raw_numel={raw.numel()} diag_shape={tuple(rewards.shape)} "
                f"min={_flat.min().item():.4f} max={_flat.max().item():.4f} "
                f"mean={_flat.mean().item():.4f} n_eq_-10={_n_neg10}/{_flat.numel()}",
                flush=True,
            )
        return rewards
