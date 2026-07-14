"""
MultiScorer: weighted multi-reward evaluation wrapper.

Adapted from DiffusionNFT's reward wrapper.
Computes multiple reward metrics and returns weighted scores.

For MeanFlowNFT evaluation, we use a simplified interface:
    scorer = MultiScorer(device, {"pickscore": 1.0, "hpsv2": 1.0})
    score_details, _ = scorer(images, prompts, metadata={})
    # score_details = {"pickscore": [...], "hpsv2": [...], "mean": [...]}

Supported reward keys are the ones used by the release configs:
pickscore, hpsv2, hpsv3, clipscore, aesthetic, imagereward, ocr, and geneval2.
"""

import logging
import warnings
from contextlib import contextmanager
import numpy as np
import torch
from PIL import Image

from meanflownft.utils.fast_init import fast_init
from meanflownft.utils.logging import setup_logging


@contextmanager
def _preserve_root_logger_state():
    """Guard root logger against side-effects from reward backend imports/init."""
    root = logging.getLogger()
    level = root.level
    disabled = root.disabled
    propagate = root.propagate
    handlers = list(root.handlers)
    handler_levels = [(handler, handler.level) for handler in handlers]
    try:
        yield
    finally:
        root.setLevel(level)
        root.disabled = disabled
        root.propagate = propagate
        root.handlers[:] = handlers
        for handler, handler_level in handler_levels:
            handler.setLevel(handler_level)


def _pil_list_to_nchw_tensor(images) -> torch.Tensor:
    """Convert a list of PIL images to NCHW uint8 tensor [0, 255].

    Handles three input formats:
    - torch.Tensor (NCHW float [0,1]): convert to uint8
    - numpy array (NHWC uint8): transpose to NCHW tensor
    - list[PIL.Image]: stack into NCHW tensor

    Returns:
        torch.Tensor of shape [B, C, H, W], dtype=torch.uint8
    """
    if isinstance(images, torch.Tensor):
        return (images * 255).round().clamp(0, 255).to(torch.uint8)
    if isinstance(images, np.ndarray):
        # NHWC → NCHW
        return torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.uint8)
    # list of PIL images
    arrays = [np.array(img) for img in images]
    # Stack → NHWC → NCHW
    stacked = np.stack(arrays, axis=0)
    return torch.tensor(stacked.transpose(0, 3, 1, 2), dtype=torch.uint8)


def _pil_list_to_pil(images) -> list:
    """Ensure images are a list of PIL.Image instances.

    Handles three input formats:
    - list[PIL.Image]: pass through
    - torch.Tensor (NCHW float [0,1]): convert to PIL
    - numpy array (NHWC uint8): convert to PIL
    """
    if isinstance(images, list) and len(images) > 0 and isinstance(images[0], Image.Image):
        return images
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)
    return [Image.fromarray(img) for img in images]


def aesthetic_score(device):
    from meanflownft.rewards.aesthetic_scorer import AestheticScorer
    scorer = AestheticScorer(dtype=torch.float32, device=device)
    def _fn(images, prompts, metadata):
        images = _pil_list_to_nchw_tensor(images)
        scores = scorer(images)
        return scores, {}
    return _fn


def clip_score(device):
    from meanflownft.rewards.clip_scorer import ClipScorer
    scorer = ClipScorer(device=device)
    def _fn(images, prompts, metadata):
        images = _pil_list_to_nchw_tensor(images).float() / 255.0
        scores = scorer(images, prompts)
        return scores, {}
    return _fn


def hpsv2_score(device):
    from meanflownft.rewards.hpsv2_scorer import HPSv2Scorer
    scorer = HPSv2Scorer(dtype=torch.float32, device=device)
    def _fn(images, prompts, metadata):
        images = _pil_list_to_nchw_tensor(images).float() / 255.0
        scores = scorer(images, prompts)
        return scores, {}
    return _fn


def hpsv3_score(device):
    from meanflownft.rewards.hpsv3_scorer import HPSv3Scorer
    scorer = HPSv3Scorer(dtype=torch.float32, device=device)
    def _fn(images, prompts, metadata):
        del metadata
        images = _pil_list_to_pil(images)
        scores = scorer(images, prompts)
        return scores, {}
    return _fn


def pickscore_score(device):
    from meanflownft.rewards.pickscore_scorer import PickScoreScorer
    scorer = PickScoreScorer(dtype=torch.float32, device=device)
    def _fn(images, prompts, metadata):
        images = _pil_list_to_pil(images)
        scores = scorer(prompts, images)
        return scores, {}
    return _fn


def imagereward_score(device):
    from meanflownft.rewards.imagereward_scorer import ImageRewardScorer
    scorer = ImageRewardScorer(dtype=torch.float32, device=device)
    def _fn(images, prompts, metadata):
        images = _pil_list_to_pil(images)
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}
    return _fn


def geneval2_score(device):
    from meanflownft.rewards.geneval2_scorer import GenEval2SoftTIFAScorer
    scorer = GenEval2SoftTIFAScorer(device=device, dtype=torch.float16, aggregation="gm")
    def _fn(images, prompts, metadata):
        del metadata
        images = _pil_list_to_pil(images)
        scores = scorer(prompts, images)
        return scores, {}
    return _fn


def ocr_score(device):
    from meanflownft.rewards.ocr_scorer import OcrScorer
    scorer = OcrScorer()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
        elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], Image.Image):
            images = np.stack([np.array(img) for img in images], axis=0)
        scores = scorer(images, prompts)
        return scores, {}
    return _fn


class MultiScorer:
    """Wrapper that computes weighted multi-reward scores and supports device offloading."""

    def __init__(self, device, score_dict, allow_unavailable: bool = False):
        score_functions = {
            "ocr": ocr_score,
            "imagereward": imagereward_score,
            "pickscore": pickscore_score,
            "aesthetic": aesthetic_score,
            "geneval2": geneval2_score,
            "clipscore": clip_score,
            "hpsv2": hpsv2_score,
            "hpsv3": hpsv3_score,
        }
        self.requested_score_dict = dict(score_dict)
        self.score_dict = {}
        self.device = device
        self.score_fns = {}
        self._scorers = []
        self._unavailable = {}
        # Use fast_init to skip redundant weight initialization in scorer models.
        # This avoids RNG consumption and speeds up loading.
        with fast_init(torch.device("cpu")):
            for score_name, weight in score_dict.items():
                if score_name not in score_functions:
                    raise KeyError(
                        f"Unsupported reward key {score_name!r}. "
                        f"Supported keys: {list(score_functions.keys())}"
                    )
                factory = score_functions[score_name]
                try:
                    with _preserve_root_logger_state():
                        fn = (
                            factory(device)
                            if "device" in factory.__code__.co_varnames
                            else factory()
                        )
                except Exception as e:
                    if not allow_unavailable:
                        raise
                    self._unavailable[score_name] = f"{type(e).__name__}: {e}"
                    warnings.warn(
                        f"[MultiScorer] Skip unavailable reward {score_name!r}: {type(e).__name__}: {e}",
                        stacklevel=2,
                    )
                    continue
                finally:
                    # Some reward backends mutate root logger state during first-time
                    # import/model construction; force MeanFlowNFT logging config back.
                    setup_logging()

                self.score_dict[score_name] = weight
                self.score_fns[score_name] = fn
                if isinstance(fn, torch.nn.Module):
                    self._scorers.append(fn)
                elif hasattr(fn, "__closure__") and fn.__closure__:
                    for cell in fn.__closure__:
                        try:
                            obj = cell.cell_contents
                            if isinstance(obj, torch.nn.Module):
                                self._scorers.append(obj)
                        except ValueError:
                            pass

        self.active_reward_names = list(self.score_dict.keys())
        self.unavailable_rewards = dict(self._unavailable)

    def __call__(self, images, prompts, metadata=None, only_strict=True):
        del only_strict
        if metadata is None:
            metadata = {}
        total_scores = []
        score_details = {}

        for score_name, weight in self.score_dict.items():
            scores, rewards = self.score_fns[score_name](
                images, prompts, metadata
            )
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["mean"] = total_scores
        return score_details, {}

    def to(self, target_device):
        for scorer in self._scorers:
            scorer.to(target_device)
            if hasattr(scorer, "device"):
                scorer.device = target_device
        return self
