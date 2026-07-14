"""
OCR-based text rendering accuracy scorer.

Uses PaddleOCR to recognize text in generated images and compares against
the target text extracted from the prompt (text between quotes).

Adapted from DiffusionNFT's OCR reward.
"""

import os
import time
import fcntl
import logging
from pathlib import Path
from contextlib import contextmanager
import torch
import numpy as np
from typing import List, Union
from PIL import Image
from Levenshtein import distance
from meanflownft.rewards import reward_ckpt_path


class OcrScorer:
    def __init__(self, use_gpu: bool = False, init_retries: int = 3):
        """
        OCR reward calculator.
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        cache_root = _resolve_ocr_cache_root()
        # Keep PaddleOCR cache location explicit and configurable.
        os.makedirs(cache_root, exist_ok=True)
        lock_path = os.path.join(cache_root, ".meanflownft_paddleocr_init.lock")
        self.ocr = None

        for attempt_idx in range(init_retries):
            try:
                # Prefer parallel init across ranks; only synchronize on rare
                # temporary-file download races.
                self.ocr = _init_paddleocr(cache_root=cache_root, use_gpu=use_gpu)
                break
            except FileNotFoundError as e:
                # PaddleOCR download can race on *.tar.tmp under multi-process init.
                # Serialize only the recovery path, not the full initialization.
                if ".tmp" in str(e) and attempt_idx + 1 < init_retries:
                    with _file_lock(lock_path):
                        time.sleep(0.5)
                    time.sleep(1.5 * (attempt_idx + 1))
                    continue
                raise

        if self.ocr is None:
            raise RuntimeError("Failed to initialize PaddleOCR scorer")

    @torch.no_grad()
    def __call__(self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]) -> list:
        """
        Calculate OCR reward.
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: List of reward scores
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        rewards = []
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        for img, prompt in zip(images, prompts):
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                result = self.ocr.ocr(img, cls=False)
                recognized_text = (
                    "".join([res[1][0] if res[1][1] > 0 else "" for res in result[0]]) if result[0] else ""
                )
                recognized_text = recognized_text.replace(" ", "").lower()
                prompt = prompt.replace(" ", "").lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                if dist > len(prompt):
                    dist = len(prompt)
            except Exception as e:
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)
            reward = 1 - dist / (len(prompt))
            rewards.append(reward)

        return rewards


def _resolve_ocr_cache_root() -> str:
    """Resolve OCR cache root path.

    Priority:
    1) MEANFLOWNFT_PADDLEOCR_HOME
    2) <reward_ckpt_path.CKPT_PATH>/.paddleocr
    3) <repo_root>/.paddleocr (legacy fallback)
    """
    explicit = os.environ.get("MEANFLOWNFT_PADDLEOCR_HOME", "").strip()
    if explicit:
        return os.path.expanduser(explicit)

    legacy_root = str(Path(__file__).resolve().parents[2] / ".paddleocr")
    ckpt_root = str(getattr(reward_ckpt_path, "CKPT_PATH", "") or "").strip()
    if ckpt_root:
        candidate = os.path.join(os.path.expanduser(ckpt_root), ".paddleocr")
        if _looks_like_populated_ocr_cache(candidate):
            return candidate
        if _looks_like_populated_ocr_cache(legacy_root):
            return legacy_root
        return candidate

    return legacy_root


def _looks_like_populated_ocr_cache(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    # PaddleOCR cache is typically under whl/{det,rec,cls} with model files.
    patterns = ("*.pdiparams", "*.pdmodel", "*.json", "*.onnx")
    return any(any(p.rglob(pattern)) for pattern in patterns)


def _init_paddleocr(cache_root: str, use_gpu: bool):
    # PaddleOCR mutates root logger level during import/init.
    # Preserve MeanFlowNFT logging verbosity (INFO on rank0).
    with _preserve_root_logger_state():
        from paddleocr import PaddleOCR
        # PaddleOCR uses this module-level constant to build default model dirs.
        import paddleocr.paddleocr as paddleocr_module

        paddleocr_module.BASE_DIR = os.path.join(cache_root, "")
        return PaddleOCR(use_angle_cls=False, lang="en", use_gpu=use_gpu, show_log=False)


@contextmanager
def _preserve_root_logger_state():
    root = logging.getLogger()
    level = root.level
    disabled = root.disabled
    handler_levels = [(handler, handler.level) for handler in list(root.handlers)]
    try:
        yield
    finally:
        root.setLevel(level)
        root.disabled = disabled
        for handler, handler_level in handler_levels:
            handler.setLevel(handler_level)


@contextmanager
def _file_lock(lock_path: str):
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
