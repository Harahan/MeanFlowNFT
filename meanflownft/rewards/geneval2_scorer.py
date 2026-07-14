"""
GenEval2 Soft-TIFA reward scorer for MeanFlowNFT.

This module adapts the Soft-TIFA logic used in GenEval2/Flow-Factory so it can
be consumed by MeanFlowNFT's MultiScorer interface.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor
from meanflownft.rewards.reward_ckpt_path import CKPT_PATH

try:
    from transformers import Qwen3VLForConditionalGeneration as _QwenVLModel
except ImportError:
    from transformers import Qwen2_5_VLForConditionalGeneration as _QwenVLModel

try:
    from scipy.stats import gmean as _scipy_gmean
except ImportError:
    _scipy_gmean = None


def _gmean(values: Sequence[float]) -> float:
    if _scipy_gmean is not None:
        return float(_scipy_gmean(values))
    safe = [max(float(v), 1e-300) for v in values]
    return float(math.exp(sum(math.log(v) for v in safe) / max(len(safe), 1)))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_hf_model_dir(path: Path) -> bool:
    """Best-effort check for a local HF model directory."""
    return path.is_dir() and (path / "config.json").is_file()


def _resolve_model_name_or_path(model_name: str | None) -> str:
    """Resolve geneval2 model from explicit arg/env/reward_ckpt_path/default."""
    if model_name:
        return model_name

    env_model = os.environ.get("MEANFLOWNFT_GENEVAL2_MODEL_NAME")
    if env_model:
        return env_model

    ckpt_root = Path(CKPT_PATH).expanduser()
    candidates = [
        ckpt_root,  # reward_ckpt_path points directly to model dir
        ckpt_root / "geneval2",  # reward_ckpt_path points to a root with geneval2 subdir
        ckpt_root / "geneval2" / "Qwen3-VL-8B-Instruct",
        ckpt_root / "Qwen3-VL-8B-Instruct",
    ]
    for candidate in candidates:
        if _is_hf_model_dir(candidate):
            return str(candidate)

    # Fallback to HF Hub id.
    return "Qwen/Qwen3-VL-8B-Instruct"


def _resolve_lookup_paths(data_path: str | None) -> list[Path]:
    if data_path:
        p = Path(data_path).expanduser()
        if p.is_file():
            return [p]
        if p.is_dir():
            merged = p / "merged.jsonl"
            if merged.is_file():
                return [merged]
            paths = [q for q in (p / "train.jsonl", p / "test.jsonl") if q.is_file()]
            if paths:
                return paths
            raise FileNotFoundError(
                f"GenEval2 data directory has no merged/train/test jsonl: {p}"
            )
        raise FileNotFoundError(f"GenEval2 data_path not found: {p}")

    root = _repo_root()
    candidate_groups = [
        [root / "dataset" / "geneval2" / "merged.jsonl"],
        [
            root / "dataset" / "geneval2" / "train.jsonl",
            root / "dataset" / "geneval2" / "test.jsonl",
        ],
        [root / "dataset" / "GenEval2" / "synthetic" / "merged.jsonl"],
        [
            root / "dataset" / "GenEval2" / "synthetic" / "train.jsonl",
            root / "dataset" / "GenEval2" / "synthetic" / "test.jsonl",
        ],
    ]
    for group in candidate_groups:
        existing = [p for p in group if p.is_file()]
        if len(existing) == len(group):
            return existing

    raise FileNotFoundError(
        "Could not find GenEval2 JSONL data. Set MEANFLOWNFT_GENEVAL2_DATA_PATH to "
        "a .jsonl file or a directory with merged.jsonl or train.jsonl/test.jsonl."
    )


def _load_prompt_to_vqa(paths: Sequence[Path]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt = item.get("prompt")
                vqa_list = item.get("vqa_list")
                if prompt is None:
                    raise ValueError(f"{path}:{lineno}: missing 'prompt'")
                if vqa_list is None:
                    raise ValueError(f"{path}:{lineno}: missing 'vqa_list'")
                if prompt in out and out[prompt] != vqa_list:
                    raise ValueError(
                        f"Duplicate prompt with conflicting vqa_list: {prompt!r}"
                    )
                out[prompt] = vqa_list
    return out


def _return_numeric_string(number: str) -> str:
    mapping = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    return mapping.get(str(number).strip().lower(), "other")


def _construct_message_with_image(text: str, image_ref: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": text},
            ],
        }
    ]


class GenEval2SoftTIFAScorer(torch.nn.Module):
    """Soft-TIFA scorer that returns one reward per (prompt, image) pair."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = torch.float16,
        aggregation: str = "gm",
        model_name: str | None = None,
        data_path: str | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.aggregation = str(aggregation).lower()
        if self.aggregation not in ("am", "gm"):
            raise ValueError(f"aggregation must be 'am' or 'gm', got {aggregation!r}")

        self.model_name = _resolve_model_name_or_path(model_name)
        resolved_data_path = data_path or os.environ.get("MEANFLOWNFT_GENEVAL2_DATA_PATH")
        lookup_paths = _resolve_lookup_paths(resolved_data_path)
        self.prompt_to_vqa = _load_prompt_to_vqa(lookup_paths)

        # Qwen3-VL processor/model used by official-style Soft-TIFA evaluation.
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = _QwenVLModel.from_pretrained(
            self.model_name,
            torch_dtype="auto",
        )
        self.model = self.model.to(self.device)
        if self.dtype is not None:
            try:
                self.model = self.model.to(dtype=self.dtype)
            except (TypeError, RuntimeError):
                # Some backends/device combinations may not support explicit cast.
                pass
        self.model.eval()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (str, int, torch.device)):
                device = first
        if device is not None:
            self.device = torch.device(device)
        return self

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def _pil_to_temp_png(img: Image.Image) -> str:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(path, format="PNG")
        return path

    @staticmethod
    def _answer_variants(question: str, answer: str) -> list[str]:
        if question.startswith("How many"):
            number = _return_numeric_string(answer)
            return [
                answer,
                answer.capitalize(),
                f" {answer}",
                f" {answer.capitalize()}",
                number,
                f" {number}",
            ]
        return ["Yes", "yes", " Yes", " yes"]

    def _send_message_with_image(
        self,
        text: str,
        image_ref: str,
        answer_list: Sequence[str],
    ) -> float:
        messages = _construct_message_with_image(text, image_ref)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model_device())
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)

        ans_prob = 0.0
        for answer in answer_list:
            token_ids = self.processor.tokenizer.encode(
                answer,
                add_special_tokens=False,
            )
            if token_ids:
                ans_prob += float(probs[0, token_ids[0]].item())
        return ans_prob

    def _soft_tifa(self, vqa_list: Sequence[Sequence[str]], image_ref: str) -> list[float]:
        atom_scores: list[float] = []
        for qa in vqa_list:
            if not isinstance(qa, (list, tuple)) or len(qa) < 2:
                raise ValueError(f"Invalid vqa entry: {qa!r}")
            question, answer = str(qa[0]), str(qa[1])
            answer_list = self._answer_variants(question, answer)
            atom_score = self._send_message_with_image(
                f"{question} Answer in one word.",
                image_ref,
                answer_list,
            )
            atom_scores.append(atom_score)
        return atom_scores

    def _aggregate(self, atom_scores: Sequence[float]) -> float:
        if self.aggregation == "gm":
            return _gmean(atom_scores)
        return float(sum(atom_scores) / len(atom_scores))

    @torch.no_grad()
    def forward(
        self,
        prompts: Sequence[str],
        images: Sequence[Image.Image],
    ) -> torch.Tensor:
        if len(prompts) != len(images):
            raise ValueError(
                f"prompts/images length mismatch: {len(prompts)} vs {len(images)}"
            )

        rewards: list[float] = []
        for idx, (prompt, image) in enumerate(zip(prompts, images)):
            vqa_list = self.prompt_to_vqa.get(prompt)
            if vqa_list is None:
                raise KeyError(
                    f"Prompt not found in GenEval2 lookup map at index {idx}: {prompt!r}"
                )
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            image_path = self._pil_to_temp_png(image)
            try:
                atom_scores = self._soft_tifa(vqa_list, image_path)
            finally:
                try:
                    os.unlink(image_path)
                except OSError:
                    pass

            rewards.append(self._aggregate(atom_scores) if atom_scores else -10.0)

        return torch.tensor(
            rewards,
            device=self._model_device(),
            dtype=torch.float32,
        )
