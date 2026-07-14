"""Standalone SD3.5-Medium and MeanFlowNFT inference.

``num_stages`` selects a contiguous training prefix:

    0: clean SD3.5-Medium, standard diffusers-style Euler + CFG
    1: base + Stage 1 AnyFlow-pretrain LoRA
    2: base + Stage 1 + Stage 2 AnyFlow on-policy LoRAs
    3: base + Stage 1 + Stage 2 + Stage 3 MeanFlowNFT LoRAs

LoRAs are always loaded in order; Stage 2 or Stage 3 can never be loaded
alone. The flow-map ``delta_embedder`` comes from the latest loaded stage
because it is a full state rather than an additive LoRA delta.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image

from meanflownft.config import LoRAConfig
from meanflownft.models.sd35 import (
    encode_prompts_sd35,
    load_sd35_models,
    predict_noise_sd35,
)
from meanflownft.models.sd35_flowmap import (
    predict_noise_sd35_flowmap,
    setup_flowmap_for_sd3,
)
from meanflownft.parallel.utils import (
    barrier,
    cleanup_distributed,
    ddp_wrap_model,
    fsdp_wrap_model,
    get_rank,
    get_transformer_wrap_policy,
    get_world_size,
    is_main_process,
    setup_distributed,
)
from meanflownft.schedulers.flowmap_scheduler import FlowMapScheduler
from meanflownft.utils.image import decode_latents_to_tensor
from meanflownft.utils.lora import inject_lora

logger = logging.getLogger(__name__)


SD35_MMDIT_LORA_TARGET_MODULES = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.add_q_proj",
    "attn.to_add_out",
]

SUPPORTED_REWARDS = {
    "pickscore",
    "hpsv2",
    "hpsv3",
    "clipscore",
    "aesthetic",
    "imagereward",
    "ocr",
    "geneval2",
}

_LORA_KEY_RE = re.compile(
    r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$"
)
_STATE_PREFIXES = (
    "module.",
    "_orig_mod.",
    "_fsdp_wrapped_module.",
    "base_model.model.",
    "transformer.",
)


def _strip_state_prefixes(name: str) -> str:
    """Normalize common FSDP, DDP, diffusers, and PEFT key prefixes."""
    normalized = name
    while True:
        for prefix in _STATE_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        else:
            return normalized


def _adapter_weight_path(directory: str) -> str | None:
    for filename in (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_model.pt",
        "adapter_model.pth",
    ):
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def _resolve_lora_path(path: str, stage_name: str) -> str:
    """Resolve a LoRA-only file or a PEFT adapter directory."""
    raw = os.path.expanduser(str(path).strip())
    if not raw:
        raise ValueError(f"{stage_name}_lora_path is empty.")
    resolved = os.path.abspath(raw)

    if os.path.isfile(resolved):
        if os.path.splitext(resolved)[1].lower() not in {".pt", ".pth"}:
            raise ValueError(
                f"{stage_name}_lora_path must be a .pt/.pth LoRA-only file "
                f"or PEFT adapter directory, got: {resolved}"
            )
        return resolved

    if os.path.isdir(resolved):
        if _adapter_weight_path(resolved) is not None:
            return resolved
        nested = os.path.join(resolved, "transformer")
        if os.path.isdir(nested) and _adapter_weight_path(nested) is not None:
            return nested
        raise FileNotFoundError(
            f"{stage_name}_lora_path directory has no PEFT adapter weights: "
            f"{resolved}. Expected adapter_model.safetensors/.bin/.pt/.pth "
            "directly or under transformer/."
        )

    raise FileNotFoundError(f"{stage_name}_lora_path does not exist: {resolved}")


def _load_tensor_state_dict(resolved_path: str) -> tuple[dict[str, torch.Tensor], str]:
    """Load tensors from a LoRA-only checkpoint or PEFT adapter directory."""
    if os.path.isdir(resolved_path):
        weight_path = _adapter_weight_path(resolved_path)
        if weight_path is None:
            raise FileNotFoundError(f"No adapter weights found in {resolved_path}")
        source_format = "peft"
    else:
        weight_path = resolved_path
        source_format = "pt"

    if weight_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as load_safetensors
        except ImportError as exc:
            raise ImportError(
                f"safetensors is required to load {weight_path}"
            ) from exc
        payload: Any = load_safetensors(weight_path, device="cpu")
    else:
        payload = torch.load(weight_path, map_location="cpu", weights_only=False)

    if (
        isinstance(payload, dict)
        and isinstance(payload.get("state_dict"), dict)
    ):
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Checkpoint is not a non-empty state_dict: {weight_path}")

    state: dict[str, torch.Tensor] = {}
    invalid: list[str] = []
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            invalid.append(str(key))
        else:
            state[key] = value
    if invalid:
        raise ValueError(
            f"Checkpoint contains non-tensor state entries: {weight_path}; "
            f"examples={invalid[:5]}"
        )
    return state, source_format


def _read_adapter_config(resolved_path: str) -> dict[str, Any] | None:
    if not os.path.isdir(resolved_path):
        return None
    path = os.path.join(resolved_path, "adapter_config.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"PEFT adapter config must be an object: {path}")
    return data


def _expected_lora_parameters(
    model: nn.Module,
    target_modules: list[str],
) -> dict[str, nn.Parameter]:
    expected: dict[str, nn.Parameter] = {}
    matched_targets = {target: 0 for target in target_modules}
    for name, parameter in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        module_name = name[:-len(".weight")]
        for target in target_modules:
            if module_name.endswith(target):
                expected[name] = parameter
                matched_targets[target] += 1
                break

    missing_targets = [
        target for target, count in matched_targets.items() if count == 0
    ]
    if missing_targets:
        raise RuntimeError(
            "Configured SD3 MMDiT LoRA targets did not match the clean "
            f"transformer: {missing_targets}"
        )
    return expected


def _expected_delta_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    expected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if "delta_embedder" in name
    }
    if not expected:
        raise RuntimeError(
            "Flow-map wrapper has no delta_embedder parameters. "
            "setup_flowmap_for_sd3 must run before loading checkpoints."
        )
    return expected


def _resolve_stage_lora_config(
    adapter_config: dict[str, Any] | None,
    *,
    resolved_path: str,
    fallback_rank: int,
    fallback_alpha: float,
    fallback_target_modules: list[str],
) -> tuple[int, float, list[str]]:
    adapter_config = adapter_config or {}
    if bool(adapter_config.get("use_rslora", False)):
        raise ValueError(
            f"RS-LoRA scaling is not supported by this release loader: "
            f"{resolved_path}"
        )
    if adapter_config.get("rank_pattern") or adapter_config.get("alpha_pattern"):
        raise ValueError(
            f"Per-module PEFT rank/alpha patterns are not supported: "
            f"{resolved_path}"
        )
    rank = int(adapter_config.get("r", fallback_rank))
    alpha = float(adapter_config.get("lora_alpha", fallback_alpha))
    raw_targets = adapter_config.get(
        "target_modules", fallback_target_modules
    )
    if not isinstance(raw_targets, (list, tuple, set)):
        raise TypeError(
            f"PEFT target_modules must be a sequence in {resolved_path}"
        )
    target_modules = [str(item) for item in raw_targets]
    if rank <= 0 or alpha <= 0:
        raise ValueError(
            f"Invalid LoRA rank/alpha for {resolved_path}: "
            f"rank={rank}, alpha={alpha}"
        )
    if not target_modules or any(not item for item in target_modules):
        raise ValueError(
            f"LoRA target_modules cannot be empty for {resolved_path}"
        )
    return rank, alpha, target_modules


def _parse_stage_state(
    state: dict[str, torch.Tensor],
    *,
    stage_name: str,
    resolved_path: str,
    expected_lora: dict[str, nn.Parameter],
    expected_delta: dict[str, nn.Parameter],
    rank: int,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, torch.Tensor],
]:
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    delta_state: dict[str, torch.Tensor] = {}
    unrecognized: list[str] = []

    for raw_key, tensor in state.items():
        key = _strip_state_prefixes(raw_key)
        match = _LORA_KEY_RE.match(key)
        if match:
            base_key = f"{match.group('module')}.weight"
            side = match.group("side")
            pair = pairs.setdefault(base_key, {})
            if side in pair:
                raise ValueError(
                    f"{stage_name} checkpoint has duplicate LoRA {side} "
                    f"for {base_key}: {resolved_path}"
                )
            pair[side] = tensor
        elif "delta_embedder" in key:
            if key in delta_state:
                raise ValueError(
                    f"{stage_name} checkpoint has duplicate delta key {key}"
                )
            delta_state[key] = tensor
        else:
            unrecognized.append(raw_key)

    if unrecognized:
        raise ValueError(
            f"{stage_name} checkpoint is not LoRA/flow-map-only; "
            f"{len(unrecognized)} unmatched keys in {resolved_path}, "
            f"examples={unrecognized[:8]}"
        )
    if not pairs:
        raise ValueError(
            f"{stage_name} checkpoint has no LoRA A/B tensors: {resolved_path}"
        )

    incomplete = sorted(
        base_key
        for base_key, pair in pairs.items()
        if set(pair) != {"A", "B"}
    )
    if incomplete:
        raise ValueError(
            f"{stage_name} checkpoint has incomplete LoRA pairs in "
            f"{resolved_path}: {incomplete[:8]}"
        )

    actual_lora = set(pairs)
    expected_lora_names = set(expected_lora)
    missing_lora = sorted(expected_lora_names - actual_lora)
    unmatched_lora = sorted(actual_lora - expected_lora_names)
    if missing_lora or unmatched_lora:
        raise RuntimeError(
            f"{stage_name} LoRA tensors do not exactly match the clean SD3.5 "
            f"MMDiT targets in {resolved_path}: missing={missing_lora[:8]} "
            f"(total {len(missing_lora)}), unmatched={unmatched_lora[:8]} "
            f"(total {len(unmatched_lora)})"
        )

    for base_key, pair in pairs.items():
        parameter = expected_lora[base_key]
        lora_a, lora_b = pair["A"], pair["B"]
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(
                f"{stage_name} expects 2-D linear LoRA tensors for {base_key}; "
                f"got A{tuple(lora_a.shape)}, B{tuple(lora_b.shape)}"
            )
        if lora_a.shape[0] != rank or lora_b.shape[1] != rank:
            raise ValueError(
                f"{stage_name} rank mismatch for {base_key}: "
                f"A{tuple(lora_a.shape)}, B{tuple(lora_b.shape)}, rank={rank}"
            )
        expected_shape = (lora_b.shape[0], lora_a.shape[1])
        if tuple(parameter.shape) != expected_shape:
            raise ValueError(
                f"{stage_name} LoRA shape mismatch for {base_key}: "
                f"B@A={expected_shape}, base={tuple(parameter.shape)}"
            )

    delta_names = set(delta_state)
    expected_delta_names = set(expected_delta)
    unmatched_delta = sorted(delta_names - expected_delta_names)
    if unmatched_delta:
        raise RuntimeError(
            f"{stage_name} has unmatched delta_embedder tensors in "
            f"{resolved_path}: {unmatched_delta}"
        )
    if delta_state and delta_names != expected_delta_names:
        missing_delta = sorted(expected_delta_names - delta_names)
        raise RuntimeError(
            f"{stage_name} has a partial delta_embedder state in "
            f"{resolved_path}; missing={missing_delta}"
        )
    for key, tensor in delta_state.items():
        if tuple(tensor.shape) != tuple(expected_delta[key].shape):
            raise ValueError(
                f"{stage_name} delta_embedder shape mismatch for {key}: "
                f"checkpoint={tuple(tensor.shape)}, "
                f"model={tuple(expected_delta[key].shape)}"
            )

    return pairs, delta_state


def _merge_stage_lora(
    model: nn.Module,
    *,
    stage_name: str,
    path: str,
    rank: int,
    alpha: float,
    target_modules: list[str],
    load_delta_embedder: bool,
) -> dict[str, Any]:
    """Fold one stage LoRA into ``model`` and optionally load its delta state."""
    resolved_path = _resolve_lora_path(path, stage_name)
    state, source_format = _load_tensor_state_dict(resolved_path)
    rank, alpha, target_modules = _resolve_stage_lora_config(
        _read_adapter_config(resolved_path),
        resolved_path=resolved_path,
        fallback_rank=rank,
        fallback_alpha=alpha,
        fallback_target_modules=target_modules,
    )

    expected_lora = _expected_lora_parameters(model, target_modules)
    expected_delta = _expected_delta_parameters(model)
    pairs, delta_state = _parse_stage_state(
        state,
        stage_name=stage_name,
        resolved_path=resolved_path,
        expected_lora=expected_lora,
        expected_delta=expected_delta,
        rank=rank,
    )

    if load_delta_embedder and set(delta_state) != set(expected_delta):
        missing = sorted(set(expected_delta) - set(delta_state))
        raise RuntimeError(
            f"{stage_name} is the latest checkpoint and must contain the full "
            f"flow-map delta_embedder state; missing={missing}, path={resolved_path}"
        )

    scaling = float(alpha) / float(rank)
    merged_names: list[str] = []
    with torch.no_grad():
        for base_key in sorted(pairs):
            parameter = expected_lora[base_key]
            lora_a = pairs[base_key]["A"].to(
                device=parameter.device, dtype=torch.float32
            )
            lora_b = pairs[base_key]["B"].to(
                device=parameter.device, dtype=torch.float32
            )
            update = torch.matmul(lora_b, lora_a)
            parameter.add_(update.to(dtype=parameter.dtype), alpha=scaling)
            merged_names.append(base_key)

        loaded_delta_names: list[str] = []
        if load_delta_embedder:
            for key in sorted(expected_delta):
                parameter = expected_delta[key]
                parameter.copy_(
                    delta_state[key].to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                )
                loaded_delta_names.append(key)

    ignored_delta_names = (
        sorted(delta_state) if not load_delta_embedder else []
    )
    if is_main_process():
        logger.info(
            "[%s] merged %d LoRA deltas from %s "
            "(format=%s, rank=%d, alpha=%g, scale=%.6f):\n  %s",
            stage_name,
            len(merged_names),
            resolved_path,
            source_format,
            rank,
            alpha,
            scaling,
            "\n  ".join(merged_names),
        )
        if ignored_delta_names:
            logger.info(
                "[%s] deliberately ignored superseded delta_embedder tensors:\n  %s",
                stage_name,
                "\n  ".join(ignored_delta_names),
            )
        if loaded_delta_names:
            logger.info(
                "[%s] loaded latest flow-map delta_embedder tensors:\n  %s",
                stage_name,
                "\n  ".join(loaded_delta_names),
            )

    return {
        "stage": stage_name,
        "path": resolved_path,
        "format": source_format,
        "rank": rank,
        "alpha": alpha,
        "scaling": scaling,
        "merged_lora_tensors": merged_names,
        "ignored_superseded_delta_tensors": ignored_delta_names,
        "loaded_delta_tensors": loaded_delta_names,
    }


def _load_active_stage_lora(
    model: nn.Module,
    *,
    stage_name: str,
    path: str,
    rank: int,
    alpha: float,
    target_modules: list[str],
) -> dict[str, Any]:
    """Load the latest stage as an active adapter, matching training eval."""
    resolved_path = _resolve_lora_path(path, stage_name)
    state, source_format = _load_tensor_state_dict(resolved_path)
    rank, alpha, target_modules = _resolve_stage_lora_config(
        _read_adapter_config(resolved_path),
        resolved_path=resolved_path,
        fallback_rank=rank,
        fallback_alpha=alpha,
        fallback_target_modules=target_modules,
    )

    expected_lora = _expected_lora_parameters(model, target_modules)
    expected_delta = _expected_delta_parameters(model)
    pairs, delta_state = _parse_stage_state(
        state,
        stage_name=stage_name,
        resolved_path=resolved_path,
        expected_lora=expected_lora,
        expected_delta=expected_delta,
        rank=rank,
    )
    if set(delta_state) != set(expected_delta):
        missing = sorted(set(expected_delta) - set(delta_state))
        raise RuntimeError(
            f"{stage_name} must contain the full flow-map delta state; "
            f"missing={missing}, path={resolved_path}"
        )

    inject_lora(
        model,
        LoRAConfig(
            enabled=True,
            rank=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            init_lora_weights="gaussian",
        ),
    )

    normalized_state: dict[str, torch.Tensor] = {}
    for raw_key, tensor in state.items():
        key = _strip_state_prefixes(raw_key)
        match = _LORA_KEY_RE.match(key)
        if match:
            key = (
                f"{match.group('module')}.lora_{match.group('side')}"
                ".default.weight"
            )
        if key in normalized_state:
            raise ValueError(
                f"{stage_name} checkpoint normalizes to duplicate key {key}"
            )
        normalized_state[key] = tensor

    load_result = model.load_state_dict(normalized_state, strict=False)
    unexpected = sorted(
        key
        for key in getattr(load_result, "unexpected_keys", [])
        if key in normalized_state
    )
    if unexpected:
        raise RuntimeError(
            f"{stage_name} active adapter keys failed to load: "
            f"{unexpected[:8]} (total {len(unexpected)})"
        )

    scaling = float(alpha) / float(rank)
    active_names = sorted(pairs)
    loaded_delta_names = sorted(delta_state)
    if is_main_process():
        logger.info(
            "[%s] loaded %d active LoRA adapters from %s "
            "(format=%s, rank=%d, alpha=%g, scale=%.6f)",
            stage_name,
            len(active_names),
            resolved_path,
            source_format,
            rank,
            alpha,
            scaling,
        )
        logger.info(
            "[%s] loaded latest flow-map delta_embedder tensors:\n  %s",
            stage_name,
            "\n  ".join(loaded_delta_names),
        )

    return {
        "stage": stage_name,
        "path": resolved_path,
        "format": source_format,
        "rank": rank,
        "alpha": alpha,
        "scaling": scaling,
        "active_lora_tensors": active_names,
        "loaded_delta_tensors": loaded_delta_names,
    }


def _apply_flat_overrides(
    data: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    for dotted_key, value in overrides.items():
        target = data
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            current = target.get(part)
            if current is None:
                current = {}
                target[part] = current
            if not isinstance(current, dict):
                raise ValueError(
                    f"Cannot apply nested override {dotted_key!r}: "
                    f"{part!r} is not a mapping"
                )
            target = current
        target[parts[-1]] = value
    return data


@dataclass
class InferenceConfig:
    """Configuration for SD3.5 and the three-stage MeanFlowNFT prefix."""

    pretrained_path: str = ""
    dtype: str = "bf16"
    image_resolution: int = 512

    num_stages: int = 3
    stage1_lora_path: str = ""
    stage2_lora_path: str = ""
    stage3_lora_path: str = ""
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(SD35_MMDIT_LORA_TARGET_MODULES)
    )

    flowmap_gate_value: float = 0.25
    flowmap_deltatime_type: str = "r"
    num_steps: int = 4
    normal_num_steps: int = 40
    normal_guidance_scale: float = 4.5
    # 0 selects the training-aligned default: 128 for legacy MeanFlowNFT
    # Stage 3, 256 for clean SD3.5 and AnyFlow Stage 1/2.
    text_max_sequence_length: int = 0
    seed: int = 12345
    batch_size: int = 4

    prompts: list[str] = field(default_factory=list)
    prompt_file: str = "dataset/pickscore/test.txt"
    max_prompts: int = 64

    output_dir: str = "./inference_outputs/sd35m_meanflow_nft/steps_4"

    eval_reward: bool = False
    reward_fn: dict[str, float] = field(
        default_factory=lambda: {"pickscore": 1.0}
    )
    reward_dataset_map: dict[str, str] = field(default_factory=dict)
    dataset_path: str = "dataset"
    reward_ckpt_path: str = ""

    distributed: bool = False
    strategy: str = "fsdp"

    @classmethod
    def from_yaml(
        cls,
        path: str,
        overrides: dict[str, Any] | None = None,
    ) -> "InferenceConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError(f"Inference YAML must contain a mapping: {path}")
        if overrides:
            raw = _apply_flat_overrides(raw, overrides)

        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown SD3.5 MeanFlowNFT inference config fields: {unknown}"
            )
        return cls(**raw)

    def validate(self) -> None:
        raw_pretrained_path = str(self.pretrained_path).strip()
        if not raw_pretrained_path:
            raise ValueError("pretrained_path is required")
        self.pretrained_path = os.path.abspath(
            os.path.expanduser(raw_pretrained_path)
        )
        if not os.path.isdir(self.pretrained_path):
            raise FileNotFoundError(
                f"SD3.5-Medium pretrained_path is not a directory: "
                f"{self.pretrained_path}"
            )
        missing_components = [
            component
            for component in ("transformer", "vae", "scheduler")
            if not os.path.isdir(os.path.join(self.pretrained_path, component))
        ]
        if missing_components:
            raise FileNotFoundError(
                "pretrained_path is not a complete SD3.5 pipeline; missing "
                f"subdirectories={missing_components}: {self.pretrained_path}"
            )

        self.num_stages = int(self.num_stages)
        if self.num_stages not in {0, 1, 2, 3}:
            raise ValueError("num_stages must be one of 0, 1, 2, or 3")

        # Resolve exactly the requested contiguous prefix. Later configured
        # paths may remain populated so one YAML can evaluate every prefix.
        stage_errors: list[str] = []
        loaded_stage_paths: list[str] = []
        for stage_index, (field_name, stage_name) in enumerate((
            ("stage1_lora_path", "stage1"),
            ("stage2_lora_path", "stage2"),
            ("stage3_lora_path", "stage3"),
        ), start=1):
            if stage_index > self.num_stages:
                continue
            try:
                resolved = _resolve_lora_path(
                    getattr(self, field_name), stage_name
                )
                setattr(self, field_name, resolved)
                loaded_stage_paths.append(resolved)
            except (TypeError, ValueError, FileNotFoundError) as exc:
                stage_errors.append(f"{field_name}: {exc}")
        if stage_errors:
            raise ValueError(
                f"num_stages={self.num_stages} requires the first "
                f"{self.num_stages} ordered LoRA checkpoint(s):\n  "
                + "\n  ".join(stage_errors)
            )
        if len(set(loaded_stage_paths)) != len(loaded_stage_paths):
            raise ValueError(
                "Loaded stage LoRA paths must resolve to distinct checkpoints."
            )

        self.lora_rank = int(self.lora_rank)
        self.lora_alpha = float(self.lora_alpha)
        self.num_steps = int(self.num_steps)
        self.normal_num_steps = int(self.normal_num_steps)
        self.normal_guidance_scale = float(self.normal_guidance_scale)
        self.text_max_sequence_length = int(self.text_max_sequence_length)
        if self.text_max_sequence_length == 0:
            self.text_max_sequence_length = (
                128 if self.num_stages == 3 else 256
            )
        self.seed = int(self.seed)
        self.batch_size = int(self.batch_size)
        self.image_resolution = int(self.image_resolution)
        self.max_prompts = int(self.max_prompts)
        self.flowmap_gate_value = float(self.flowmap_gate_value)
        self.dtype = str(self.dtype).lower()

        if self.num_stages > 0 and (
            self.lora_rank <= 0 or self.lora_alpha <= 0
        ):
            raise ValueError(
                "Fallback LoRA rank and alpha must both be positive."
            )
        if self.num_stages > 0 and (
            not isinstance(self.lora_target_modules, list)
            or not self.lora_target_modules
            or any(
                not isinstance(module, str) or not module
                for module in self.lora_target_modules
            )
        ):
            raise ValueError(
                "Fallback lora_target_modules must be a non-empty string list."
            )
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be one of: bf16, fp16, fp32")
        if self.flowmap_deltatime_type not in {"r", "t-r"}:
            raise ValueError("flowmap_deltatime_type must be 'r' or 't-r'")
        if not 0.0 <= self.flowmap_gate_value <= 1.0:
            raise ValueError("flowmap_gate_value must be in [0, 1]")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.normal_num_steps < 1:
            raise ValueError("normal_num_steps must be >= 1")
        if self.normal_guidance_scale < 1.0:
            raise ValueError("normal_guidance_scale must be >= 1.0")
        if self.text_max_sequence_length < 1:
            raise ValueError("text_max_sequence_length must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.image_resolution < 8 or self.image_resolution % 8 != 0:
            raise ValueError("image_resolution must be a positive multiple of 8")
        if self.max_prompts < 0:
            raise ValueError("max_prompts must be >= 0 (0 means unlimited)")

        if not isinstance(self.prompts, list):
            raise TypeError("prompts must be a list")
        self.prompts = [str(prompt).strip() for prompt in self.prompts]
        if any(not prompt for prompt in self.prompts):
            raise ValueError("prompts cannot contain empty strings")
        if not self.prompts and not str(self.prompt_file).strip():
            raise ValueError("Set prompts or prompt_file")

        if not isinstance(self.reward_fn, dict):
            raise TypeError("reward_fn must be a mapping")
        if not isinstance(self.reward_dataset_map, dict):
            raise TypeError("reward_dataset_map must be a mapping")
        unsupported = sorted(set(self.reward_fn) - SUPPORTED_REWARDS)
        if unsupported:
            raise ValueError(
                f"Unsupported rewards: {unsupported}; "
                f"supported={sorted(SUPPORTED_REWARDS)}"
            )
        for name, weight in self.reward_fn.items():
            if not isinstance(weight, (int, float)):
                raise TypeError(f"Reward weight for {name!r} must be numeric")
        unknown_dataset_rewards = sorted(
            set(self.reward_dataset_map) - set(self.reward_fn)
        )
        if unknown_dataset_rewards:
            raise ValueError(
                "reward_dataset_map contains keys absent from reward_fn: "
                f"{unknown_dataset_rewards}"
            )
        invalid_datasets = {
            name: value
            for name, value in self.reward_dataset_map.items()
            if not isinstance(value, str) or not value.strip()
        }
        if invalid_datasets:
            raise ValueError(
                "reward_dataset_map values must be non-empty dataset names: "
                f"{invalid_datasets}"
            )
        if self.eval_reward and not self.reward_fn:
            raise ValueError("eval_reward=true requires a non-empty reward_fn")
        if not str(self.dataset_path).strip():
            raise ValueError("dataset_path cannot be empty")
        if not str(self.output_dir).strip():
            raise ValueError("output_dir cannot be empty")

        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if env_world_size > 1:
            self.distributed = True
        if self.distributed:
            missing_env = [
                key
                for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
                if key not in os.environ
            ]
            if missing_env:
                raise RuntimeError(
                    "distributed=true requires torchrun environment variables: "
                    f"missing={missing_env}"
                )
        self.strategy = str(self.strategy).lower()
        if self.strategy not in {"fsdp", "ddp"}:
            raise ValueError("strategy must be 'fsdp' or 'ddp'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MeanFlowNFTInference:
    """SD3.5-Medium and MeanFlowNFT inference/reward evaluation."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoders: list[nn.Module] = []
        self.tokenizers: list[Any] = []
        self.base_scheduler = None
        self.flowmap_scheduler: FlowMapScheduler | None = None
        self.device: torch.device | None = None
        self.model_dtype: torch.dtype | None = None
        self.latent_channels = 0
        self.latent_size = 0
        self.merge_manifest: list[dict[str, Any]] = []
        self._closed = False

    def setup(self) -> "MeanFlowNFTInference":
        """Load SD3.5, apply the requested stage prefix, then wrap."""
        cfg = self.config
        cfg.validate()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SD3.5-Medium inference")
        if cfg.distributed:
            setup_distributed()

        self.device = torch.device("cuda", torch.cuda.current_device())
        if is_main_process():
            mode = (
                "normal SD3.5 diffusers-style inference"
                if cfg.num_stages == 0
                else f"MeanFlowNFT stage prefix 1..{cfg.num_stages}"
            )
            logger.info("Loading SD3.5-Medium: %s", mode)

        from meanflownft.config import ModelConfig

        components = load_sd35_models(
            ModelConfig(
                pretrained_path=cfg.pretrained_path,
                model_type="sd35",
                dtype=cfg.dtype,
                image_resolution=cfg.image_resolution,
                gradient_checkpointing=False,
            )
        )
        transformer = components["transformer"]
        self.vae = components["vae"]
        self.text_encoders = components["text_encoders"]
        self.tokenizers = components["tokenizers"]
        self.base_scheduler = components["scheduler"]

        if cfg.num_stages > 0:
            # The wrapper must exist before the latest full delta state loads.
            setup_flowmap_for_sd3(
                transformer,
                gate_value=cfg.flowmap_gate_value,
                deltatime_type=cfg.flowmap_deltatime_type,
            )

            stage_paths = (
                ("stage1", cfg.stage1_lora_path),
                ("stage2", cfg.stage2_lora_path),
                ("stage3", cfg.stage3_lora_path),
            )
            for stage_index, (stage_name, path) in enumerate(
                stage_paths[: cfg.num_stages], start=1
            ):
                if stage_index == cfg.num_stages:
                    stage_manifest = _load_active_stage_lora(
                        transformer,
                        stage_name=stage_name,
                        path=path,
                        rank=cfg.lora_rank,
                        alpha=cfg.lora_alpha,
                        target_modules=cfg.lora_target_modules,
                    )
                else:
                    stage_manifest = _merge_stage_lora(
                        transformer,
                        stage_name=stage_name,
                        path=path,
                        rank=cfg.lora_rank,
                        alpha=cfg.lora_alpha,
                        target_modules=cfg.lora_target_modules,
                        load_delta_embedder=False,
                    )
                self.merge_manifest.append(stage_manifest)

        transformer.eval()
        self.model_dtype = next(transformer.parameters()).dtype
        self.vae.requires_grad_(False).eval()
        for encoder in self.text_encoders:
            encoder.requires_grad_(False).eval()

        vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1
        )
        if cfg.image_resolution % vae_scale_factor != 0:
            raise ValueError(
                f"image_resolution={cfg.image_resolution} is not divisible by "
                f"the VAE scale factor {vae_scale_factor}"
            )
        self.latent_channels = int(transformer.config.in_channels)
        self.latent_size = cfg.image_resolution // vae_scale_factor

        scheduler_config = self.base_scheduler.config
        static_shift = float(getattr(scheduler_config, "shift", 1.0))
        if cfg.num_stages > 0:
            self.flowmap_scheduler = FlowMapScheduler(
                num_train_timesteps=int(
                    getattr(scheduler_config, "num_train_timesteps", 1000)
                ),
                shift=static_shift,
                weight_type="uniform",
            )

        self.vae.to(self.device)
        for encoder in self.text_encoders:
            encoder.to(self.device)

        if cfg.distributed and cfg.strategy == "fsdp":
            from diffusers.models.transformers.transformer_sd3 import (
                JointTransformerBlock,
            )

            transformer = fsdp_wrap_model(
                transformer,
                sharding_strategy="full_shard",
                fsdp_precision=cfg.dtype,
                auto_wrap_policy=get_transformer_wrap_policy(
                    JointTransformerBlock
                ),
            )
        elif cfg.distributed and cfg.strategy == "ddp":
            # DDP rejects a module whose parameters are already frozen, so
            # freeze immediately after wrapping. No backward pass is performed.
            transformer.to(self.device)
            transformer = ddp_wrap_model(transformer)
        else:
            transformer.to(self.device)

        transformer.requires_grad_(False).eval()
        self.transformer = transformer

        if is_main_process():
            if cfg.num_stages == 0:
                logger.info(
                    "Setup complete: normal SD3.5, resolution=%d, steps=%d, "
                    "guidance=%.3f, text_length=%d",
                    cfg.image_resolution,
                    cfg.normal_num_steps,
                    cfg.normal_guidance_scale,
                    cfg.text_max_sequence_length,
                )
            else:
                logger.info(
                    "Setup complete: num_stages=%d, resolution=%d, "
                    "latent=[%d,%d,%d], static_shift=%.6g, steps=%d, "
                    "guidance=1.0, text_length=%d",
                    cfg.num_stages,
                    cfg.image_resolution,
                    self.latent_channels,
                    self.latent_size,
                    self.latent_size,
                    static_shift,
                    cfg.num_steps,
                    cfg.text_max_sequence_length,
                )
        return self

    def _autocast(self):
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }.get(self.config.dtype)
        if dtype is None:
            return nullcontext()
        return torch.autocast("cuda", dtype=dtype)

    def _set_eval_seed(self) -> int:
        """Match BaseTrainer evaluation's fixed per-rank RNG setup."""
        eval_seed = int(self.config.seed) + get_rank()
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        torch.cuda.manual_seed_all(eval_seed)
        return eval_seed

    def _initial_noise(self, batch_size: int) -> torch.Tensor:
        assert self.device is not None and self.model_dtype is not None
        return torch.randn(
            (
                batch_size,
                self.latent_channels,
                self.latent_size,
                self.latent_size,
            ),
            device=self.device,
            dtype=self.model_dtype,
        )

    def _encode_prompts(
        self,
        prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.device is not None and self.model_dtype is not None
        prompt_embeds, pooled_embeds = encode_prompts_sd35(
            prompts,
            self.text_encoders,
            self.tokenizers,
            self.device,
            max_sequence_length=self.config.text_max_sequence_length,
        )
        return (
            prompt_embeds.to(dtype=self.model_dtype),
            pooled_embeds.to(dtype=self.model_dtype),
        )

    @torch.no_grad()
    def _generate_flowmap_batch(self, prompts: list[str]) -> torch.Tensor:
        if self.transformer is None or self.flowmap_scheduler is None:
            raise RuntimeError("Flow-map inference requires num_stages >= 1")
        assert self.device is not None

        prompt_embeds, pooled_embeds = self._encode_prompts(prompts)
        self.flowmap_scheduler.set_timesteps(
            self.config.num_steps,
            device=self.device,
        )
        timesteps = self.flowmap_scheduler.timesteps
        batch_size = len(prompts)
        latents = self._initial_noise(batch_size)

        for index in range(self.config.num_steps):
            timestep = timesteps[index].expand(batch_size).to(
                device=self.device, dtype=torch.float32
            )
            r_timestep = timesteps[index + 1].expand(batch_size).to(
                device=self.device, dtype=torch.float32
            )
            with self._autocast():
                velocity = predict_noise_sd35_flowmap(
                    model=self.transformer,
                    noisy_latents=latents,
                    text_embeddings=prompt_embeds,
                    timesteps=timestep,
                    pooled_prompt_embeds=pooled_embeds,
                    r_timesteps=r_timestep,
                    guidance_scale=1.0,
                )
            latents = self.flowmap_scheduler.step(
                velocity,
                latents,
                timestep,
                r_timestep,
            )
        return decode_latents_to_tensor(self.vae, latents)

    @torch.no_grad()
    def _generate_normal_batch(self, prompts: list[str]) -> torch.Tensor:
        """Standard SD3.5 FlowMatchEulerDiscreteScheduler inference with CFG."""
        if self.transformer is None or self.base_scheduler is None:
            raise RuntimeError("Call setup() before normal SD3.5 inference")
        assert self.device is not None

        prompt_embeds, pooled_embeds = self._encode_prompts(prompts)
        uncond_embeds, uncond_pooled = self._encode_prompts(
            [""] * len(prompts)
        )
        scheduler = copy.deepcopy(self.base_scheduler)
        scheduler.set_timesteps(
            self.config.normal_num_steps,
            device=self.device,
        )
        batch_size = len(prompts)
        latents = self._initial_noise(batch_size)

        for timestep_scalar in scheduler.timesteps:
            timesteps = timestep_scalar.expand(batch_size).to(
                device=self.device, dtype=torch.float32
            )
            with self._autocast():
                velocity = predict_noise_sd35(
                    model=self.transformer,
                    noisy_latents=latents,
                    text_embeddings=prompt_embeds,
                    timesteps=timesteps,
                    pooled_prompt_embeds=pooled_embeds,
                    guidance_scale=self.config.normal_guidance_scale,
                    uncond_text_embeddings=uncond_embeds,
                    uncond_pooled_prompt_embeds=uncond_pooled,
                )
            latents = scheduler.step(
                velocity,
                timestep_scalar,
                latents,
            ).prev_sample
        return decode_latents_to_tensor(self.vae, latents)

    @torch.no_grad()
    def _generate_batch_tensor(self, prompts: list[str]) -> torch.Tensor:
        if self.config.num_stages == 0:
            return self._generate_normal_batch(prompts)
        return self._generate_flowmap_batch(prompts)

    @staticmethod
    def _tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
        uint8_images = (
            (images * 255)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .permute(0, 2, 3, 1)
            .numpy()
        )
        return [Image.fromarray(image) for image in uint8_images]

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
    ) -> list[Image.Image]:
        """Generate one image per prompt without running the file pipeline."""
        self._set_eval_seed()
        return self._tensor_to_pil(
            self._generate_batch_tensor(prompts)
        )

    @staticmethod
    def _read_prompt_file(
        path: str,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")
        extension = os.path.splitext(path)[1].lower()
        if extension == ".json":
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError(f"Prompt JSON must contain a list: {path}")
            if not data:
                return [], None
            if all(isinstance(item, str) for item in data):
                return [str(item) for item in data], None
            if all(isinstance(item, dict) for item in data):
                prompts = [
                    str(item.get("prompt", item.get("text", "")))
                    for item in data
                ]
                return prompts, data
            raise ValueError(f"Prompt JSON has mixed/unsupported entries: {path}")

        if extension == ".jsonl":
            prompts: list[str] = []
            metadata: list[dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if isinstance(item, dict):
                        prompt = str(
                            item.get("prompt", item.get("text", ""))
                        )
                        metadata.append(item)
                    else:
                        prompt = str(item)
                        metadata.append({"text": prompt})
                    if not prompt:
                        raise ValueError(
                            f"Empty prompt at {path}:{line_number}"
                        )
                    prompts.append(prompt)
            return prompts, metadata

        with open(path, "r", encoding="utf-8") as handle:
            return [
                line.strip()
                for line in handle
                if line.strip()
            ], None

    def _load_prompts(
        self,
        dataset_override: str | None = None,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        cfg = self.config
        if dataset_override is not None:
            dataset_dir = os.path.join(
                cfg.dataset_path, dataset_override
            )
            prompt_path = ""
            for filename in (
                "test.txt",
                "test.jsonl",
                "prompts.json",
                "prompts.txt",
                "metadata.jsonl",
            ):
                candidate = os.path.join(dataset_dir, filename)
                if os.path.isfile(candidate):
                    prompt_path = candidate
                    break
            if not prompt_path:
                raise FileNotFoundError(
                    f"No retained evaluation prompt file found under "
                    f"{dataset_dir}"
                )
            prompts, metadata = self._read_prompt_file(prompt_path)
        elif cfg.prompts:
            prompts, metadata = list(cfg.prompts), None
        else:
            prompts, metadata = self._read_prompt_file(cfg.prompt_file)

        if cfg.max_prompts > 0:
            prompts = prompts[:cfg.max_prompts]
            if metadata is not None:
                metadata = metadata[:cfg.max_prompts]
        prompts = [str(prompt).strip() for prompt in prompts]
        if not prompts:
            source = dataset_override or cfg.prompt_file or "inline prompts"
            raise ValueError(f"No prompts loaded from {source}")
        if any(not prompt for prompt in prompts):
            raise ValueError("Prompt source contains an empty prompt")
        if metadata is not None and len(metadata) != len(prompts):
            raise RuntimeError("Prompt metadata is not aligned with prompts")
        return prompts, metadata

    @staticmethod
    def _safe_label(label: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
        return safe or "prompts"

    def _generate_prompt_set(
        self,
        *,
        prompts: list[str],
        metadata: list[dict[str, Any]] | None,
        dataset_label: str,
        filename_prefix: str,
    ) -> tuple[
        torch.Tensor | None,
        list[str],
        list[dict[str, Any]] | None,
        list[dict[str, Any]],
        float,
    ]:
        cfg = self.config
        rank = get_rank()
        world_size = get_world_size() if cfg.distributed else 1
        rank_indices = list(range(rank, len(prompts), world_size))
        max_local = (len(prompts) + world_size - 1) // world_size
        max_batches = (max_local + cfg.batch_size - 1) // cfg.batch_size

        local_images: list[torch.Tensor] = []
        local_prompts: list[str] = []
        local_metadata: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        elapsed = 0.0

        for batch_index in range(max_batches):
            start = batch_index * cfg.batch_size
            actual_indices = rank_indices[start:start + cfg.batch_size]
            actual_count = len(actual_indices)
            if actual_count:
                padded_indices = list(actual_indices)
            else:
                padded_indices = [0]
            padded_indices.extend(
                [padded_indices[-1]] * (cfg.batch_size - len(padded_indices))
            )
            batch_prompts = [prompts[index] for index in padded_indices]

            torch.cuda.synchronize()
            started = time.perf_counter()
            image_tensors = self._generate_batch_tensor(batch_prompts)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - started

            if actual_count == 0:
                continue
            image_tensors = image_tensors[:actual_count]
            images = self._tensor_to_pil(image_tensors)
            local_images.append(image_tensors)

            for image, global_index in zip(images, actual_indices):
                filename = f"{filename_prefix}{global_index:06d}.png"
                image.save(os.path.join(cfg.output_dir, filename))
                local_prompts.append(prompts[global_index])
                if metadata is not None:
                    local_metadata.append(metadata[global_index])
                records.append(
                    {
                        "dataset": dataset_label,
                        "index": global_index,
                        "filename": filename,
                        "prompt": prompts[global_index],
                        "eval_seed": int(cfg.seed) + rank,
                        "rank": rank,
                    }
                )

        images_tensor = (
            torch.cat(local_images, dim=0) if local_images else None
        )
        metadata_out = local_metadata if metadata is not None else None
        return (
            images_tensor,
            local_prompts,
            metadata_out,
            records,
            elapsed,
        )

    def _configure_reward_checkpoints(self) -> None:
        if self.config.reward_ckpt_path:
            from meanflownft.rewards.reward_ckpt_path import set_ckpt_path

            set_ckpt_path(self.config.reward_ckpt_path)

    def _score_reward(
        self,
        reward_name: str,
        images: torch.Tensor | None,
        prompts: list[str],
        metadata: list[dict[str, Any]] | None,
    ) -> float | None:
        """Score one reward at a time and aggregate valid values across ranks."""
        from meanflownft.rewards.multi_scorer import MultiScorer

        assert self.device is not None
        self._configure_reward_checkpoints()
        scorer = MultiScorer(
            device=torch.device("cpu"),
            score_dict={reward_name: self.config.reward_fn[reward_name]},
            allow_unavailable=True,
        )
        local_active = int(
            reward_name in getattr(scorer, "active_reward_names", [])
        )
        active = torch.tensor(local_active, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(active, op=dist.ReduceOp.MIN)
        if not bool(active.item()):
            if is_main_process():
                logger.warning(
                    "Reward %r is unavailable on at least one rank; skipped.",
                    reward_name,
                )
            return None

        scorer.to(self.device)
        local_error: Exception | None = None
        scores: Any = []
        if images is not None and prompts:
            try:
                details, _ = scorer(
                    images,
                    prompts,
                    metadata=metadata,
                    only_strict=False,
                )
                scores = details.get(reward_name, [])
            except Exception as exc:  # third-party scorer failures are optional
                local_error = exc

        failed = torch.tensor(
            int(local_error is not None),
            device=self.device,
        )
        if dist.is_initialized():
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if bool(failed.item()):
            if is_main_process():
                logger.warning(
                    "Reward %r failed on at least one rank and was skipped%s",
                    reward_name,
                    f": {local_error}" if local_error is not None else ".",
                )
            scorer.to(torch.device("cpu"))
            torch.cuda.empty_cache()
            return None

        if isinstance(scores, torch.Tensor):
            values = scores.detach().float().cpu().numpy().reshape(-1)
        elif isinstance(scores, (list, tuple)):
            values = np.asarray(
                [float(value) for value in scores],
                dtype=np.float64,
            ).reshape(-1)
        else:
            values = np.asarray(scores, dtype=np.float64).reshape(-1)
        valid = values[np.isfinite(values) & (values != -10)]
        totals = torch.tensor(
            [
                float(valid.sum()) if len(valid) else 0.0,
                float(len(valid)),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        scorer.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        if totals[1].item() == 0:
            if is_main_process():
                logger.warning("Reward %r produced no valid values.", reward_name)
            return None
        return float((totals[0] / totals[1]).item())

    @staticmethod
    def _gather_records(
        local_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not dist.is_initialized():
            return local_records
        gathered: list[list[dict[str, Any]] | None] = [
            None
        ] * dist.get_world_size()
        dist.all_gather_object(gathered, local_records)
        return [
            record
            for rank_records in gathered
            if rank_records is not None
            for record in rank_records
        ]

    def run(self) -> dict[str, Any]:
        """Generate PNGs, optionally evaluate rewards, and write metadata."""
        if self.transformer is None:
            raise RuntimeError("Call setup() before run()")
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        eval_seed = self._set_eval_seed()

        # A non-empty map selects per-reward datasets. Rewards without an
        # override use the direct/default prompt source.
        grouped_rewards: dict[str | None, list[str]] = {}
        if cfg.eval_reward:
            for reward_name in cfg.reward_fn:
                dataset = cfg.reward_dataset_map.get(reward_name)
                dataset_key = str(dataset).strip() if dataset else None
                grouped_rewards.setdefault(dataset_key, []).append(reward_name)
        else:
            grouped_rewards[None] = []

        use_multiple_sources = len(grouped_rewards) > 1 or any(
            key is not None for key in grouped_rewards
        )
        all_local_records: list[dict[str, Any]] = []
        scores: dict[str, float] = {}
        total_elapsed = 0.0

        for dataset_key, reward_names in grouped_rewards.items():
            prompts, metadata = self._load_prompts(dataset_key)
            dataset_label = dataset_key or "prompts"
            prefix = (
                f"{self._safe_label(dataset_label)}_"
                if use_multiple_sources
                else ""
            )
            if is_main_process():
                logger.info(
                    "Generating %d prompts for %s (rewards=%s)",
                    len(prompts),
                    dataset_label,
                    reward_names or "disabled",
                )
            (
                images,
                local_prompts,
                local_metadata,
                records,
                elapsed,
            ) = self._generate_prompt_set(
                prompts=prompts,
                metadata=metadata,
                dataset_label=dataset_label,
                filename_prefix=prefix,
            )
            all_local_records.extend(records)
            total_elapsed += elapsed

            for reward_name in reward_names:
                value = self._score_reward(
                    reward_name,
                    images,
                    local_prompts,
                    local_metadata,
                )
                if value is not None:
                    scores[reward_name] = value
            del images
            torch.cuda.empty_cache()

        if scores:
            scores["mean"] = sum(
                float(cfg.reward_fn[name]) * value
                for name, value in scores.items()
                if name != "mean"
            )

        records = self._gather_records(all_local_records)
        elapsed_tensor = torch.tensor(
            total_elapsed,
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_initialized():
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        wall_time = float(elapsed_tensor.item())

        records.sort(key=lambda item: (item["dataset"], item["index"]))
        result: dict[str, Any] = {
            "num_images": len(records),
            "time_seconds": wall_time,
        }
        if scores:
            result["scores"] = scores

        if is_main_process():
            if cfg.num_stages == 0:
                pipeline_name = "sd35m_normal"
                reconstruction = "clean SD3.5-Medium"
                guidance_scale = cfg.normal_guidance_scale
            else:
                pipeline_name = "sd35m_meanflow_nft_stage_prefix"
                reconstruction = (
                    "base + "
                    + " + ".join(
                        f"stage{index}_lora"
                        for index in range(1, cfg.num_stages + 1)
                    )
                    + f"; delta_embedder from stage{cfg.num_stages}"
                )
                guidance_scale = 1.0
            metadata = {
                "pipeline": pipeline_name,
                "reconstruction": reconstruction,
                "guidance_scale": guidance_scale,
                "eval_seed_rank0": eval_seed,
                "config": cfg.to_dict(),
                "merge_manifest": self.merge_manifest,
                "num_images": len(records),
                "time_seconds": wall_time,
                "images": records,
            }
            if scores:
                metadata["scores"] = scores
            metadata_path = os.path.join(cfg.output_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, default=str)
            logger.info("Saved %d PNGs and %s", len(records), metadata_path)

        self.close()
        return result

    def close(self, synchronize: bool = True) -> None:
        if self._closed:
            return
        if self.config.distributed and dist.is_initialized():
            if synchronize:
                barrier()
            cleanup_distributed()
        self._closed = True
