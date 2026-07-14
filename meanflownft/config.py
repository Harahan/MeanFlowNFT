"""Strict configuration schema for Wan2.1 MeanFlowNFT training."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import yaml


def _apply_flat_overrides(data: dict, overrides: dict[str, Any]) -> dict:
    for key, value in overrides.items():
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(
                    f"Cannot apply override {key!r}: {part!r} is not a mapping."
                )
            target = child
        target[parts[-1]] = value
    return data


def _dataclass_from_dict(cls, data: Any):
    if not isinstance(data, dict):
        return data
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} field(s): {', '.join(unknown)}"
        )
    kwargs = {}
    for item in fields(cls):
        if item.name not in data:
            continue
        value = data[item.name]
        item_type = item.type
        if isinstance(item_type, str):
            item_type = eval(item_type, globals())
        if isinstance(value, dict) and hasattr(item_type, "__dataclass_fields__"):
            value = _dataclass_from_dict(item_type, value)
        kwargs[item.name] = value
    return cls(**kwargs)


@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 32
    lora_alpha: int = 64
    target_modules: list[str] = field(
        default_factory=lambda: [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "net.0.proj",
            "net.2",
        ]
    )
    init_lora_weights: str = "gaussian"
    load_path: str = ""
    pre_merge_paths: list[str] = field(default_factory=list)
    merge_before_training: bool = False

    def __post_init__(self) -> None:
        if self.rank < 1 or self.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive.")
        if self.enabled and not self.target_modules:
            raise ValueError("LoRA target_modules cannot be empty.")


@dataclass
class ModelConfig:
    pretrained_path: str = "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
    generator_init_path: str = ""
    model_type: str = "wan"
    gradient_checkpointing: bool = True
    dtype: str = "bf16"
    generator_lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self) -> None:
        if self.model_type.lower() != "wan":
            raise ValueError("The wan branch only supports model_type='wan'.")
        self.model_type = "wan"
        self.dtype = self.dtype.lower()
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("model.dtype must be bf16, fp16, or fp32.")
        if not str(self.pretrained_path).strip():
            raise ValueError("model.pretrained_path is required.")


@dataclass
class VideoConfig:
    num_frames: int = 81
    eval_num_frames: int = 81
    height: int = 480
    width: int = 832
    fps: int = 16
    max_sequence_length: int = 512

    def __post_init__(self) -> None:
        for name in ("num_frames", "eval_num_frames"):
            value = int(getattr(self, name))
            if value < 1 or (value - 1) % 4 != 0:
                raise ValueError(f"video.{name} must satisfy (value - 1) % 4 == 0.")
        if self.height < 8 or self.width < 8:
            raise ValueError("video height and width must be positive.")
        if self.height % 8 or self.width % 8:
            raise ValueError("video height and width must be divisible by 8.")
        if self.fps < 1 or self.max_sequence_length < 1:
            raise ValueError("video fps and max_sequence_length must be positive.")


@dataclass
class MeanFlowNFTRLConfig:
    num_epochs: int = 10000
    num_batches_per_epoch: int = 2
    num_inner_epochs: int = 1
    nft_inner_batch_size: int = 0
    nft_start_epoch: int = 0
    num_image_per_prompt: int = 16
    nft_sample_use_ema: bool = False
    sampling_num_steps: int = 4
    sampling_guidance_scale: float = 1.0
    per_prompt_stat_tracking: bool = True
    per_prompt_global_std: bool = True
    weight_advantages: bool = True
    adv_clip_max: float = 5.0
    beta: float = 0.1
    kl_weight: float = 0.0001
    decay_type: int = 1
    nft_decay_interval: int = 1
    reward_fn: dict[str, float] = field(
        default_factory=lambda: {
            "hpsv3_general": 0.5,
            "hpsv3_percentile": 0.5,
            "videoalign_mq": 0.5,
            "videoalign_ta": 1.0,
        }
    )
    reward_model_paths: dict[str, str] = field(default_factory=dict)
    log_sample_images: int = 0
    nft_velocity_mode: str = "meanflow_v"
    central_diff_epsilon: float = 5.0
    cd_velocity_source: str = "noise_minus_x0"
    share_cd_with_old: bool = True
    num_training_timesteps_per_sample: int = 4
    nft_min_timestep: int = 0
    nft_max_timestep: int = 1000
    diffusion_ratio: float = 0.5
    consistency_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.sampling_num_steps < 1:
            raise ValueError("meanflow_nft.sampling_num_steps must be positive.")
        if self.sampling_guidance_scale != 1.0:
            raise ValueError("Wan AnyFlow sampling is CFG-free; use guidance 1.0.")
        if self.num_training_timesteps_per_sample < 1:
            raise ValueError(
                "meanflow_nft.num_training_timesteps_per_sample must be positive."
            )
        if self.nft_velocity_mode != "meanflow_v":
            raise ValueError(
                "The release supports only meanflow_nft.nft_velocity_mode='meanflow_v'."
            )
        if self.cd_velocity_source not in {"noise_minus_x0", "u_self"}:
            raise ValueError(
                "meanflow_nft.cd_velocity_source must be noise_minus_x0 or u_self."
            )
        if self.central_diff_epsilon <= 0:
            raise ValueError("meanflow_nft.central_diff_epsilon must be positive.")
        if self.nft_min_timestep > self.nft_max_timestep:
            raise ValueError(
                "meanflow_nft.nft_min_timestep cannot exceed nft_max_timestep."
            )
        ratio = self.diffusion_ratio + self.consistency_ratio
        if (
            self.diffusion_ratio < 0
            or self.consistency_ratio < 0
            or ratio > 1.0 + 1e-6
        ):
            raise ValueError(
                "meanflow_nft diffusion_ratio and consistency_ratio must be "
                "non-negative and sum to at most 1."
            )
        if self.adv_clip_max <= 0 or self.beta <= 0:
            raise ValueError("meanflow_nft adv_clip_max and beta must be positive.")
        valid_rewards = {
            "hpsv3_general",
            "hpsv3_percentile",
            "videoalign_mq",
            "videoalign_ta",
        }
        unsupported = sorted(set(self.reward_fn) - valid_rewards)
        if unsupported:
            raise ValueError(f"Unsupported Wan reward(s): {unsupported}")


@dataclass
class GeneratorSolverConfig:
    lr: float = 3e-6
    warmup_steps: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0


@dataclass
class SolverConfig:
    generator: GeneratorSolverConfig = field(default_factory=GeneratorSolverConfig)


@dataclass
class TrainConfig:
    total_steps: int = 0
    save_interval: int = 50
    log_interval: int = 1
    seed: int = 42
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    autocast_dtype: str = "bf16"
    resume_from: str = ""
    use_ema: bool = True
    ema_decay: float = 0.9
    ema_every: int = 1


@dataclass
class DistributedConfig:
    strategy: str = "hsdp"
    fsdp_sharding: str = "hybrid"
    ddp_find_unused_parameters: bool | None = None

    def __post_init__(self) -> None:
        self.strategy = self.strategy.lower()
        self.fsdp_sharding = self.fsdp_sharding.lower()
        if self.strategy == "hsdp":
            self.strategy = "fsdp"
            self.fsdp_sharding = "hybrid"
        if self.strategy not in {"fsdp", "ddp"}:
            raise ValueError("distributed.strategy must be fsdp, hsdp, or ddp.")


@dataclass
class VBenchConfig:
    enabled: bool = False
    aug_info_json: str = "dataset/vbench/VBench_aug_full_info.json"
    full_info_json: str = "dataset/vbench/VBench_full_info.json"
    num_samples_per_prompt: int = 5
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    cache_dir: str = "models/vbench"
    output_subdir: str = "vbench"
    eval_every_epochs: int = 0

    def __post_init__(self) -> None:
        if self.num_samples_per_prompt < 1 or self.num_inference_steps < 1:
            raise ValueError("VBench sample and inference counts must be positive.")
        if self.guidance_scale != 1.0:
            raise ValueError("Wan AnyFlow VBench generation is CFG-free.")


@dataclass
class EvalConfig:
    enabled: bool = True
    eval_interval: int = 50
    eval_prompt_source: str = "dataset"
    eval_prompt_path: str = ""
    dataset: str = "wan_dancegrpo"
    dataset_path: str = "dataset"
    reward_fn: dict[str, float] = field(
        default_factory=lambda: {
            "hpsv3_general": 0.5,
            "hpsv3_percentile": 0.5,
            "videoalign_mq": 0.5,
            "videoalign_ta": 1.0,
        }
    )
    reward_dataset_map: dict[str, str] = field(default_factory=dict)
    reward_ckpt_path: str = ""
    eval_num_steps: int = 4
    eval_guidance_scale: float = 1.0
    eval_batch_size: int = 1
    num_media_images: int = 4
    eval_seed: int = 12345
    vbench: VBenchConfig = field(default_factory=VBenchConfig)


@dataclass
class LoggingConfig:
    project: str = "meanflownft-wan"
    run_name: str = "wan2.1-t2v-1.3b-meanflow-nft"
    entity: str = ""
    tags: list[str] = field(
        default_factory=lambda: [
            "wan2.1",
            "video",
            "anyflow",
            "meanflow-nft",
            "480p",
            "81f",
        ]
    )
    enabled: bool = True


@dataclass
class MeanFlowNFTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    meanflow_nft: MeanFlowNFTRLConfig = field(default_factory=MeanFlowNFTRLConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    prompt_path: str = "dataset/wan_dancegrpo"
    output_dir: str = "./outputs/wan_meanflow_nft"

    @classmethod
    def from_yaml(
        cls,
        path: str,
        overrides: dict[str, Any] | None = None,
    ) -> "MeanFlowNFTConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"Configuration must be a YAML mapping: {path}")
        if overrides:
            raw = _apply_flat_overrides(raw, overrides)
        return _dataclass_from_dict(cls, raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
