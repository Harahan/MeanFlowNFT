"""Dataclass configuration for the SD3.5 MeanFlowNFT."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import yaml


def _apply_flat_overrides(data: dict, overrides: dict[str, Any]) -> dict:
    """Apply dot-separated command-line overrides to a nested dictionary."""
    for key, value in overrides.items():
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return data


def _dataclass_from_dict(cls, data: Any):
    """Instantiate nested dataclasses and reject misspelled release fields."""
    if not isinstance(data, dict):
        return data
    field_names = {item.name for item in fields(cls)}
    unknown = sorted(set(data).difference(field_names))
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
    """LoRA configuration for one trainable transformer role."""

    enabled: bool = False
    rank: int = 32
    lora_alpha: int = 8
    target_modules: list = field(
        default_factory=lambda: [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
        ]
    )
    init_lora_weights: str = "gaussian"
    load_path: str = ""
    # Earlier-stage adapters are merged in order before load_path. This
    # preserves the Stage 1 -> Stage 2 -> reward-fine-tuning LoRA chain.
    pre_merge_paths: list = field(default_factory=list)
    merge_before_training: bool = False


@dataclass
class ModelConfig:
    """SD3.5 model and role-specific initialization."""

    pretrained_path: str = ""
    generator_init_path: str = ""
    fake_score_init_path: str = ""
    teacher_init_path: str = ""
    model_type: str = "sd35"
    gradient_checkpointing: bool = True
    dtype: str = "bf16"
    image_resolution: int = 1024
    generator_lora: LoRAConfig = field(default_factory=LoRAConfig)
    fake_score_lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class FlowMapModelConfig:
    """Two-time embedding wrapper used by MeanFlowNFT workflows."""

    enabled: bool = True
    gate_value: float = 0.25
    deltatime_type: str = "r"

    def __post_init__(self) -> None:
        if self.deltatime_type not in {"r", "t-r"}:
            raise ValueError(
                "flowmap_model.deltatime_type must be 'r' or 't-r'; "
                f"got {self.deltatime_type!r}."
            )


@dataclass
class AnyFlowPretrainConfig:
    """Stage 1 forward-training parameters."""

    diffusion_ratio: float = 0.25
    consistency_ratio: float = 0.25
    epsilon: float = 0.5
    cd_velocity_source: str = "noise_minus_x0"
    v_target_source: str = "noise_minus_x0"
    forward_guidance_scale: float = 1.0
    weight_type: str = "gaussian"
    negative_embedding_path: str = ""
    drop_text_ratio: float = 0.1
    ema_warmup_steps: int = 0

    def __post_init__(self) -> None:
        total = self.diffusion_ratio + self.consistency_ratio
        if self.diffusion_ratio < 0 or self.consistency_ratio < 0 or total > 1.0 + 1e-6:
            raise ValueError(
                "anyflow_pretrain diffusion_ratio and consistency_ratio must be "
                f"non-negative and sum to at most 1; got {total}."
            )
        valid_sources = {"noise_minus_x0", "u_self"}
        if self.cd_velocity_source not in valid_sources:
            raise ValueError(
                "anyflow_pretrain.cd_velocity_source must be 'noise_minus_x0' "
                f"or 'u_self'; got {self.cd_velocity_source!r}."
            )
        if self.v_target_source not in valid_sources:
            raise ValueError(
                "anyflow_pretrain.v_target_source must be 'noise_minus_x0' "
                f"or 'u_self'; got {self.v_target_source!r}."
            )
        if not 0.0 <= self.drop_text_ratio <= 1.0:
            raise ValueError(
                "anyflow_pretrain.drop_text_ratio must be in [0, 1]; "
                f"got {self.drop_text_ratio}."
            )


@dataclass
class AnyFlowOnPolicyConfig:
    """Stage 2 on-policy flow-map distribution matching."""

    num_inference_steps_list: list = field(default_factory=lambda: [4, 8, 16, 32])
    dmd_weight: float = 1.0
    dmd_batch_size: int = 0
    dmd_min_timestep: int = 0
    dmd_max_timestep: int = 1000
    real_guidance_scale: float = 0.0
    gradient_normalization: bool = True
    cotrain_forward_kl: bool = True
    discriminator_update_ratio: int = 1
    rollout_detach_between_jumps: bool = False
    rollout_full_steps: bool = False

    def __post_init__(self) -> None:
        if not self.num_inference_steps_list or any(
            int(step) <= 0 for step in self.num_inference_steps_list
        ):
            raise ValueError(
                "anyflow_onpolicy.num_inference_steps_list must contain "
                "positive integers."
            )
        if self.dmd_min_timestep > self.dmd_max_timestep:
            raise ValueError(
                "anyflow_onpolicy.dmd_min_timestep must not exceed "
                "dmd_max_timestep."
            )
        if self.discriminator_update_ratio < 1:
            raise ValueError(
                "anyflow_onpolicy.discriminator_update_ratio must be at least 1."
            )
        if self.rollout_full_steps and not self.rollout_detach_between_jumps:
            raise ValueError(
                "anyflow_onpolicy.rollout_full_steps=True requires "
                "rollout_detach_between_jumps=True."
            )


@dataclass
class MeanFlowNFTRLConfig:
    """MeanFlowNFT reward fine-tuning configuration."""

    # Epoch loop and sampling.
    num_epochs: int = 1000
    num_batches_per_epoch: int = 1
    num_inner_epochs: int = 1
    nft_inner_batch_size: int = 0
    nft_start_epoch: int = 0
    num_image_per_prompt: int = 4
    nft_sample_use_ema: bool = False
    sampling_num_steps: int = 4
    sampling_guidance_scale: float = 1.0
    # The legacy MeanFlowNFT MeanFlowNFT checkpoints were trained/evaluated with
    # encode_prompts_sd35's historical T5 length of 128.
    text_max_sequence_length: int = 128

    # Advantage and policy loss.
    per_prompt_stat_tracking: bool = True
    per_prompt_global_std: bool = True
    weight_advantages: bool = False
    adv_clip_max: float = 5.0
    beta: float = 0.1
    kl_weight: float = 0.0

    # Old-policy update.
    decay_type: int = 1
    nft_decay_interval: int = 1

    # Reward scoring and logging.
    reward_fn: dict = field(default_factory=dict)
    reward_ckpt_path: str = ""
    log_sample_images: int = 0

    # MeanFlowNFT velocity construction.
    nft_velocity_mode: str = "meanflow_v"
    central_diff_epsilon: float = 5.0
    cd_velocity_source: str = "noise_minus_x0"
    share_cd_with_old: bool = True

    # Fresh MeanFlowNFT (t, r) pairs used by every inner update.
    num_training_timesteps_per_sample: int = 4
    nft_min_timestep: int = 0
    nft_max_timestep: int = 1000
    diffusion_ratio: float = 0.25
    consistency_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.sampling_num_steps < 1:
            raise ValueError("meanflow_nft.sampling_num_steps must be at least 1.")
        if self.num_training_timesteps_per_sample < 1:
            raise ValueError(
                "meanflow_nft.num_training_timesteps_per_sample must be at least 1."
            )
        if self.text_max_sequence_length < 1:
            raise ValueError(
                "meanflow_nft.text_max_sequence_length must be at least 1."
            )
        if self.nft_velocity_mode not in {"meanflow_v", "direct_u"}:
            raise ValueError(
                "meanflow_nft.nft_velocity_mode must be 'meanflow_v' or "
                f"'direct_u'; got {self.nft_velocity_mode!r}."
            )
        if self.cd_velocity_source not in {"noise_minus_x0", "u_self"}:
            raise ValueError(
                "meanflow_nft.cd_velocity_source must be 'noise_minus_x0' or "
                f"'u_self'; got {self.cd_velocity_source!r}."
            )
        total = self.diffusion_ratio + self.consistency_ratio
        if self.diffusion_ratio < 0 or self.consistency_ratio < 0 or total > 1.0 + 1e-6:
            raise ValueError(
                "meanflow_nft diffusion_ratio and consistency_ratio must be "
                f"non-negative and sum to at most 1; got {total}."
            )
        if self.nft_min_timestep > self.nft_max_timestep:
            raise ValueError(
                "meanflow_nft.nft_min_timestep must not exceed nft_max_timestep."
            )
        if self.adv_clip_max <= 0:
            raise ValueError("meanflow_nft.adv_clip_max must be positive.")
        if self.beta <= 0:
            raise ValueError("meanflow_nft.beta must be positive.")


@dataclass
class GeneratorSolverConfig:
    lr: float = 1e-6
    warmup_steps: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class FakeSolverConfig:
    lr: float = 1e-6
    warmup_steps: int = 500
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class SolverConfig:
    generator: GeneratorSolverConfig = field(default_factory=GeneratorSolverConfig)
    fake_score: FakeSolverConfig = field(default_factory=FakeSolverConfig)


@dataclass
class TrainConfig:
    total_steps: int = 100000
    save_interval: int = 1000
    log_interval: int = 10
    seed: int = 42
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    autocast_dtype: str = "bf16"
    resume_from: str = ""
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_every: int = 1


@dataclass
class DistributedConfig:
    """FSDP/HSDP/DDP settings."""

    strategy: str = "fsdp"
    fsdp_sharding: str = "full_shard"
    ddp_find_unused_parameters: bool | None = None

    def __post_init__(self) -> None:
        self.strategy = self.strategy.lower()
        self.fsdp_sharding = self.fsdp_sharding.lower()
        if self.strategy == "hsdp":
            self.strategy = "fsdp"
            self.fsdp_sharding = "hybrid"
        if self.strategy not in {"fsdp", "ddp"}:
            raise ValueError(
                "distributed.strategy must be 'fsdp', 'hsdp', or 'ddp'; "
                f"got {self.strategy!r}."
            )


@dataclass
class EvalConfig:
    """Periodic SD3.5 image generation and reward evaluation."""

    enabled: bool = True
    eval_interval: int = 5000
    eval_prompt_source: str = "prompt_file"
    eval_prompt_path: str = "prompts.json"
    dataset: str = "pickscore"
    dataset_path: str = "dataset"
    reward_fn: dict = field(default_factory=lambda: {"pickscore": 1.0})
    reward_dataset_map: dict = field(default_factory=dict)
    reward_ckpt_path: str = ""
    eval_num_steps: int = 4
    eval_guidance_scale: float = 1.0
    eval_batch_size: int = 4
    num_media_images: int = 8
    eval_seed: int = 12345


@dataclass
class LoggingConfig:
    project: str = "MeanFlowNFT"
    run_name: str = ""
    entity: str = ""
    tags: list = field(default_factory=list)
    enabled: bool = True


@dataclass
class MeanFlowNFTConfig:
    """Root configuration for the three supported release workflows."""

    model: ModelConfig = field(default_factory=ModelConfig)
    flowmap_model: FlowMapModelConfig = field(default_factory=FlowMapModelConfig)
    anyflow_pretrain: AnyFlowPretrainConfig = field(default_factory=AnyFlowPretrainConfig)
    anyflow_onpolicy: AnyFlowOnPolicyConfig = field(default_factory=AnyFlowOnPolicyConfig)
    meanflow_nft: MeanFlowNFTRLConfig = field(default_factory=MeanFlowNFTRLConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    prompt_path: str = "prompts.json"
    output_dir: str = "./outputs"

    @classmethod
    def from_yaml(
        cls,
        path: str,
        overrides: dict[str, Any] | None = None,
    ) -> "MeanFlowNFTConfig":
        with open(path, "r") as handle:
            raw = yaml.safe_load(handle) or {}
        if overrides:
            raw = _apply_flat_overrides(raw, overrides)
        return _dataclass_from_dict(cls, raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
