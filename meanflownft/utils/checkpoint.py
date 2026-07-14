"""
Checkpoint utilities for MeanFlowNFT.

Checkpoints contain model adapters, optimizer state, scheduler/RNG metadata,
and an inference-facing transformer adapter directory.

Future extensions:
- Distributed checkpointing (torch.distributed.checkpoint) for large models
- Automatic checkpoint management (keep last N, best K by metric)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TYPE_CHECKING

import torch
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

if TYPE_CHECKING:
    from meanflownft.config import LoRAConfig

from meanflownft.parallel.utils import is_main_process, barrier

logger = logging.getLogger(__name__)


def _gather_fsdp_state_dict(model: torch.nn.Module) -> dict:
    """Gather full state dict from an FSDP-wrapped model.

    For FSDP models, gathers shards to rank 0 with CPU offloading.
    For non-FSDP models, returns state_dict directly.
    """
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            return model.state_dict()
    else:
        return model.state_dict()


def _filter_lora_state_dict(state_dict: dict) -> dict:
    """Filter a state_dict to only keep trainable adapter weights.

    Keeps:
      - LoRA adapter weights (``lora_A`` / ``lora_B`` keys), produced by
        ``peft.add_adapter`` for all LoRA-enabled trainers (DMD / CDM /
        FlowGRPO / AnyFlow).
      - Wan flow-map ``delta_embedder.*`` weights, which are trainable full
        tensors outside the LoRA adapter.

    Args:
        state_dict: Full model state_dict.

    Returns:
        Filtered state_dict containing LoRA + flow-map delta_embedder weights.
    """
    return {
        k: v for k, v in state_dict.items()
        if ("lora_A" in k) or ("lora_B" in k) or ("delta_embedder" in k)
    }


def _is_lora_checkpoint(state_dict: dict) -> bool:
    """Detect if a state_dict is a LoRA / adapter-only checkpoint.

    Such a checkpoint contains only ``lora_A`` / ``lora_B`` keys (plus the
    AnyFlow ``delta_embedder`` weights) and is much smaller than a full
    model checkpoint. We check if every key matches one of these patterns.

    Args:
        state_dict: A loaded state_dict.

    Returns:
        True if this appears to be a LoRA-only / adapter-only checkpoint.
    """
    if not state_dict:
        return False
    return all(
        ("lora_A" in k) or ("lora_B" in k) or ("delta_embedder" in k)
        for k in state_dict.keys()
    )


def save_lora_peft_format(
    state_dict: dict,
    lora_config: LoRAConfig,
    save_dir: str,
) -> None:
    """Save LoRA weights in standard peft format.

    Creates a directory with:
      - adapter_config.json: peft LoRA configuration
      - adapter_model.safetensors: LoRA weights (lora_A/lora_B only)

    This format is compatible with:
      - pipe.load_lora_weights(save_dir)
      - PeftModel.from_pretrained(model, save_dir)

    Args:
        state_dict: Full model state_dict (will be filtered to LoRA keys).
        lora_config: LoRA configuration used during training.
        save_dir: Target directory for the peft-format output.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Filter to LoRA-only keys
    lora_state = _filter_lora_state_dict(state_dict)

    # Save weights in safetensors format
    safetensors_save_file(lora_state, os.path.join(save_dir, "adapter_model.safetensors"))

    # Write adapter_config.json (peft-compatible)
    adapter_config = {
        "r": lora_config.rank,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules),
        "lora_dropout": 0.0,
        "bias": "none",
        "peft_type": "LORA",
        "task_type": None,
        "init_lora_weights": lora_config.init_lora_weights,
        "use_rslora": False,
    }
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    logger.info(f"  LoRA saved in peft format: {save_dir} ({len(lora_state)} keys)")


def load_lora_peft_format(
    model: torch.nn.Module,
    lora_dir: str,
    map_location: str = "cpu",
) -> None:
    """Load LoRA weights from peft format directory.

    Loads adapter_model.safetensors and applies to the model with strict=False.

    Args:
        model: Model with LoRA adapters already injected.
        lora_dir: Path to directory containing adapter_model.safetensors.
        map_location: Device to map tensors to.
    """
    safetensors_path = os.path.join(lora_dir, "adapter_model.safetensors")
    if os.path.exists(safetensors_path):
        state_dict = safetensors_load_file(safetensors_path, device=map_location)
    else:
        # Fallback to legacy .pt format
        pt_path = os.path.join(lora_dir, "adapter_model.pt")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=map_location, weights_only=False)
        else:
            raise FileNotFoundError(
                f"No adapter_model.safetensors or adapter_model.pt found in {lora_dir}"
            )
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    logger.info(f"  Loaded LoRA weights from peft format: {lora_dir} ({len(state_dict)} keys)")


def is_peft_lora_dir(path: str) -> bool:
    """Check if a path is a peft-format LoRA directory.

    A peft LoRA directory contains adapter_config.json and adapter_model.safetensors.
    """
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "adapter_model.safetensors"))
        or os.path.exists(os.path.join(path, "adapter_config.json"))
    )


def resolve_meanflow_nft_adapter(path: str) -> str:
    """Resolve a Wan MeanFlowNFT adapter file or directory."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"MeanFlowNFT checkpoint not found: {path}")
    candidates = (
        os.path.join(path, "generator_ema.pt"),
        os.path.join(path, "transformer", "adapter_model.safetensors"),
        os.path.join(path, "adapter_model.safetensors"),
        os.path.join(path, "generator.pt"),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No generator_ema.pt, generator.pt, or adapter_model.safetensors "
        f"under {path}"
    )


def load_wan_meanflow_nft_adapter(
    model: torch.nn.Module,
    path: str,
    lora_config: "LoRAConfig",
) -> dict[str, int | str]:
    """Load the composite Wan adapter (LoRA plus full delta embedder)."""
    from dataclasses import replace

    from meanflownft.utils.lora import setup_lora

    resolved = resolve_meanflow_nft_adapter(path)
    clean_config = replace(
        lora_config,
        load_path="",
        pre_merge_paths=[],
        merge_before_training=False,
    )
    setup_lora(model, clean_config)
    if resolved.endswith(".safetensors"):
        state_dict = safetensors_load_file(resolved, device="cpu")
    else:
        state_dict = torch.load(
            resolved,
            map_location="cpu",
            weights_only=False,
        )
    if not isinstance(state_dict, dict):
        raise TypeError(f"Adapter must contain a state dict: {resolved}")
    lora_keys = [
        key
        for key in state_dict
        if "lora_A" in key or "lora_B" in key
    ]
    delta_keys = [key for key in state_dict if "delta_embedder" in key]
    if not lora_keys:
        raise ValueError(f"No LoRA tensors found in {resolved}")
    if not delta_keys:
        raise ValueError(
            "Wan MeanFlowNFT checkpoints must include delta_embedder tensors."
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "Unexpected MeanFlowNFT checkpoint keys: "
            f"{incompatible.unexpected_keys[:20]}"
        )
    model.requires_grad_(False)
    model.eval()
    logger.info(
        "Loaded Wan MeanFlowNFT adapter: %s (%d LoRA, %d delta tensors)",
        resolved,
        len(lora_keys),
        len(delta_keys),
    )
    return {
        "path": resolved,
        "lora_tensors": len(lora_keys),
        "delta_tensors": len(delta_keys),
    }


def _save_diffusers_transformer(
    state_dict: dict,
    config_path: str,
    save_dir: str,
) -> None:
    """Save a transformer state_dict in diffusers format.

    Creates a fresh model instance from config (to avoid FSDP sharded parameter
    issues), loads the gathered state_dict, and calls save_pretrained().

    Args:
        state_dict: Full (unsharded) model state_dict.
        config_path: Path to the pretrained model dir containing transformer/config.json.
        save_dir: Target directory for the diffusers-format output.
    """
    import diffusers
    import json
    from meanflownft.utils.fast_init import fast_init

    # Load config from the original pretrained model
    config_json = os.path.join(config_path, "transformer", "config.json")
    if not os.path.exists(config_json):
        logger.warning(
            f"Cannot save diffusers format: config not found at {config_json}. "
            "Skipping save_pretrained."
        )
        return

    # Create a fresh (unsharded) model from config, load weights, save
    # Use fast_init to avoid RNG consumption (this runs inside the RNG bracket)
    with open(config_json, "r") as f:
        config = json.load(f)
    class_name = config.get(
        "_class_name",
        "WanAnyFlowTransformer3DModel",
    )
    model_cls = getattr(diffusers, class_name, None)
    if model_cls is None and class_name in {
        "AnyFlowTransformer3DModel",
        "FAR_Wan_Transformer3DModel",
        "WanAnyFlowTransformer3DModel",
    }:
        from meanflownft.models.wan_transformer import (
            WanAnyFlowTransformer3DModel,
        )

        model_cls = WanAnyFlowTransformer3DModel
    if model_cls is None:
        logger.warning(
            f"Cannot save diffusers format: unsupported transformer class '{class_name}' "
            f"from {config_json}. Skipping save_pretrained."
        )
        return

    with fast_init(torch.device("cpu")):
        model = model_cls(**{k: v for k, v in config.items() if not k.startswith("_")})
    model.load_state_dict(state_dict)
    model.save_pretrained(save_dir)


def _resolve_lora_config(
    model_name: str,
    lora_configs: dict[str, "LoRAConfig"],
) -> "LoRAConfig | None":
    """Resolve the LoRAConfig for a given model name.

    Handles aliasing: generator_ema uses the same LoRA config as generator.

    Args:
        model_name: Name of the model (e.g., "generator", "generator_ema",
            "fake_score_net").
        lora_configs: Dict mapping model name -> LoRAConfig.

    Returns:
        The matching LoRAConfig, or None if not found.
    """
    if model_name in lora_configs:
        return lora_configs[model_name]
    # generator_ema shares config with generator
    if model_name == "generator_ema" and "generator" in lora_configs:
        return lora_configs["generator"]
    return None


def save_checkpoint(
    output_dir: str,
    step: int,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    extra_state: dict[str, Any] | None = None,
    pretrained_path: str = "",
    lora_models: set[str] | None = None,
    lora_configs: dict[str, "LoRAConfig"] | None = None,
) -> None:
    """Save a training checkpoint in split format.

    Saves each component as a separate file to avoid a single huge file.

    For full-weight models:
    - generator is saved in diffusers format (transformer/ subfolder)
    - all models saved as .pt state_dicts

    For LoRA models:
    - Resume files: {name}.pt containing only LoRA keys (fast, for resume)
    - Inference files: transformer/ subfolder in peft format
      (adapter_config.json + adapter_model.safetensors), just like full-weight
      saves a transformer/ folder. Prefers EMA generator if available.

    Optimizer states are always saved as .pt files regardless of LoRA.

    All ranks participate in FSDP state dict gathering, but only rank 0 writes.

    Args:
        output_dir: Base output directory.
        step: Current training step.
        models: Dict of named models to save (state_dicts).
        optimizers: Dict of named optimizers to save.
        extra_state: Additional state (step, schedulers, rng_state, etc.).
        pretrained_path: Path to the original pretrained model dir (for reading
            transformer/config.json). If provided, saves generator in diffusers
            format under transformer/ subfolder.
        lora_models: Set of model names that use LoRA. Their .pt files will
            contain only LoRA keys (for resume).
        lora_configs: Dict mapping model name -> LoRAConfig for peft format
            metadata. Required when lora_models is non-empty.
    """
    if lora_models is None:
        lora_models = set()
    if lora_configs is None:
        lora_configs = {}

    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")

    # Gather model state dicts (all ranks participate for FSDP)
    model_states = {}
    for name, model in models.items():
        model_states[name] = _gather_fsdp_state_dict(model)

    # Gather optimizer state dicts (all ranks participate for FSDP)
    # Under FSDP, optimizer.state_dict() returns rank-local sharded state.
    # We must use FSDP.full_optim_state_dict() to gather the full state to rank 0.
    optimizer_states = {}
    for name, optimizer in optimizers.items():
        # Find the corresponding FSDP model for this optimizer.
        # Convention: optimizer name matches model name, or model name + "_net" suffix.
        fsdp_model = None
        for model_name, model in models.items():
            if isinstance(model, FSDP) and (
                model_name == name or model_name.startswith(name)
            ):
                fsdp_model = model
                break
        if fsdp_model is not None:
            optimizer_states[name] = FSDP.full_optim_state_dict(fsdp_model, optimizer)
        else:
            optimizer_states[name] = optimizer.state_dict()

    # Only rank 0 writes to disk
    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)

        # --- Determine which generator state to use for the transformer/ folder ---
        # Prefer EMA generator if available (more stable for inference)
        gen_source_name = None
        if "generator_ema" in model_states:
            gen_source_name = "generator_ema"
        elif "generator" in model_states:
            gen_source_name = "generator"

        has_lora_generator = any(
            name in lora_models
            for name in ["generator", "generator_ema"]
            if name in model_states
        )

        # 1. Save transformer/ folder for easy inference loading
        if gen_source_name:
            transformer_dir = os.path.join(ckpt_dir, "transformer")
            if has_lora_generator:
                # LoRA mode: save peft format in transformer/ folder
                lora_cfg = _resolve_lora_config(gen_source_name, lora_configs)
                if lora_cfg is not None:
                    save_lora_peft_format(
                        model_states[gen_source_name], lora_cfg, transformer_dir,
                    )
                    logger.info(
                        f"  {gen_source_name} LoRA saved in peft format: {transformer_dir}"
                    )
                else:
                    logger.warning(
                        f"  No LoRAConfig for {gen_source_name}, "
                        f"skipping transformer/ peft save"
                    )
            elif pretrained_path:
                # Full-weight mode: save in diffusers format
                _save_diffusers_transformer(
                    state_dict=model_states[gen_source_name],
                    config_path=pretrained_path,
                    save_dir=transformer_dir,
                )
                logger.info(
                    f"  {gen_source_name} saved in diffusers format: {transformer_dir}"
                )

        # 2. Save each model state dict as .pt (for resume)
        #    For LoRA models: filter to LoRA keys only
        #    For full-weight models: save full state_dict
        for name, state_dict in model_states.items():
            if name in lora_models:
                lora_state = _filter_lora_state_dict(state_dict)
                path = os.path.join(ckpt_dir, f"{name}.pt")
                torch.save(lora_state, path)
                logger.info(
                    f"  Model saved (LoRA resume): {path} ({len(lora_state)} keys)"
                )
            else:
                path = os.path.join(ckpt_dir, f"{name}.pt")
                torch.save(state_dict, path)
                logger.info(f"  Model saved: {path}")

        # 3. Save each optimizer as separate files
        for name, state_dict in optimizer_states.items():
            path = os.path.join(ckpt_dir, f"optimizer_{name}.pt")
            torch.save(state_dict, path)
            logger.info(f"  Optimizer saved: {path}")

        # 4. Save meta state (step, schedulers, rng)
        meta = {"step": step}
        if extra_state:
            meta.update(extra_state)
        meta_path = os.path.join(ckpt_dir, "meta.pt")
        torch.save(meta, meta_path)
        logger.info(f"  Meta saved: {meta_path}")

        logger.info(f"Checkpoint saved: {ckpt_dir} (step={step})")

    barrier()


def load_checkpoint(
    ckpt_path: str,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint.

    Supports both the new split format and legacy single-file format.

    Args:
        ckpt_path: Path to the checkpoint directory or legacy file.
        models: Dict of named models to load state into.
        optimizers: Optional dict of named optimizers to load state into.
        map_location: Device to map tensors to during loading.

    Returns:
        Extra state dict (contains 'step', 'schedulers', 'rng_state', etc.).
    """
    # Determine format: split (directory with meta.pt) or legacy (single .pt)
    if os.path.isdir(ckpt_path):
        meta_path = os.path.join(ckpt_path, "meta.pt")
        if os.path.exists(meta_path):
            return _load_split_checkpoint(ckpt_path, models, optimizers, map_location)
        # Legacy: directory containing checkpoint.pt
        legacy_path = os.path.join(ckpt_path, "checkpoint.pt")
        if os.path.exists(legacy_path):
            return _load_legacy_checkpoint(legacy_path, models, optimizers, map_location)
        raise FileNotFoundError(
            f"Checkpoint directory {ckpt_path} contains neither meta.pt nor checkpoint.pt"
        )
    elif os.path.isfile(ckpt_path):
        return _load_legacy_checkpoint(ckpt_path, models, optimizers, map_location)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


def _load_split_checkpoint(
    ckpt_dir: str,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None,
    map_location: str,
) -> dict[str, Any]:
    """Load from the new split checkpoint format."""
    logger.info(f"Loading split checkpoint from: {ckpt_dir}")

    # Load model states
    for name, model in models.items():
        path = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=map_location, weights_only=False)
            # LoRA checkpoints contain only lora_A/lora_B keys — use strict=False
            is_lora = _is_lora_checkpoint(state_dict)
            if isinstance(model, FSDP):
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    model.load_state_dict(state_dict, strict=not is_lora)
            else:
                model.load_state_dict(state_dict, strict=not is_lora)
            suffix = " (LoRA partial)" if is_lora else ""
            logger.info(f"  Loaded model: {name}{suffix}")
        else:
            logger.warning(f"  Model file not found: {path}, skipping")

    # Load optimizer states
    # Under FSDP, the saved state is the full (gathered) optimizer state.
    # We must use FSDP.optim_state_dict_to_load() to scatter it back to shards.
    if optimizers:
        for name, optimizer in optimizers.items():
            path = os.path.join(ckpt_dir, f"optimizer_{name}.pt")
            if os.path.exists(path):
                full_osd = torch.load(path, map_location=map_location, weights_only=False)
                # Find the corresponding FSDP model
                fsdp_model = None
                for model_name, model in models.items():
                    if isinstance(model, FSDP) and (
                        model_name == name or model_name.startswith(name)
                    ):
                        fsdp_model = model
                        break
                if fsdp_model is not None:
                    # Scatter the full optimizer state back to FSDP shards
                    sharded_osd = FSDP.optim_state_dict_to_load(
                        fsdp_model, optimizer, full_osd,
                    )
                    optimizer.load_state_dict(sharded_osd)
                else:
                    optimizer.load_state_dict(full_osd)
                logger.info(f"  Loaded optimizer: {name}")
            else:
                logger.warning(f"  Optimizer file not found: {path}, skipping")

    # Load meta state
    meta_path = os.path.join(ckpt_dir, "meta.pt")
    meta = torch.load(meta_path, map_location=map_location, weights_only=False)
    logger.info(f"  Resumed from step {meta.get('step', 'unknown')}")
    return meta


def _load_legacy_checkpoint(
    ckpt_path: str,
    models: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None,
    map_location: str,
) -> dict[str, Any]:
    """Load from the legacy single-file checkpoint format (backward compat)."""
    logger.info(f"Loading legacy checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # Load model states
    for name, model in models.items():
        key = f"model_{name}"
        if key in state:
            if isinstance(model, FSDP):
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    model.load_state_dict(state[key])
            else:
                model.load_state_dict(state[key])
            logger.info(f"  Loaded model: {name}")
        else:
            logger.warning(f"  Model '{name}' not found in checkpoint, skipping")

    # Load optimizer states
    if optimizers:
        for name, optimizer in optimizers.items():
            key = f"optimizer_{name}"
            if key in state:
                optimizer.load_state_dict(state[key])
                logger.info(f"  Loaded optimizer: {name}")
            else:
                logger.warning(f"  Optimizer '{name}' not found in checkpoint, skipping")

    # Return extra state
    extra = {k: v for k, v in state.items()
             if not k.startswith("model_") and not k.startswith("optimizer_")}
    logger.info(f"  Resumed from step {extra.get('step', 'unknown')}")
    return extra
