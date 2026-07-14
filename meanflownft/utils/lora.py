"""
LoRA utilities for MeanFlowNFT.

Provides functions to inject LoRA adapters into transformer models and
filter trainable parameters. Reference: ViGen-DiT dit_trainer.py.

Usage:
    from meanflownft.utils.lora import setup_lora
    trainable_names = setup_lora(model, lora_config)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from meanflownft.config import LoRAConfig

logger = logging.getLogger(__name__)


def inject_lora(model: nn.Module, lora_config: LoRAConfig) -> nn.Module:
    """Inject LoRA adapter into a model.

    Tries model.add_adapter() first (diffusers native), falls back to
    peft's inject_adapter_in_model(). Optionally loads pretrained LoRA weights.

    Args:
        model: The target model to inject LoRA into.
        lora_config: LoRA configuration (rank, alpha, target_modules, etc.).

    Returns:
        The model with LoRA adapters injected (modified in-place).
    """
    from peft import LoraConfig as PeftLoraConfig, inject_adapter_in_model

    peft_config = PeftLoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.lora_alpha,
        init_lora_weights=lora_config.init_lora_weights,
        target_modules=lora_config.target_modules,
        use_rslora=False,
    )

    # --- Pre-merge previous LoRA stages into base (optional) ---
    # Use ``pre_merge_paths`` to chain multiple LoRA checkpoints into the
    # base BEFORE the main load_path / merge_before_training step. Useful
    # for multi-stage training (e.g., AnyFlow pretrain LoRA → onpolicy
    # LoRA → NFT LoRA): all upstream stages are folded into the base so
    # the new adapter starts from a model that includes all prior fine-tunes.
    #
    # Behavior: for each path in pre_merge_paths, the SAME peft_config is
    # used for inject + load + merge; after merge the adapter weights are
    # reset (lora_B → 0, A re-init) so the NEXT merge_in step starts from
    # a clean adapter that won't re-add the just-merged delta.
    if lora_config.pre_merge_paths:
        from peft.tuners.tuners_utils import BaseTunerLayer
        if hasattr(model, "add_adapter") and callable(model.add_adapter):
            model.add_adapter(peft_config)
        else:
            inject_adapter_in_model(peft_config, model)
        for idx, pre_path in enumerate(lora_config.pre_merge_paths):
            state_dict = torch.load(pre_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            for module in model.modules():
                if isinstance(module, BaseTunerLayer):
                    module.merge()
            # Reset adapter weights (lora_B → 0) so next merge starts clean,
            # then unmerge() to clear the flag and re-enable the adapter.
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    nn.init.zeros_(param.data)
                elif "lora_A" in name:
                    nn.init.zeros_(param.data)
            for module in model.modules():
                if isinstance(module, BaseTunerLayer) and module.merged:
                    module.unmerge()
            logger.info(
                f"Pre-merge LoRA[{idx}] folded into base: {pre_path}"
            )

    # --- merge_before_training path ---
    # When enabled, loads existing LoRA weights, merges them into the base model
    # weights, then re-initializes a fresh LoRA adapter for new training. This
    # allows training a new LoRA on top of a previously fine-tuned model.
    if lora_config.merge_before_training and lora_config.load_path:
        from peft.tuners.tuners_utils import BaseTunerLayer

        # Step 1: Inject LoRA (same as normal path, ensures key format compatibility).
        # If pre_merge_paths already injected the adapter, skip re-injection.
        adapter_already_injected = any(
            isinstance(m, BaseTunerLayer) for m in model.modules()
        )
        if not adapter_already_injected:
            if hasattr(model, "add_adapter") and callable(model.add_adapter):
                model.add_adapter(peft_config)
            else:
                inject_adapter_in_model(peft_config, model)

        # Step 2: Load pretrained LoRA weights
        state_dict = torch.load(lora_config.load_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained LoRA weights from {lora_config.load_path}")

        # Step 3: Merge LoRA into base weights via peft BaseTunerLayer.merge()
        merged_count = 0
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                module.merge()
                merged_count += 1
        logger.info(f"Merged {merged_count} LoRA layers into base model weights")

        # Step 4: Re-initialize LoRA A/B to fresh random weights in-place.
        # IMPORTANT: Cannot use delete_adapters() — it calls unmerge() first,
        # which reverses the merge from Step 3. Instead, zero out lora_B (makes
        # the adapter an identity transform) so merged base weights are preserved.
        reinit_count = 0
        for name, param in model.named_parameters():
            if "lora_B" in name:
                nn.init.zeros_(param.data)
                reinit_count += 1
            elif "lora_A" in name:
                if lora_config.init_lora_weights == "gaussian":
                    nn.init.normal_(param.data)
                else:
                    nn.init.kaiming_uniform_(param.data, a=5 ** 0.5)
                reinit_count += 1
        logger.info(f"Re-initialized {reinit_count} LoRA params for fresh training")

        # Step 5: Clear the merged flag so LoRA is active in forward passes.
        # After merge(), BaseTunerLayer skips lora_A/lora_B computation in forward.
        # We call unmerge() to re-enable it. Since lora_B is zeros, unmerge()
        # computes base_weight -= scaling * 0 = no change to base weights.
        for module in model.modules():
            if isinstance(module, BaseTunerLayer) and module.merged:
                module.unmerge()
        logger.info("Cleared merged flag — fresh LoRA adapter is now active")

        return model

    # --- Normal path ---
    # If pre_merge_paths already injected the adapter, skip re-injection.
    from peft.tuners.tuners_utils import BaseTunerLayer
    adapter_already_injected = any(
        isinstance(m, BaseTunerLayer) for m in model.modules()
    )
    if not adapter_already_injected:
        if hasattr(model, "add_adapter") and callable(model.add_adapter):
            model.add_adapter(peft_config)
        else:
            logger.info("add_adapter() not found, fallback to inject_adapter_in_model()")
            inject_adapter_in_model(peft_config, model)

    # Optionally load pretrained LoRA weights for initialization.
    # Note: this runs during setup_models(), BEFORE _maybe_resume(). If
    # train.resume_from is set, the checkpoint will overwrite these weights.
    # Use load_path only for initializing a NEW training run from existing
    # LoRA weights; use train.resume_from for resuming interrupted training.
    if lora_config.load_path:
        state_dict = torch.load(lora_config.load_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained LoRA weights from {lora_config.load_path}")
    elif adapter_already_injected:
        # pre_merge_paths zeroed out lora_A too (so subsequent merges start
        # clean); now we need a non-zero lora_A for fresh training.
        # lora_B stays 0 → adapter contribution is 0 at init (standard LoRA).
        for name, param in model.named_parameters():
            if "lora_A" in name:
                if lora_config.init_lora_weights == "gaussian":
                    nn.init.normal_(param.data)
                else:
                    nn.init.kaiming_uniform_(param.data, a=5 ** 0.5)

    return model


def filter_lora_params(model: nn.Module, target_modules: list[str]) -> list[str]:
    """Filter out LoRA A/B parameter names from a model.

    Args:
        model: Model with LoRA adapters injected.
        target_modules: List of target module name patterns (e.g., ["attn.to_q"]).

    Returns:
        List of parameter names that are LoRA A or B weights.
    """
    lora_names = []
    for name, _ in model.named_parameters():
        for module in target_modules:
            if f"{module}.lora_A" in name or f"{module}.lora_B" in name:
                lora_names.append(name)
                break
    return lora_names


def setup_lora(model: nn.Module, lora_config: LoRAConfig) -> list[str]:
    """Inject LoRA into a model and return the trainable parameter names.

    This is the main entry point: injects LoRA adapters, identifies trainable
    parameters, and freezes all non-LoRA parameters.

    Args:
        model: The target model.
        lora_config: LoRA configuration.

    Returns:
        List of trainable (LoRA) parameter names.
    """
    inject_lora(model, lora_config)
    trainable_names = filter_lora_params(model, lora_config.target_modules)

    # Freeze all non-LoRA parameters, enable LoRA parameters
    trainable_set = set(trainable_names)
    for name, param in model.named_parameters():
        param.requires_grad = name in trainable_set

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA injected: {num_trainable:,} trainable / {num_total:,} total params "
        f"({100 * num_trainable / num_total:.2f}%), rank={lora_config.rank}"
    )

    return trainable_names
