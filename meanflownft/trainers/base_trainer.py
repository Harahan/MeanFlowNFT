"""
Base trainer for MeanFlowNFT.

Provides the foundational training infrastructure that all algorithm-specific
trainers inherit from. Handles:
- Distributed setup (FSDP / DDP)
- Training loop skeleton with gradient accumulation
- Checkpoint saving / loading
- EMA model updates
- Wandb logging with structured sections
- Reproducibility (seeding)

Subclasses must implement:
- setup_models(): Load and prepare all models
- setup_optimizers(): Create optimizers and LR schedulers
- train_step(batch) -> dict: One training iteration

Class hierarchy:
    BaseTrainer
      ├── AnyFlowPretrainTrainer
      │     └── AnyFlowOnPolicyTrainer
      └── NFTTrainer (shared MeanFlowNFT infrastructure)
            └── MeanFlowNFTTrainer

Reference: ViGen-DiT dit_trainer.py for distributed training patterns.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from meanflownft.config import MeanFlowNFTConfig
from meanflownft.parallel.utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
)
from meanflownft.utils.checkpoint import save_checkpoint, load_checkpoint
from meanflownft.utils.logging import (
    WandbLogger,
    Timer,
    get_gpu_memory_gb,
    setup_logging,
)

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer with distributed training infrastructure.

    Args:
        config: The root MeanFlowNFT configuration.
    """

    _AUTOCAST_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}

    def __init__(self, config: MeanFlowNFTConfig):
        self.config = config
        self.global_step = 0
        self.models: dict[str, torch.nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, Any] = {}
        self.timer = Timer()
        self.wandb_logger: WandbLogger | None = None
        self._autocast_dtype = self._AUTOCAST_DTYPE.get(config.train.autocast_dtype)
        # Evaluation state (lazy-initialized on first _evaluate() call)
        self._eval_scorer = None
        self._eval_prompts: list[str] | None = None
        # Cache prompts for default and reward-specific dataset overrides.
        # key: "__default__" or "dataset::<name>" -> list[str]
        self._eval_prompts_cache: dict[str, list[str]] = {}
        # Cache optional metadata aligned with cached prompts.
        # key: "__default__" or "dataset::<name>" -> list[dict] | None
        self._eval_metadata_cache: dict[str, list[dict[str, Any]] | None] = {}
        # Per-reward eval scorers for reward-specific dataset evaluation.
        # key: (scorer_attr_name, reward_name) -> MultiScorer({reward_name: weight})
        self._eval_single_reward_scorers: dict[tuple[str, str], Any] = {}
        # Cached media samples for eval logging.
        # key: dataset name -> (images tensor [N,C,H,W] on CPU, prompt list)
        self._eval_media_groups: dict[str, tuple[torch.Tensor, list[str]]] = {}
        # When True, _evaluate() skips offloading scorer to CPU after scoring.
        # Subclasses can set this to reuse the scorer on GPU for a second
        # eval pass before offloading.
        self._defer_scorer_offload = False

    def _autocast(self):
        """Return a torch.autocast context manager using the configured precision."""
        return torch.autocast(
            device_type="cuda",
            dtype=self._autocast_dtype,
            enabled=self._autocast_dtype is not None,
        )

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Main training entry point. Sets up everything and runs the loop."""
        self._setup_environment()
        self.setup_models()
        self.setup_optimizers()
        self._setup_logging()
        self._maybe_resume()
        self._train_loop()
        self._cleanup()

    def _setup_environment(self) -> None:
        """Initialize distributed backend, set seeds, and prepare output dir."""
        setup_distributed()
        setup_logging()
        self._set_seed(self.config.train.seed)

        # Append date suffix to output_dir for run isolation (e.g., outputs/sd35_dmd/20260316)
        if is_main_process():
            date_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.output_dir = os.path.join(self.config.output_dir, date_suffix)
        # Broadcast output_dir to all ranks so they agree on the path
        if dist.is_initialized() and get_world_size() > 1:
            output_dir_list = [self.config.output_dir]
            dist.broadcast_object_list(output_dir_list, src=0)
            self.config.output_dir = output_dir_list[0]

        logger.info(
            f"Environment ready: rank={get_rank()}, world_size={get_world_size()}, "
            f"device=cuda:{torch.cuda.current_device()}, output_dir={self.config.output_dir}"
        )

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility.

        TF32 is always enabled for performance. Full deterministic mode
        (cudnn.benchmark=False, deterministic algorithms) is only activated
        when TORCH_DETERMINISTIC=1 env var is set — useful for resume
        verification but too slow for normal training.
        """
        seed = seed + get_rank()  # Different seed per rank for data diversity
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # TF32 for performance (always on)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Full deterministic mode: set TORCH_DETERMINISTIC=1 to enable
        # (disables cudnn benchmark, forces deterministic algorithms — slower)
        if os.environ.get("TORCH_DETERMINISTIC", "0") == "1":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            logger.info("Full deterministic mode enabled (TORCH_DETERMINISTIC=1)")

    def _setup_logging(self) -> None:
        """Initialize wandb logger on the main process."""
        log_cfg = self.config.logging
        self.wandb_logger = WandbLogger(
            project=log_cfg.project,
            run_name=log_cfg.run_name,
            entity=log_cfg.entity,
            config=self.config.to_dict(),
            tags=log_cfg.tags,
            enabled=log_cfg.enabled,
            output_dir=self.config.output_dir,
        )
        # Save config YAML alongside checkpoints for reproducibility
        if is_main_process():
            os.makedirs(self.config.output_dir, exist_ok=True)
            self.config.save_yaml(
                os.path.join(self.config.output_dir, "config.yaml")
            )

    def _maybe_resume(self) -> None:
        """Resume from checkpoint if configured.

        Restores: model weights, optimizer states, LR schedulers, and global step.
        RNG states are saved to self._resume_rng_state and restored LATER in
        _train_loop(), after the dataloader is created and fast-forwarded.
        This ensures the training RNG is identical whether or not a checkpoint
        was loaded (dataloader creation/shuffle consumes RNG).

        Subclasses should NOT override this method. Instead, override:
        - _restore_extra_checkpoint_state() to restore fields from meta.pt
        - _post_load_checkpoint() for per-rank or custom loading
        """
        self._resume_rng_state = None  # Will be set if resuming
        resume_path = self.config.train.resume_from
        if resume_path:
            extra = load_checkpoint(
                resume_path,
                models=self.models,
                optimizers=self.optimizers,
            )
            self.global_step = extra.get("step", 0)

            # Restore LR scheduler states
            scheduler_states = extra.get("schedulers", {})
            for name, sched in self.schedulers.items():
                if name in scheduler_states:
                    sched.load_state_dict(scheduler_states[name])
                    logger.info(f"  Restored LR scheduler: {name}")

            # Save RNG state for deferred restoration (after dataloader setup)
            rng_state = extra.get("rng_state", None)
            if rng_state:
                self._resume_rng_state = rng_state
                logger.info("  RNG states loaded (will restore after dataloader setup)")

            # Subclass hook: restore algorithm-specific state from meta.pt
            self._restore_extra_checkpoint_state(extra)

            # Subclass hook: per-rank loading (e.g., per-GPU buffers)
            ckpt_dir = resume_path if os.path.isdir(resume_path) else os.path.dirname(resume_path)
            self._post_load_checkpoint(ckpt_dir, extra)

            logger.info(f"Resumed training from step {self.global_step}")

    def _restore_extra_checkpoint_state(self, extra: dict) -> None:
        """Restore extra fields from meta.pt saved by _get_extra_checkpoint_state().

        Override in subclasses to restore algorithm-specific state.

        Args:
            extra: The full meta.pt dict loaded from checkpoint.
        """
        pass

    def _post_load_checkpoint(self, ckpt_dir: str, extra: dict) -> None:
        """Hook called after main checkpoint is loaded.

        Override in subclasses for per-rank loading (e.g., per-GPU buffers).

        Args:
            ckpt_dir: Path to the checkpoint directory.
            extra: The full meta.pt dict loaded from checkpoint.
        """
        pass

    def _restore_rng_state(self) -> None:
        """Restore RNG states saved during resume.

        Called after dataloader creation and fast-forward so that the training
        loop sees exactly the same RNG state as the original run.
        """
        rng_state = self._resume_rng_state
        if rng_state is None:
            return

        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.random.set_rng_state(rng_state["torch_cpu"])
        # Restore per-rank CUDA RNG state
        if "torch_cuda_all_ranks" in rng_state:
            rank = get_rank()
            all_states = rng_state["torch_cuda_all_ranks"]
            if rank < len(all_states):
                torch.cuda.set_rng_state(all_states[rank])
                logger.info(f"  Restored CUDA RNG state for rank {rank}")
            else:
                logger.warning(
                    f"  CUDA RNG state not found for rank {rank} "
                    f"(saved {len(all_states)} ranks), using default"
                )
        elif "torch_cuda" in rng_state:
            # Legacy format: single CUDA state (only correct for rank 0)
            torch.cuda.set_rng_state(rng_state["torch_cuda"])
        logger.info("  Restored RNG states")
        self._resume_rng_state = None  # Clear after restoration

    def _cleanup(self) -> None:
        """Clean up distributed resources and finalize logging."""
        if self.wandb_logger:
            self.wandb_logger.finish()
        cleanup_distributed()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_loop(self) -> None:
        """Main training loop with gradient accumulation and logging."""
        cfg = self.config.train
        dataloader, sampler = self._create_dataloader()

        logger.info(
            f"Starting training: total_steps={cfg.total_steps}, "
            f"batch_size={cfg.batch_size}, grad_accum={cfg.gradient_accumulation_steps}, "
            f"effective_batch_size={cfg.batch_size * cfg.gradient_accumulation_steps * get_world_size()}"
        )

        data_iter = iter(dataloader)
        # Compute which epoch we're in and how far into it, so we can fast-forward
        # the dataloader to the correct position after resuming.
        steps_per_epoch = len(dataloader)
        epoch = self.global_step // steps_per_epoch if steps_per_epoch > 0 else 0
        step_in_epoch = self.global_step % steps_per_epoch if steps_per_epoch > 0 else 0

        if sampler is not None:
            sampler.set_epoch(epoch)
            data_iter = iter(dataloader)

        # Fast-forward dataloader to resume position
        if step_in_epoch > 0:
            logger.info(f"Fast-forwarding dataloader: skipping {step_in_epoch} batches (epoch={epoch})")
            for _ in range(step_in_epoch):
                try:
                    next(data_iter)
                except StopIteration:
                    break

        # Restore RNG states AFTER dataloader creation and fast-forward.
        # This ensures the training RNG matches the original run exactly,
        # since dataloader shuffle and fast-forward consume RNG.
        self._restore_rng_state()

        while self.global_step < cfg.total_steps:
            # Evaluation (at loop start so step 0 / resume point also triggers eval)
            eval_cfg = self.config.eval
            if eval_cfg.enabled and self.global_step % eval_cfg.eval_interval == 0:
                with self.timer.measure("eval"):
                    self._evaluate()

            # Get next batch, handle epoch boundaries
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Training step (subclass implements the actual algorithm)
            with self.timer.measure("train_step"):
                metrics = self.train_step(batch)

            self.global_step += 1

            # Logging
            if self.global_step % cfg.log_interval == 0:
                self._log_step(metrics)

            # Checkpointing
            if self.global_step % cfg.save_interval == 0:
                self._save_checkpoint()

        logger.info(f"Training complete at step {self.global_step}")
        # Final checkpoint (always, regardless of save_interval)
        self._save_checkpoint()
        # Save final generator in diffusers format at output_dir root for easy access
        self._save_final_transformer()

    def _log_step(self, metrics: dict[str, Any]) -> None:
        """Log training metrics via wandb and console.

        Averages scalar metrics across all ranks before logging to ensure
        consistent reporting in distributed training.

        Args:
            metrics: Dict of section -> {metric_name: value} from train_step.
        """
        # Average metrics across ranks for accurate logging
        if dist.is_initialized() and get_world_size() > 1:
            metrics = self._reduce_metrics(metrics)

        # Add profile metrics (timing + GPU memory)
        profile_metrics = {
            "gpu_mem_gb": get_gpu_memory_gb(),
        }
        profile_metrics.update(self.timer.get_metrics())

        all_sections = dict(metrics)
        all_sections["profile"] = profile_metrics

        # Wandb logging
        if self.wandb_logger:
            self.wandb_logger.log_multi_section(all_sections, step=self.global_step)

        # Console logging (main process only, concise summary)
        if is_main_process():
            parts = [f"step={self.global_step}"]
            for section, section_metrics in all_sections.items():
                for k, v in section_metrics.items():
                    if isinstance(v, float):
                        parts.append(f"{section}/{k}={v:.4g}")
                    else:
                        parts.append(f"{section}/{k}={v}")
            logger.info(" | ".join(parts))

    def _save_checkpoint(self) -> None:
        """Save a training checkpoint with full resumable state.

        Captures RNG states BEFORE saving, then restores them AFTER saving.
        This is critical because _save_diffusers_transformer() creates a new
        model instance which consumes CPU RNG, causing RNG state drift between
        a full run and a resumed run.

        Subclasses should NOT override this method. Instead, override:
        - _get_extra_checkpoint_state() to add fields to meta.pt
        - _post_save_checkpoint() for per-rank or custom saving after meta.pt
        """
        # Collect LR scheduler states
        scheduler_states = {
            name: sched.state_dict() for name, sched in self.schedulers.items()
        }
        # Collect RNG states for perfect reproducibility.
        # Each rank saves its own CUDA RNG state (they differ across ranks).
        # We gather all ranks' states and save the full list so each rank
        # can restore its own state on resume.
        local_cuda_rng = torch.cuda.get_rng_state()
        if dist.is_initialized() and get_world_size() > 1:
            # NCCL only supports GPU tensors, so move to CUDA for all_gather
            local_rng_cuda = local_cuda_rng.to(torch.cuda.current_device())
            all_cuda_rng = [torch.empty_like(local_rng_cuda) for _ in range(get_world_size())]
            dist.all_gather(all_cuda_rng, local_rng_cuda)
            all_cuda_rng = [t.cpu() for t in all_cuda_rng]  # move back to CPU for saving
        else:
            all_cuda_rng = [local_cuda_rng]

        # Capture all RNG states BEFORE save (save may consume RNG)
        saved_python_rng = random.getstate()
        saved_numpy_rng = np.random.get_state()
        saved_torch_cpu_rng = torch.random.get_rng_state()

        rng_state = {
            "python": saved_python_rng,
            "numpy": saved_numpy_rng,
            "torch_cpu": saved_torch_cpu_rng,
            "torch_cuda_all_ranks": all_cuda_rng,
        }

        # Determine which models use LoRA (only save LoRA params for them)
        lora_models = self._get_lora_model_names()
        lora_configs = self._get_lora_configs()

        # Build extra state: base fields + subclass extensions
        extra_state = {
            "step": self.global_step,
            "schedulers": scheduler_states,
            "rng_state": rng_state,
        }
        extra_state.update(self._get_extra_checkpoint_state())

        save_checkpoint(
            output_dir=self.config.output_dir,
            step=self.global_step,
            models=self.models,
            optimizers=self.optimizers,
            extra_state=extra_state,
            pretrained_path=self.config.model.pretrained_path,
            lora_models=lora_models,
            lora_configs=lora_configs,
        )

        # Hook for subclass per-rank saving (e.g., NFT buffer)
        self._post_save_checkpoint()

        # Restore RNG states to the captured point so that training continues
        # identically whether or not a checkpoint was saved at this step.
        random.setstate(saved_python_rng)
        np.random.set_state(saved_numpy_rng)
        torch.random.set_rng_state(saved_torch_cpu_rng)
        torch.cuda.set_rng_state(local_cuda_rng)

    def _get_extra_checkpoint_state(self) -> dict:
        """Return extra fields to include in meta.pt.

        Override in subclasses to add algorithm-specific state.
        Called inside the RNG bracket of _save_checkpoint().
        """
        return {}

    def _post_save_checkpoint(self) -> None:
        """Hook called after main checkpoint is saved, before RNG restore.

        Override in subclasses for per-rank saving (e.g., per-GPU buffers).
        The checkpoint directory is at:
            os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        """
        pass

    def _save_final_transformer(self) -> None:
        """Save the final generator in a loadable format at output_dir/transformer.

        For full-weight training:
            Saves in diffusers format, loadable via
            SD3Transformer2DModel.from_pretrained().

        For LoRA training:
            Saves in peft format (adapter_config.json + adapter_model.safetensors),
            loadable via pipe.load_lora_weights().

        Prefers EMA generator if available (more stable for inference),
        falls back to the regular generator otherwise.
        """
        from meanflownft.utils.checkpoint import (
            _gather_fsdp_state_dict,
            _save_diffusers_transformer,
            save_lora_peft_format,
        )

        # Prefer EMA model for the final saved transformer
        generator = self.models.get("generator_ema") or self.models.get("generator")
        if generator is None:
            return

        source_name = "generator_ema" if "generator_ema" in self.models else "generator"
        lora_models = self._get_lora_model_names()
        lora_configs = self._get_lora_configs()

        # Gather full state dict (all ranks participate)
        state_dict = _gather_fsdp_state_dict(generator)

        if is_main_process():
            transformer_dir = os.path.join(self.config.output_dir, "transformer")
            if source_name in lora_models:
                # LoRA mode: save in peft format
                from meanflownft.utils.checkpoint import _resolve_lora_config
                lora_cfg = _resolve_lora_config(source_name, lora_configs)
                if lora_cfg is not None:
                    save_lora_peft_format(state_dict, lora_cfg, transformer_dir)
                    logger.info(
                        f"Final {source_name} LoRA saved in peft format: {transformer_dir}"
                    )
                else:
                    logger.warning(
                        f"No LoRAConfig for {source_name}, skipping final peft save"
                    )
            else:
                # Full weight mode: save in diffusers format
                _save_diffusers_transformer(
                    state_dict=state_dict,
                    config_path=self.config.model.pretrained_path,
                    save_dir=transformer_dir,
                )
                logger.info(f"Final {source_name} saved: {transformer_dir}")

        if dist.is_initialized():
            from meanflownft.parallel.utils import barrier
            barrier()

    def _get_lora_model_names(self) -> set[str]:
        """Return the set of model names that use LoRA.

        Checks model config for generator_lora / fake_score_lora enabled flags.
        Subclasses can override for additional LoRA models.

        Returns:
            Set of model names (e.g., {"generator", "generator_ema", "fake_score_net"}).
        """
        lora_models = set()
        model_cfg = self.config.model
        if hasattr(model_cfg, "generator_lora") and model_cfg.generator_lora.enabled:
            lora_models.add("generator")
            if "generator_ema" in self.models:
                lora_models.add("generator_ema")
        if hasattr(model_cfg, "fake_score_lora") and model_cfg.fake_score_lora.enabled:
            lora_models.add("fake_score_net")
        return lora_models

    def _get_lora_configs(self) -> dict[str, Any]:
        """Return a dict mapping model name -> LoRAConfig for peft format saving.

        Only includes entries for models that actually have LoRA enabled.
        The checkpoint module uses this to write adapter_config.json with the
        correct rank, alpha, target_modules, etc.

        Returns:
            Dict mapping model name -> LoRAConfig instance.
        """
        lora_configs = {}
        model_cfg = self.config.model
        if hasattr(model_cfg, "generator_lora") and model_cfg.generator_lora.enabled:
            lora_configs["generator"] = model_cfg.generator_lora
        if hasattr(model_cfg, "fake_score_lora") and model_cfg.fake_score_lora.enabled:
            lora_configs["fake_score_net"] = model_cfg.fake_score_lora
        return lora_configs

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _load_eval_prompts_and_metadata(
        self,
        dataset_override: str | None = None,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        """Load evaluation prompts and optional metadata from file or dataset directory."""
        eval_cfg = self.config.eval

        use_dataset = dataset_override is not None or eval_cfg.eval_prompt_source == "dataset"
        if use_dataset:
            # Load from dataset directory (e.g., dataset/pickscore/).
            dataset_name = dataset_override if dataset_override is not None else eval_cfg.dataset
            dataset_dir = os.path.join(eval_cfg.dataset_path, dataset_name)
            # Try common prompt file names (eval uses test split)
            for fname in [
                "test.txt",                # TextPromptDataset (pickscore, ocr)
                "test.jsonl",              # GenEval2 synthetic dataset
                "prompts.json",
                "prompts.txt",
                "metadata.jsonl",
            ]:
                fpath = os.path.join(dataset_dir, fname)
                if os.path.exists(fpath):
                    return self._read_prompt_file_with_metadata(fpath)
            raise FileNotFoundError(
                f"No prompt file found in dataset directory: {dataset_dir}"
            )
        else:
            # Load from explicit prompt file path
            return self._read_prompt_file_with_metadata(eval_cfg.eval_prompt_path)

    @staticmethod
    def _read_prompt_file_with_metadata(
        path: str,
    ) -> tuple[list[str], list[dict[str, Any]] | None]:
        """Read prompts and optional metadata from a JSON/JSONL/text file."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], str):
                    return data, None
                elif len(data) > 0 and isinstance(data[0], dict):
                    prompts = [item.get("prompt", item.get("text", "")) for item in data]
                    return prompts, data
            raise ValueError(f"Cannot parse prompts from JSON: {path}")
        elif ext == ".jsonl":
            prompts = []
            metadata: list[dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            prompts.append(item.get("prompt", item.get("text", "")))
                            metadata.append(item)
                        else:
                            prompts.append(str(item))
                            metadata.append({"text": str(item)})
            return prompts, metadata
        else:
            # Plain text: one prompt per line
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()], None

    def _load_eval_prompts(self, dataset_override: str | None = None) -> list[str]:
        """Load evaluation prompts from file or dataset directory."""
        prompts, _ = self._load_eval_prompts_and_metadata(dataset_override=dataset_override)
        return prompts

    @staticmethod
    def _cache_key_for_dataset(dataset_override: str | None = None) -> str:
        return "__default__" if dataset_override is None else f"dataset::{dataset_override}"

    def _get_cached_eval_prompts(self, dataset_override: str | None = None) -> list[str]:
        """Get cached eval prompts for default or reward-specific dataset."""
        cache_key = self._cache_key_for_dataset(dataset_override=dataset_override)
        prompts = self._eval_prompts_cache.get(cache_key)
        if prompts is None:
            prompts, metadata = self._load_eval_prompts_and_metadata(
                dataset_override=dataset_override
            )
            self._eval_prompts_cache[cache_key] = prompts
            self._eval_metadata_cache[cache_key] = metadata
            if dataset_override is None:
                self._eval_prompts = prompts
            source = (
                f"dataset={dataset_override}"
                if dataset_override is not None
                else f"{self.config.eval.eval_prompt_source}"
            )
            logger.info(f"[Eval] Loaded {len(prompts)} prompts ({source})")
        return prompts

    def _get_cached_eval_metadata(
        self,
        dataset_override: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Get cached optional metadata aligned with eval prompts."""
        cache_key = self._cache_key_for_dataset(dataset_override=dataset_override)
        if cache_key not in self._eval_prompts_cache:
            self._get_cached_eval_prompts(dataset_override=dataset_override)
        return self._eval_metadata_cache.get(cache_key)

    def _resolve_eval_dataset_name(self, dataset_override: str | None = None) -> str:
        """Resolve the effective dataset name used for evaluation."""
        if dataset_override is not None:
            return dataset_override
        if self.config.eval.eval_prompt_source == "dataset":
            return str(self.config.eval.dataset)
        return "prompt_file"

    def _maybe_cache_eval_media_group(
        self,
        dataset_name: str,
        images: torch.Tensor | None,
        prompts: list[str],
    ) -> None:
        """Cache per-dataset media samples for later wandb logging."""
        if (
            not is_main_process()
            or self.config.eval.num_media_images <= 0
            or dataset_name in self._eval_media_groups
            or images is None
            or len(images) == 0
        ):
            return

        num_images = min(self.config.eval.num_media_images, len(images))
        self._eval_media_groups[dataset_name] = (
            images[:num_images].detach().cpu(),
            prompts[:num_images],
        )

    @staticmethod
    def _sanitize_media_key_token(name: str) -> str:
        """Make dataset names safe as wandb media key suffixes."""
        token = (name or "").strip()
        if not token:
            return "default"
        token = token.replace("/", "_").replace(" ", "_")
        token = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in token)
        return token or "default"

    def _log_eval_media_groups(
        self,
        *,
        step: int,
        base_key: str,
        fallback_images: torch.Tensor | None = None,
        fallback_prompts: list[str] | None = None,
    ) -> None:
        """Log evaluation media; one panel per dataset when multiple are used."""
        if (
            self.wandb_logger is None
            or self.config.eval.num_media_images <= 0
            or not is_main_process()
        ):
            return

        media_groups = dict(self._eval_media_groups)
        if not media_groups and fallback_images is not None and len(fallback_images) > 0:
            dataset_name = self._resolve_eval_dataset_name(dataset_override=None)
            num_images = min(self.config.eval.num_media_images, len(fallback_images))
            media_groups[dataset_name] = (
                fallback_images[:num_images].detach().cpu(),
                (fallback_prompts or [])[:num_images],
            )

        if not media_groups:
            return

        multi_dataset = len(media_groups) > 1
        for dataset_name, (images, prompts) in media_groups.items():
            if images is None or len(images) == 0:
                continue
            log_pil = [
                Image.fromarray(
                    (img.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                )
                for img in images
            ]
            key = (
                f"{base_key}/{self._sanitize_media_key_token(dataset_name)}"
                if multi_dataset
                else base_key
            )
            self.wandb_logger.log_images(
                images=log_pil,
                captions=prompts[:len(log_pil)],
                step=step,
                key=key,
            )

    def _get_eval_scorer(self, scorer_attr: str = "_eval_scorer", reward_name: str | None = None):
        """Get or initialize an eval scorer.

        - reward_name is None: initialize a shared scorer for full eval.reward_fn.
        - reward_name is set: initialize a single-reward scorer for that reward.
        """
        eval_cfg = self.config.eval
        from meanflownft.rewards.reward_ckpt_path import set_ckpt_path
        from meanflownft.rewards.multi_scorer import MultiScorer

        if eval_cfg.reward_ckpt_path:
            set_ckpt_path(eval_cfg.reward_ckpt_path)

        if reward_name is None:
            scorer = getattr(self, scorer_attr, None)
            if scorer is None:
                try:
                    scorer = MultiScorer(
                        device=torch.device("cpu"),
                        score_dict=eval_cfg.reward_fn,
                        allow_unavailable=True,
                    )
                finally:
                    # Harden against third-party reward imports mutating root logger.
                    setup_logging()
                setattr(self, scorer_attr, scorer)
                logger.info(
                    f"[Eval] Initialized scorer: available={getattr(scorer, 'active_reward_names', list(eval_cfg.reward_fn.keys()))}"
                )
                unavailable = getattr(scorer, "unavailable_rewards", {})
                if unavailable:
                    logger.warning(f"[Eval] Skipped unavailable rewards: {unavailable}")
            return scorer

        cache_key = (scorer_attr, reward_name)
        scorer = self._eval_single_reward_scorers.get(cache_key)
        if scorer is None:
            if reward_name not in eval_cfg.reward_fn:
                raise KeyError(
                    f"Reward {reward_name!r} not found in eval.reward_fn keys={list(eval_cfg.reward_fn.keys())}"
                )
            try:
                scorer = MultiScorer(
                    device=torch.device("cpu"),
                    score_dict={reward_name: eval_cfg.reward_fn[reward_name]},
                    allow_unavailable=True,
                )
            finally:
                # Harden against third-party reward imports mutating root logger.
                setup_logging()
            self._eval_single_reward_scorers[cache_key] = scorer
            active = getattr(scorer, "active_reward_names", [])
            if active:
                logger.info(f"[Eval] Initialized single-reward scorer: {reward_name}")
            else:
                unavailable = getattr(scorer, "unavailable_rewards", {})
                logger.warning(
                    f"[Eval] Skipped unavailable single-reward scorer {reward_name!r}: {unavailable}"
                )
        return scorer

    def _offload_eval_scorers(self, scorer_attr: str = "_eval_scorer") -> None:
        """Move eval scorers to CPU and release cached GPU memory."""
        cpu = torch.device("cpu")
        scorer = getattr(self, scorer_attr, None)
        if scorer is not None:
            scorer.to(cpu)
        for (attr_name, _reward_name), single_scorer in self._eval_single_reward_scorers.items():
            if attr_name == scorer_attr:
                single_scorer.to(cpu)
        torch.cuda.empty_cache()

    @staticmethod
    def _score_details_to_mean_metrics(score_details: dict[str, Any]) -> dict[str, float]:
        """Convert per-sample score arrays/lists into mean scalar metrics."""
        metrics: dict[str, float] = {}
        for metric_name, scores in score_details.items():
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            elif isinstance(scores, list):
                scores = np.array([float(s) for s in scores])
            else:
                scores = np.array(scores)
            valid = scores[scores != -10]
            if len(valid) > 0:
                metrics[metric_name] = float(np.mean(valid))
        return metrics

    @staticmethod
    def _collect_cross_rank_error_messages(
        local_failed: bool,
        local_error: str | None,
    ) -> list[str]:
        """Gather unique error messages from all ranks for clearer eval diagnostics."""
        if not local_failed:
            return []
        world_size = get_world_size()
        payload = local_error.strip() if local_error else None
        if payload and len(payload) > 8000:
            payload = payload[:8000] + "\n...[truncated-before-gather]"
        if dist.is_initialized() and world_size > 1:
            gathered_errors: list[str | None] = [None] * world_size
            dist.all_gather_object(gathered_errors, payload)
            dedup: list[str] = []
            seen: set[str] = set()
            for item in gathered_errors:
                if isinstance(item, str):
                    text = item.strip()
                    if text and text not in seen:
                        dedup.append(text)
                        seen.add(text)
            return dedup
        return [payload] if payload else []

    @staticmethod
    def _truncate_error_message(message: str, max_chars: int = 1600) -> str:
        text = message.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    @staticmethod
    def _aggregate_eval_metrics_across_ranks(
        eval_metrics: dict[str, float],
        num_local: int,
    ) -> dict[str, float]:
        """Cross-rank weighted average aggregation for eval metrics.

        Supports per-rank metric key mismatch safely (e.g., some ranks only have
        invalid values for a reward and thus emit no local metric key).
        """
        world_size = get_world_size()
        if not dist.is_initialized() or world_size <= 1:
            return eval_metrics

        device = torch.device("cuda", torch.cuda.current_device())
        # Ensure every rank iterates the same key set to avoid collective mismatch.
        local_keys = list(eval_metrics.keys())
        gathered_keys: list[list[str] | None] = [None] * world_size
        dist.all_gather_object(gathered_keys, local_keys)
        all_keys = sorted({k for keys in gathered_keys if keys is not None for k in keys})
        aggregated = {}
        for k in all_keys:
            has_local = k in eval_metrics
            local_num = torch.tensor(
                float(eval_metrics[k] * num_local) if has_local else 0.0,
                device=device,
            )
            local_den = torch.tensor(
                float(num_local) if has_local else 0.0,
                device=device,
            )
            dist.all_reduce(local_num, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_den, op=dist.ReduceOp.SUM)
            if local_den.item() > 0:
                aggregated[k] = local_num.item() / local_den.item()
        return aggregated

    def _generate_eval_images_for_prompts(
        self,
        all_eval_prompts: list[str],
        batch_size: int,
        batch_generate_fn: Callable[[list[str]], torch.Tensor],
    ) -> tuple[torch.Tensor | None, list[str]]:
        """Generate eval images for a prompt list with distributed-safe batching.

        Args:
            all_eval_prompts: Full prompt list before sharding.
            batch_size: Per-rank eval batch size.
            batch_generate_fn: Receives one padded batch of prompts and returns
                decoded image tensor [B, C, H, W] in [0,1].

        Returns:
            (all_images_tensor_or_none, all_prompts_local_non_padded)
        """
        rank = get_rank()
        world_size = get_world_size()

        if len(all_eval_prompts) == 0:
            return None, []

        rank_prompts = all_eval_prompts[rank::world_size]
        max_prompts_per_rank = (len(all_eval_prompts) + world_size - 1) // world_size
        max_batches = (max_prompts_per_rank + batch_size - 1) // batch_size

        all_images: list[torch.Tensor] = []
        all_prompts_local: list[str] = []

        for batch_idx in range(max_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(rank_prompts))

            if start_idx < len(rank_prompts):
                batch_prompts = rank_prompts[start_idx:end_idx]
                if len(batch_prompts) < batch_size:
                    batch_prompts = batch_prompts + [batch_prompts[-1]] * (batch_size - len(batch_prompts))
                actual_count = end_idx - start_idx
            else:
                # This rank has no more prompts but must still run forward for
                # symmetric FSDP call counts across ranks.
                batch_prompts = [rank_prompts[0] if rank_prompts else all_eval_prompts[0]] * batch_size
                actual_count = 0

            image_tensors = batch_generate_fn(batch_prompts)
            if actual_count > 0:
                all_images.append(image_tensors[:actual_count])
                all_prompts_local.extend(batch_prompts[:actual_count])

        if not all_images:
            return None, []
        return torch.cat(all_images, dim=0), all_prompts_local

    def _run_eval_for_reward_sets(
        self,
        batch_size: int,
        batch_generate_fn: Callable[[list[str]], torch.Tensor],
        scorer_attr: str = "_eval_scorer",
    ) -> tuple[dict[str, float], torch.Tensor | None, list[str]]:
        """Run eval with optional per-reward dataset overrides.

        If eval.reward_dataset_map is empty, behavior is identical to the
        previous implementation (single prompt set, shared scorer).
        If set, each reward is evaluated on its own dataset (if provided),
        and eval/mean is computed as weighted sum of per-reward means.
        """
        eval_cfg = self.config.eval
        self._eval_media_groups = {}
        reward_dataset_map = getattr(eval_cfg, "reward_dataset_map", {}) or {}
        if not isinstance(reward_dataset_map, dict):
            raise TypeError(
                f"eval.reward_dataset_map must be a dict, got {type(reward_dataset_map)}"
            )
        use_reward_specific_datasets = bool(reward_dataset_map)
        gpu_device = torch.device("cuda", torch.cuda.current_device())

        if not use_reward_specific_datasets:
            all_eval_prompts = self._get_cached_eval_prompts()
            all_eval_metadata = self._get_cached_eval_metadata()
            all_images_tensor, all_prompts_local = self._generate_eval_images_for_prompts(
                all_eval_prompts=all_eval_prompts,
                batch_size=batch_size,
                batch_generate_fn=batch_generate_fn,
            )
            num_local = len(all_images_tensor) if all_images_tensor is not None else 0
            metadata_local = (
                all_eval_metadata[get_rank()::get_world_size()]
                if isinstance(all_eval_metadata, list)
                else None
            )
            if isinstance(metadata_local, list) and num_local > 0 and len(metadata_local) != num_local:
                metadata_local = metadata_local[:num_local]
            scorer = self._get_eval_scorer(scorer_attr=scorer_attr, reward_name=None)
            scorer.to(gpu_device)
            scorer_failed_local = False
            scorer_error_local: str | None = None
            if all_images_tensor is not None and num_local > 0:
                try:
                    score_details, _ = scorer(
                        all_images_tensor,
                        all_prompts_local,
                        metadata=metadata_local,
                        only_strict=False,
                    )
                except Exception as e:
                    scorer_failed_local = True
                    scorer_error_local = (
                        f"{type(e).__name__}: {e}\n"
                        f"{traceback.format_exc(limit=8)}"
                    )
                    score_details = {}
            else:
                score_details = {}
            if dist.is_initialized() and get_world_size() > 1:
                fail_flag = torch.tensor(
                    1 if scorer_failed_local else 0,
                    device=gpu_device,
                    dtype=torch.int32,
                )
                dist.all_reduce(fail_flag, op=dist.ReduceOp.MAX)
                scorer_failed_local = bool(fail_flag.item())
            scorer_error_messages = self._collect_cross_rank_error_messages(
                local_failed=scorer_failed_local,
                local_error=scorer_error_local,
            )
            if scorer_failed_local:
                if is_main_process():
                    logger.warning("[Eval] Shared scorer failed on at least one rank and was skipped.")
                    if scorer_error_messages:
                        logger.warning(
                            "[Eval] Shared scorer failure example:\n"
                            + self._truncate_error_message(scorer_error_messages[0])
                        )
                score_details = {}
            dataset_name = self._resolve_eval_dataset_name(dataset_override=None)
            self._maybe_cache_eval_media_group(
                dataset_name=dataset_name,
                images=all_images_tensor,
                prompts=all_prompts_local,
            )
            local_metrics = self._score_details_to_mean_metrics(score_details)
            eval_metrics = self._aggregate_eval_metrics_across_ranks(local_metrics, num_local)
            for reward_name in eval_cfg.reward_fn.keys():
                if reward_name not in eval_metrics:
                    if is_main_process():
                        logger.warning(
                            f"[Eval] Reward {reward_name!r} has no valid metric and was skipped."
                        )
            return eval_metrics, all_images_tensor, all_prompts_local

        logger.info("[Eval] reward_dataset_map enabled: evaluating rewards grouped by dataset")
        eval_metrics: dict[str, float] = {}
        weighted_mean = 0.0
        has_weighted = False

        # Group rewards by effective dataset to avoid redundant image generation.
        dataset_to_rewards: dict[str, list[str]] = {}
        for reward_name in eval_cfg.reward_fn.keys():
            ds_raw = reward_dataset_map.get(reward_name, None)
            dataset_override = (
                str(ds_raw).strip() if isinstance(ds_raw, str) and str(ds_raw).strip() else None
            )
            dataset_name = self._resolve_eval_dataset_name(dataset_override=dataset_override)
            dataset_to_rewards.setdefault(dataset_name, []).append(reward_name)

        for dataset_name, reward_names in dataset_to_rewards.items():
            # dataset_name is an effective label ("prompt_file" for non-dataset source).
            prompt_dataset_override = None if dataset_name == "prompt_file" else dataset_name
            all_eval_prompts = self._get_cached_eval_prompts(
                dataset_override=prompt_dataset_override
            )
            all_eval_metadata = self._get_cached_eval_metadata(
                dataset_override=prompt_dataset_override
            )
            all_images_tensor, all_prompts_local = self._generate_eval_images_for_prompts(
                all_eval_prompts=all_eval_prompts,
                batch_size=batch_size,
                batch_generate_fn=batch_generate_fn,
            )
            self._maybe_cache_eval_media_group(
                dataset_name=dataset_name,
                images=all_images_tensor,
                prompts=all_prompts_local,
            )

            num_local = len(all_images_tensor) if all_images_tensor is not None else 0
            if is_main_process():
                logger.info(
                    f"[Eval] Dataset {dataset_name!r}: generated {num_local} local samples; "
                    f"running rewards {reward_names}"
                )
            metadata_local = (
                all_eval_metadata[get_rank()::get_world_size()]
                if isinstance(all_eval_metadata, list)
                else None
            )
            if isinstance(metadata_local, list) and num_local > 0 and len(metadata_local) != num_local:
                metadata_local = metadata_local[:num_local]
            cpu_device = torch.device("cpu")
            for reward_name in reward_names:
                weight = eval_cfg.reward_fn[reward_name]
                scorer = self._get_eval_scorer(
                    scorer_attr=scorer_attr,
                    reward_name=reward_name,
                )
                scorer.to(gpu_device)
                reward_failed_local = False
                reward_error_local: str | None = None
                reward_t0 = time.perf_counter()
                if is_main_process():
                    logger.info(
                        f"[Eval] Start reward {reward_name!r} on dataset {dataset_name!r}"
                    )
                if all_images_tensor is not None and num_local > 0:
                    try:
                        score_details, _ = scorer(
                            all_images_tensor,
                            all_prompts_local,
                            metadata=metadata_local,
                            only_strict=False,
                        )
                    except Exception as e:
                        reward_failed_local = True
                        reward_error_local = (
                            f"{type(e).__name__}: {e}\n"
                            f"{traceback.format_exc(limit=8)}"
                        )
                        score_details = {}
                else:
                    score_details = {}
                if dist.is_initialized() and get_world_size() > 1:
                    fail_flag = torch.tensor(
                        1 if reward_failed_local else 0,
                        device=gpu_device,
                        dtype=torch.int32,
                    )
                    dist.all_reduce(fail_flag, op=dist.ReduceOp.MAX)
                    reward_failed_local = bool(fail_flag.item())
                reward_error_messages = self._collect_cross_rank_error_messages(
                    local_failed=reward_failed_local,
                    local_error=reward_error_local,
                )
                if reward_failed_local:
                    if is_main_process():
                        elapsed = time.perf_counter() - reward_t0
                        logger.warning(
                            f"[Eval] Skipped reward {reward_name!r} on dataset {dataset_name!r}: "
                            f"failed on at least one rank (elapsed={elapsed:.2f}s)."
                        )
                        if reward_error_messages:
                            logger.warning(
                                f"[Eval] Failure example for reward {reward_name!r}:\n"
                                + self._truncate_error_message(reward_error_messages[0])
                            )
                    # reward_dataset_map runs scorers sequentially; eagerly offload each
                    # scorer to avoid cumulative GPU memory buildup before later
                    # rewards. Subclasses can defer this for two-pass eval reuse.
                    if not self._defer_scorer_offload:
                        scorer.to(cpu_device)
                        torch.cuda.empty_cache()
                    continue

                if is_main_process():
                    elapsed = time.perf_counter() - reward_t0
                    logger.info(
                        f"[Eval] Finished reward {reward_name!r} on dataset {dataset_name!r} "
                        f"(elapsed={elapsed:.2f}s)"
                    )

                local_metrics = self._score_details_to_mean_metrics(score_details)
                aggregated = self._aggregate_eval_metrics_across_ranks(local_metrics, num_local)

                reward_value = aggregated.get(reward_name)
                if reward_value is not None:
                    weighted_mean += float(weight) * float(reward_value)
                    has_weighted = True
                else:
                    if is_main_process():
                        logger.warning(
                            f"[Eval] Reward {reward_name!r} has no valid metric and was skipped."
                        )

                for key, value in aggregated.items():
                    if key == "mean":
                        continue
                    out_key = key
                    if out_key in eval_metrics and out_key != reward_name:
                        out_key = f"{reward_name}_{key}"
                    eval_metrics[out_key] = value

                # Same eager-offload policy after successful reward evaluation.
                if not self._defer_scorer_offload:
                    scorer.to(cpu_device)
                    torch.cuda.empty_cache()

        if has_weighted:
            eval_metrics["mean"] = weighted_mean

        if self._eval_media_groups:
            first_dataset = next(iter(self._eval_media_groups))
            preview_images, preview_prompts = self._eval_media_groups[first_dataset]
            return eval_metrics, preview_images, preview_prompts
        return eval_metrics, None, []

    def _warmup_eval_reward_resources(self, scorer_attr: str = "_eval_scorer") -> None:
        """Preload eval prompts/scorers before setting eval RNG seed.

        This keeps eval RNG behavior stable across runs by ensuring first-time
        initialization (model loading, file IO) happens outside the seeded block.
        """
        eval_cfg = self.config.eval
        reward_dataset_map = getattr(eval_cfg, "reward_dataset_map", {}) or {}
        if not isinstance(reward_dataset_map, dict):
            raise TypeError(
                f"eval.reward_dataset_map must be a dict, got {type(reward_dataset_map)}"
            )
        use_reward_specific_datasets = bool(reward_dataset_map)

        if not use_reward_specific_datasets:
            self._get_cached_eval_prompts()
            self._get_eval_scorer(scorer_attr=scorer_attr, reward_name=None)
            return

        for reward_name in eval_cfg.reward_fn.keys():
            ds_raw = reward_dataset_map.get(reward_name, None)
            dataset_override = (
                str(ds_raw).strip() if isinstance(ds_raw, str) and str(ds_raw).strip() else None
            )
            self._get_cached_eval_prompts(dataset_override=dataset_override)
            self._get_eval_scorer(scorer_attr=scorer_attr, reward_name=reward_name)

    def _decode_latents_to_pil(self, latents: torch.Tensor) -> list[Image.Image]:
        """Decode latent tensors to PIL images using the VAE.

        Delegates to meanflownft.utils.image for the actual decode logic.

        Args:
            latents: Latent tensors [B, C, H, W] on GPU.

        Returns:
            List of PIL Images.
        """
        from meanflownft.utils.image import decode_latents_to_pil
        return decode_latents_to_pil(self.vae, latents)

    @torch.no_grad()
    def _decode_latents_to_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent tensors to float NCHW tensor [0, 1] using the VAE.

        Returns float tensors without uint8 quantization, matching DiffusionNFT's
        output_type="pt" pipeline output. This ensures reward scores computed on
        the decoded images are directly comparable to DiffusionNFT.

        Args:
            latents: Latent tensors [B, C, H, W] on GPU.

        Returns:
            Float tensor [B, C, H, W] in [0, 1] range on GPU.
        """
        from meanflownft.utils.image import decode_latents_to_tensor
        return decode_latents_to_tensor(self.vae, latents)

    @torch.no_grad()
    def _evaluate(self) -> None:
        """Periodic evaluation: generate images → reward scoring → wandb logging.

        Distributed-aware design:
        - Prompts are sharded across ranks so each rank generates different images
        - All ranks must call _generate_latents the same number of times (FSDP requirement)
        - Scores are all_reduced across ranks for aggregation
        - A barrier at the end ensures all ranks finish eval before resuming training

        Guarantees evaluation consistency via:
        1. Fixed eval_seed per rank (eval_seed + rank for data diversity)
        2. Fixed prompt order (no shuffle, deterministic sharding)
        3. Fixed inference steps (no random_stop)
        4. RNG isolation (save/restore training RNG state)
        """
        eval_cfg = self.config.eval
        rank = get_rank()
        logger.info(f"[Eval] Starting evaluation at step {self.global_step} (rank={rank})")

        # --- 1. Save training RNG state ---
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(),
        }

        try:
            # --- 2. Warmup prompts/scorers BEFORE eval seed ---
            # Keep deterministic eval RNG independent from first-time init.
            self._warmup_eval_reward_resources(scorer_attr="_eval_scorer")

            # --- 3. Set fixed eval seed (per-rank for diverse generations) ---
            eval_seed = eval_cfg.eval_seed + rank
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            torch.cuda.manual_seed_all(eval_seed)

            # --- 4. Prepare generator for eval image generation ---
            generator_model = self.models.get("generator_ema") or self.models.get("generator")
            was_training = generator_model.training
            generator_model.eval()

            def _batch_generate(batch_prompts: list[str]) -> torch.Tensor:
                prompt_embeds, pooled_embeds = self._encode_prompts(batch_prompts)
                uncond_embeds, uncond_pooled = self._get_uncond_embeds(len(batch_prompts))

                # Generate latents (fixed steps, no random_stop, no gradient)
                # Use EMA model if available for better quality
                latents, _ = self._generate_latents(
                    prompt_embeds, pooled_embeds,
                    uncond_embeds, uncond_pooled,
                    num_steps=eval_cfg.eval_num_steps,
                    gradient_truncation=False,
                    random_stop=False,
                    model=generator_model,
                )
                # Decode latents → float tensor [0,1] NCHW (no uint8 round-trip)
                return self._decode_latents_to_tensor(latents)

            # --- 5. Evaluate rewards (shared dataset or per-reward datasets) ---
            eval_metrics, all_images_tensor, all_prompts_local = self._run_eval_for_reward_sets(
                batch_size=eval_cfg.eval_batch_size,
                batch_generate_fn=_batch_generate,
                scorer_attr="_eval_scorer",
            )

            # Restore generator training mode
            if was_training:
                generator_model.train()

            # Offload scorer back to CPU to free GPU memory during training
            # (unless deferred by subclass for a subsequent eval pass)
            if not self._defer_scorer_offload:
                self._offload_eval_scorers(scorer_attr="_eval_scorer")

            # --- 6. Log to wandb (main process only) ---
            if self.wandb_logger:
                self.wandb_logger.log(eval_metrics, step=self.global_step, section="eval")

                self._log_eval_media_groups(
                    step=self.global_step,
                    base_key="media/eval",
                    fallback_images=all_images_tensor,
                    fallback_prompts=all_prompts_local,
                )

            # Console log
            if is_main_process():
                parts = [f"[Eval] step={self.global_step}"]
                for k, v in eval_metrics.items():
                    parts.append(f"{k}={v:.4g}")
                logger.info(" | ".join(parts))

        finally:
            # --- 10. Restore training RNG state ---
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch_cpu"])
            torch.cuda.set_rng_state(rng_state["torch_cuda"])

            # --- 11. Barrier: ensure all ranks finish eval before resuming training ---
            if dist.is_initialized():
                dist.barrier()

    # ------------------------------------------------------------------
    # Metric aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _reduce_metrics(
        metrics: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Average scalar metrics across all ranks via all_reduce.

        Non-float values (e.g., int, str) are left as-is (rank 0 value).

        Args:
            metrics: Nested dict of section -> {metric_name: value}.

        Returns:
            New metrics dict with float values averaged across world_size.
        """
        world_size = get_world_size()
        reduced = {}
        for section, section_metrics in metrics.items():
            reduced_section = {}
            for k, v in section_metrics.items():
                if isinstance(v, float):
                    tensor = torch.tensor(v, device="cuda")
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    reduced_section[k] = tensor.item() / world_size
                else:
                    reduced_section[k] = v
            reduced[section] = reduced_section
        return reduced

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def ema_update(
        src_model: torch.nn.Module,
        tgt_model: torch.nn.Module,
        decay: float,
    ) -> None:
        """Exponential Moving Average update: tgt = decay * tgt + (1 - decay) * src.

        Casts source params to float32 for numerical stability
        (following ViGen-DiT's update_ema).

        Args:
            src_model: Source model (latest parameters).
            tgt_model: Target model (EMA parameters).
            decay: EMA decay rate (e.g., 0.9999). Higher = slower update.
        """
        for src_param, tgt_param in zip(src_model.parameters(), tgt_model.parameters()):
            tgt_param.data.mul_(decay).add_(
                src_param.data.to(torch.float32), alpha=1.0 - decay
            )

    # ------------------------------------------------------------------
    # LR Scheduler helpers
    # ------------------------------------------------------------------

    @staticmethod
    def create_warmup_constant_scheduler(
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Create a learning rate scheduler: linear warmup then constant.

        Args:
            optimizer: The optimizer to schedule.
            warmup_steps: Number of warmup steps.

        Returns:
            LambdaLR scheduler.
        """
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Abstract methods (subclass must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def setup_models(self) -> None:
        """Load, configure, and wrap all models (FSDP/DDP).

        Must populate self.models dict with all named models.
        """
        ...

    @abstractmethod
    def setup_optimizers(self) -> None:
        """Create optimizers and LR schedulers for all trainable models.

        Must populate self.optimizers and self.schedulers dicts.
        """
        ...

    @abstractmethod
    def train_step(self, batch: Any) -> dict[str, dict[str, float]]:
        """Execute one training step.

        Args:
            batch: A batch from the dataloader (e.g., list of prompt strings).

        Returns:
            Dict of section_name -> {metric_name: value} for logging.
            Example: {"dmd": {"loss_dm": 0.5}, "fake_score": {"loss_fake": 0.3}}
        """
        ...

    @abstractmethod
    def _create_dataloader(self):
        """Create the training dataloader and optional sampler.

        Returns:
            Tuple of (DataLoader, DistributedSampler or None).
        """
        ...
