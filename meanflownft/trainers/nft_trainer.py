"""Shared DiffusionNFT engine for MeanFlowNFT reward fine-tuning.

The public :class:`MeanFlowNFTTrainer` supplies flow-map rollout and velocity
construction while this class owns sampling, rewards, advantages, optimization,
EMA, and checkpoint state. The policy objective operates in instantaneous
velocity space:

    x_t = (1 - t/T) * x_0 + (t/T) * eps
    v_t = eps - x_0
    v_theta_plus  = (1 - beta) * v_old + beta * v_theta
    v_theta_minus = (1 + beta) * v_old - beta * v_theta
    L = r_hat * ||v_theta_plus - v_t||^2 + (1 - r_hat) * ||v_theta_minus - v_t||^2
    r_hat = 0.5 + 0.5 * clip(advantage / adv_clip_max, -1, 1)
    (+ kl_weight * ||v_theta - v_ref||^2 if kl_weight > 0)

Per train_step:
    1. Sample M_local unique prompts (from the standard distributed prompt
       dataloader), K-repeat them locally to (M_local * K) samples per rank.
    2. Sampling phase (no_grad): MeanFlowNFT rollout from `old_model` (or
       `generator_ema` if configured).
    3. VAE decode -> images -> MultiScorer rewards.
    4. all_gather (prompts, rewards) -> PerPromptStatTracker -> advantages.
    5. Inner-loop update (inner_epochs x inner_batches x num_timesteps):
       sample t, build (x_t, v_t), compute v_theta (with grad), v_old / v_ref
       (no grad), NFT MSE loss (+ optional KL), backward, optimizer.step().
    6. EMA decay of old_model toward generator (fixed or linear ramp).

This is internal infrastructure and is not registered as a standalone trainer.
"""

from __future__ import annotations

import copy
import logging
import os
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler

from torch.utils.data import DataLoader

from meanflownft.config import MeanFlowNFTConfig
from meanflownft.data.prompt_dataset import PromptDataset
from meanflownft.models.sd35 import (
    encode_prompts_sd35,
    load_sd35_models,
    predict_noise_sd35,
)
from meanflownft.parallel.utils import (
    fsdp_wrap_model,
    get_rank,
    get_transformer_wrap_policy,
    get_world_size,
    is_main_process,
)
from meanflownft.rewards.multi_scorer import MultiScorer
from meanflownft.rewards.reward_ckpt_path import set_ckpt_path
from meanflownft.trainers.base_trainer import BaseTrainer
from meanflownft.trainers.nft_utils import DistributedKRepeatSampler, PerPromptStatTracker
from meanflownft.utils.lora import setup_lora

logger = logging.getLogger(__name__)


class NFTTrainer(BaseTrainer):
    """Internal DiffusionNFT engine used by :class:`MeanFlowNFTTrainer`."""

    def __init__(self, config: MeanFlowNFTConfig):
        super().__init__(config)
        self.model_type = str(self.config.model.model_type).lower()
        if self.model_type != "sd35":
            raise ValueError(
                "MeanFlowNFT release only supports model_type='sd35'; "
                f"got {self.model_type!r}."
            )

        # Models (populated in setup_models)
        self.generator: Optional[nn.Module] = None
        self.generator_ema: Optional[nn.Module] = None
        self.old_model: Optional[nn.Module] = None
        self.ref_model: Optional[nn.Module] = None  # only built if kl_weight > 0
        self.text_encoders: list[nn.Module] = []
        self.tokenizers: list = []
        self.scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None
        self.vae = None
        self.latent_channels: Optional[int] = None
        self.latent_size: Optional[int] = None

        # Reward scorer (lazy-initialized on first sampling phase).
        self._reward_scorer: Optional[MultiScorer] = None
        self._stat_tracker: Optional[PerPromptStatTracker] = None
        self._uncond_embeds: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        # Epoch-based outer loop state.
        self._nft_epoch: int = 0
        self._nft_dataloader: Optional[DataLoader] = None
        self._nft_sampler: Optional[DistributedKRepeatSampler] = None
        self._nft_sample_iter = None

    # ==================================================================
    # SD3.5 model setup
    # ==================================================================

    def setup_models(self) -> None:
        model_cfg = self.config.model
        dist_cfg = self.config.distributed

        logger.info("=" * 60)
        logger.info("Setting up NFT models")
        logger.info("=" * 60)

        components = load_sd35_models(model_cfg)
        base_transformer = components["transformer"]

        self.text_encoders = components["text_encoders"]
        self.tokenizers = components["tokenizers"]
        self.scheduler = components["scheduler"]
        self.vae = components["vae"]

        if model_cfg.gradient_checkpointing:
            base_transformer.enable_gradient_checkpointing()

        self.generator = copy.deepcopy(base_transformer)
        del base_transformer

        # Optional role-specific base-weight initialization.
        self._load_model_init_from_path(
            self.generator, model_cfg.generator_init_path, "generator",
        )

        # Hook for subclass extensions (flow-map wrapper, etc.).
        self._pre_setup_lora()

        if model_cfg.generator_lora.enabled:
            setup_lora(self.generator, model_cfg.generator_lora)
            logger.info(
                f"  Generator: LoRA injected (rank={model_cfg.generator_lora.rank})"
            )
        else:
            self.generator.train()
            logger.info("  Generator: full-weight trainable")

        # Subclass hook (e.g., re-unfreeze flow-map delta_embedder after LoRA freeze).
        self._post_setup_lora()

        # Freeze text encoders + VAE; move to GPU.
        for enc in self.text_encoders:
            enc.requires_grad_(False)
            enc.eval()
        self.vae.requires_grad_(False)
        self.vae.eval()
        device = torch.device("cuda")
        self.vae.to(device)
        for enc in self.text_encoders:
            enc.to(device)

        # SD3.5 latent geometry.
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_channels = self.generator.config.in_channels
        latent_divisor = vae_scale_factor
        if model_cfg.image_resolution % latent_divisor != 0:
            raise ValueError(
                f"image_resolution={model_cfg.image_resolution} must be divisible by "
                f"the SD3.5 VAE scale factor {latent_divisor}."
            )
        self.latent_size = model_cfg.image_resolution // latent_divisor

        # EMA generator (optional). MUST be deepcopy'd BEFORE FSDP wrap.
        if self.config.train.use_ema:
            self.generator_ema = copy.deepcopy(self.generator)
            self.generator_ema.requires_grad_(False)
            self.generator_ema.eval()

        # old_model: frozen deepcopy of generator at training start, decays
        # slowly toward `generator` via EMA. Required for NFT loss.
        self.old_model = copy.deepcopy(self.generator)
        self.old_model.requires_grad_(False)
        self.old_model.eval()

        # ref_model: only built when KL regularization is enabled. Frozen
        # for the entire training run (snapshot of the initial generator).
        if self._nft_cfg().kl_weight > 0.0:
            self.ref_model = copy.deepcopy(self.generator)
            self.ref_model.requires_grad_(False)
            self.ref_model.eval()

        self._wrap_models(dist_cfg)

        # Register models for checkpointing. The base trainer saves all
        # models in self.models with their respective LoRA-or-full path.
        self.models = {"generator": self.generator}
        if self.generator_ema is not None:
            self.models["generator_ema"] = self.generator_ema
        if self.old_model is not None:
            self.models["old_model"] = self.old_model
        if self.ref_model is not None:
            self.models["ref_model"] = self.ref_model

        # Per-prompt stat tracker (lazy-initialized).
        if self._nft_cfg().per_prompt_stat_tracking:
            self._stat_tracker = PerPromptStatTracker(
                global_std=self._nft_cfg().per_prompt_global_std,
            )

        logger.info("NFT model setup complete")
        logger.info("=" * 60)

    def _pre_setup_lora(self) -> None:
        """Hook called BEFORE LoRA injection. Override for flow-map setup."""
        pass

    def _post_setup_lora(self) -> None:
        """Hook called AFTER LoRA setup (e.g., re-unfreeze delta_embedder)."""
        pass

    def _wrap_models(self, dist_cfg) -> None:
        """Wrap generator + EMA + old_model (+ ref_model) with FSDP/DDP."""
        strategy = dist_cfg.strategy
        fsdp_precision = self.config.model.dtype

        if strategy == "fsdp":
            try:
                from diffusers.models.transformers.transformer_sd3 import (
                    JointTransformerBlock,
                )
                wrap_policy = get_transformer_wrap_policy(JointTransformerBlock)
            except ImportError:
                wrap_policy = None
            for name in ("generator", "generator_ema", "old_model", "ref_model"):
                mod = getattr(self, name)
                if mod is not None:
                    setattr(
                        self,
                        name,
                        fsdp_wrap_model(
                            mod,
                            sharding_strategy=dist_cfg.fsdp_sharding,
                            fsdp_precision=fsdp_precision,
                            auto_wrap_policy=wrap_policy,
                        ),
                    )
        elif strategy == "ddp":
            from meanflownft.parallel.utils import ddp_wrap_model
            device = torch.device("cuda")
            has_lora = self.config.model.generator_lora.enabled
            ddp_find_unused_cfg = self.config.distributed.ddp_find_unused_parameters
            if ddp_find_unused_cfg is None:
                # Default heuristic (historical behavior): enable for LoRA.
                find_unused = has_lora
            else:
                find_unused = bool(ddp_find_unused_cfg)
            self.generator = ddp_wrap_model(
                self.generator.to(device), find_unused_parameters=find_unused,
            )
            if get_rank() == 0:
                logger.info(
                    "DDP config: find_unused_parameters=%s (lora_enabled=%s, user_override=%s)",
                    find_unused, has_lora, ddp_find_unused_cfg,
                )
            for name in ("generator_ema", "old_model", "ref_model"):
                mod = getattr(self, name)
                if mod is not None:
                    setattr(self, name, mod.to(device))
        else:
            raise ValueError(f"Unknown distributed strategy: {strategy}")

    def setup_optimizers(self) -> None:
        gen_cfg = self.config.solver.generator
        gen_params = [p for p in self.generator.parameters() if p.requires_grad]
        self.optimizers["generator"] = torch.optim.AdamW(
            gen_params,
            lr=gen_cfg.lr,
            betas=(gen_cfg.beta1, gen_cfg.beta2),
            eps=gen_cfg.eps,
            weight_decay=gen_cfg.weight_decay,
        )
        self.schedulers["generator"] = self.create_warmup_constant_scheduler(
            self.optimizers["generator"], gen_cfg.warmup_steps,
        )
        logger.info(
            f"Optimizer created: generator (lr={gen_cfg.lr}, "
            f"trainable_params={sum(p.numel() for p in gen_params) / 1e6:.1f} M)"
        )

    def _create_dataloader(self):
        """Not used — :class:`NFTTrainer` overrides :meth:`_train_loop`.

        Returns ``(None, None)`` to satisfy :class:`BaseTrainer`'s abstract
        interface. The actual K-repeat dataloader is constructed in
        :meth:`_setup_nft_dataloader`, called from ``_train_loop``.
        """
        return None, None

    def _setup_nft_dataloader(self) -> None:
        """Build the K-repeat prompt dataloader (aligned with DiffusionNFT).

        Each rank gets ``train.batch_size`` prompt indices per sampling
        iteration; globally each call draws ``M = train.batch_size *
        world_size / K`` unique prompts and K-repeats them, then shards
        across ranks. K samples per prompt are scattered across ranks and
        re-grouped via cross-rank ``all_gather`` in :meth:`_compute_advantages`.
        """
        cfg = self._nft_cfg()
        K = int(cfg.num_image_per_prompt)
        per_rank_bs = int(self.config.train.batch_size)
        ws = get_world_size()
        if (per_rank_bs * ws) % K != 0:
            raise ValueError(
                f"train.batch_size * world_size ({per_rank_bs * ws}) must be divisible "
                f"by nft.num_image_per_prompt={K} for K-repeat sampling."
            )

        dataset = PromptDataset(self.config.prompt_path, repeat=1)
        self._nft_sampler = DistributedKRepeatSampler(
            dataset_size=len(dataset),
            batch_size=per_rank_bs,
            k=K,
            num_replicas=ws,
            rank=get_rank(),
            seed=self.config.train.seed,
        )
        self._nft_dataloader = DataLoader(
            dataset,
            batch_sampler=self._nft_sampler,
            num_workers=0,
            collate_fn=lambda items: list(items),
        )
        self._nft_sampler.set_epoch(
            self._nft_epoch * int(cfg.num_batches_per_epoch)
        )
        self._nft_sample_iter = iter(self._nft_dataloader)
        if is_main_process():
            logger.info(
                f"NFT dataloader: dataset_size={len(dataset)}, K={K}, "
                f"per_rank_bs={per_rank_bs}, num_batches_per_epoch="
                f"{cfg.num_batches_per_epoch}"
            )

    # ==================================================================
    # Active MeanFlowNFT config
    # ==================================================================

    def _nft_cfg(self):
        return self.config.meanflow_nft

    # ==================================================================
    # SD3.5 helpers
    # ==================================================================

    def _predict_noise(
        self,
        model: nn.Module,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_embeds: torch.Tensor,
        guidance_scale: float = 1.0,
        uncond_text_embeddings: Optional[torch.Tensor] = None,
        uncond_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return predict_noise_sd35(
            model,
            noisy_latents,
            prompt_embeds,
            timesteps,
            pooled_embeds,
            guidance_scale=guidance_scale,
            uncond_text_embeddings=uncond_text_embeddings,
            uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds,
        )

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device("cuda")
        return encode_prompts_sd35(
            prompts,
            self.text_encoders,
            self.tokenizers,
            device,
            max_sequence_length=int(
                self._nft_cfg().text_max_sequence_length
            ),
        )

    @torch.no_grad()
    def _get_uncond_embeds(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._uncond_embeds is None:
            device = torch.device("cuda")
            self._uncond_embeds = encode_prompts_sd35(
                [""],
                self.text_encoders,
                self.tokenizers,
                device,
                max_sequence_length=int(
                    self._nft_cfg().text_max_sequence_length
                ),
            )
        emb, pool = self._uncond_embeds
        return emb.expand(batch_size, -1, -1), pool.expand(batch_size, -1)

    @torch.no_grad()
    def _decode_latents_to_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        return super()._decode_latents_to_tensor(latents)

    # ==================================================================
    # Eval adapter. MeanFlowNFTTrainer implements the actual rollout.
    # ==================================================================

    @torch.no_grad()
    def _generate_latents(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,  # noqa: ARG002 - signature compatibility w/ BaseTrainer._evaluate
        uncond_pooled: torch.Tensor,  # noqa: ARG002 - same as above
        num_steps: int = 4,
        gradient_truncation: bool = False,  # noqa: ARG002 - eval path is no_grad
        random_stop: bool = False,  # noqa: ARG002 - eval uses fixed num_steps
        model: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, int]:
        """Eval-side adapter over :meth:`_rollout_samples`.

        The signature matches :meth:`BaseTrainer._evaluate`.
        """
        if model is None:
            model = self.generator_ema if self.generator_ema is not None else self.generator
        x0 = self._rollout_samples(
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            num_steps=int(num_steps),
            guidance_scale=float(self.config.eval.eval_guidance_scale),
            model=model,
        )
        return x0, int(num_steps)

    @torch.no_grad()
    def _rollout_samples(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        *,
        num_steps: int,
        guidance_scale: float,
        model: nn.Module,
    ) -> torch.Tensor:
        raise NotImplementedError("MeanFlowNFTTrainer must implement rollout.")

    # ==================================================================
    # Velocity computation for NFT loss.
    #
    # Single dispatch: subclasses (MeanFlowNFTTrainer) override this to swap
    # in V_theta derived via central difference; callers below pick which
    # model and whether gradients flow through it.
    # ==================================================================

    def _compute_v(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        *,
        with_grad: bool,
        x_0: Optional[torch.Tensor] = None,    # noqa: ARG002 - used by MeanFlowNFT
        r_raw: Optional[torch.Tensor] = None,  # noqa: ARG002 - used by MeanFlowNFT
    ) -> torch.Tensor:
        """Compute velocity from ``model``; ``with_grad`` controls autograd.

        ``x_0`` and ``r_raw`` are consumed by the MeanFlowNFT override.
        """
        ctx = nullcontext() if with_grad else torch.no_grad()
        with ctx:
            with self._autocast():
                return self._predict_noise(
                    model, x_t, prompt_embeds, t, pooled_embeds,
                    guidance_scale=1.0,
                )

    # ------------------------------------------------------------------
    # Hooks around the 3 ``_compute_v`` calls in ``_nft_loss``.
    #
    # Subclasses can use these to set up / tear down state shared across
    # the per-network velocity computations. The canonical use case is
    # :class:`MeanFlowNFTTrainer` precomputing one central-difference
    # ``dF/dt`` on ``self.old_model`` and reusing it for generator /
    # old / ref (see ``MeanFlowNFTRLConfig.share_cd_with_old``).
    #
    # Default implementations are no-ops (matches the original NFT
    # behavior exactly).
    # ------------------------------------------------------------------

    def _pre_compute_v_hook(
        self,
        *,
        x_t: torch.Tensor,         # noqa: ARG002 - hook contract; default no-op
        t_for_model: torch.Tensor, # noqa: ARG002
        r_for_model: Optional[torch.Tensor],  # noqa: ARG002
        prompt_embeds: torch.Tensor,          # noqa: ARG002
        pooled_embeds: torch.Tensor,          # noqa: ARG002
        x_0: torch.Tensor,                    # noqa: ARG002
    ) -> None:
        """Called once before the three ``_compute_v`` calls in ``_nft_loss``.

        Receives the freshly noised ``x_t`` and per-sample ``(t, r)``
        (raw timestep units) so subclasses can precompute state that the
        three ``_compute_v`` calls will all reference (e.g. a shared
        central-difference ``dF/dt``). Default no-op.
        """
        return None

    def _post_compute_v_hook(self) -> None:
        """Called after the three ``_compute_v`` calls (always, via
        ``try / finally``), so subclasses can clear any per-step state
        they set up in :meth:`_pre_compute_v_hook`. Default no-op.
        """
        return None

    # ==================================================================
    # Reward + advantage pipeline
    # ==================================================================

    def _ensure_reward_scorer(self) -> None:
        if self._reward_scorer is not None:
            return
        cfg = self._nft_cfg()
        if not cfg.reward_fn:
            raise ValueError(
                f"{type(self).__name__} requires meanflow_nft.reward_fn "
                "with at least one weighted reward."
            )
        if cfg.reward_ckpt_path:
            set_ckpt_path(cfg.reward_ckpt_path)
        # Init on CPU and move to GPU per call (matches eval scorer pattern).
        self._reward_scorer = MultiScorer(
            device=torch.device("cpu"),
            score_dict=cfg.reward_fn,
            allow_unavailable=False,
        )

    @staticmethod
    def _normalize_score_array(vals: Any) -> torch.Tensor:
        """Convert MultiScorer per-reward output to a CPU fp32 tensor [B]."""
        if isinstance(vals, torch.Tensor):
            return vals.detach().cpu().float()
        if isinstance(vals, np.ndarray):
            return torch.from_numpy(vals).float()
        if isinstance(vals, list):
            return torch.tensor(
                [
                    float(v.detach().cpu().item()) if isinstance(v, torch.Tensor)
                    else float(v)
                    for v in vals
                ],
                dtype=torch.float32,
            )
        return torch.tensor(np.asarray(vals), dtype=torch.float32)

    @torch.no_grad()
    def _score_rewards(
        self, images: torch.Tensor, prompts: list[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score images with the multi-reward backend.

        Returns:
            mean_t:        weighted sum reward, shape [B], device=cuda.
            per_reward:    dict {name: [B] cpu fp32 tensor} for each active
                           reward backend (used for per-reward wandb panels).
        """
        self._ensure_reward_scorer()
        device = torch.device("cuda")
        self._reward_scorer.to(device)
        try:
            score_details, _ = self._reward_scorer(
                images, prompts, metadata=None, only_strict=False,
            )
            mean_vals = score_details["mean"]
            per_reward: dict[str, torch.Tensor] = {}
            for name in getattr(
                self._reward_scorer, "active_reward_names",
                list(self._nft_cfg().reward_fn.keys()),
            ):
                if name in score_details:
                    per_reward[name] = self._normalize_score_array(score_details[name])
        finally:
            self._reward_scorer.to(torch.device("cpu"))
            torch.cuda.empty_cache()

        mean_t = self._normalize_score_array(mean_vals).to(device=device)
        return mean_t, per_reward

    @torch.no_grad()
    def _gather_local_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """All-gather a 1-D tensor across ranks; returns concatenated tensor.

        Used for global reward stats so every rank reports the same mean / std.
        """
        ws = get_world_size()
        if not dist.is_initialized() or ws <= 1:
            return t.clone().to(device=torch.device("cuda"))
        t = t.contiguous().to(device=torch.device("cuda"))
        bucket = [torch.empty_like(t) for _ in range(ws)]
        dist.all_gather(bucket, t)
        return torch.cat(bucket, dim=0)

    @torch.no_grad()
    def _gather_prompts_and_rewards(
        self, prompts_local: list[str], rewards_local: torch.Tensor,
    ) -> tuple[list[str], torch.Tensor]:
        """All-gather prompts and rewards across ranks."""
        ws = get_world_size()
        if not dist.is_initialized() or ws <= 1:
            return list(prompts_local), rewards_local.clone()

        prompts_bucket: list[list[str] | None] = [None] * ws
        dist.all_gather_object(prompts_bucket, list(prompts_local))
        all_prompts: list[str] = []
        for chunk in prompts_bucket:
            if chunk:
                all_prompts.extend(chunk)

        rewards_local = rewards_local.contiguous()
        all_rewards = [torch.empty_like(rewards_local) for _ in range(ws)]
        dist.all_gather(all_rewards, rewards_local)
        all_rewards = torch.cat(all_rewards, dim=0)
        return all_prompts, all_rewards

    @torch.no_grad()
    def _compute_advantages(
        self, prompts_local: list[str], rewards_local: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample advantages with cross-rank stat tracking."""
        cfg = self._nft_cfg()
        rank = get_rank()
        ws = get_world_size()
        n_local = rewards_local.shape[0]

        all_prompts, all_rewards = self._gather_prompts_and_rewards(
            prompts_local, rewards_local,
        )

        if self._stat_tracker is None:
            # Without per-prompt tracking, fall back to global standardization.
            r = all_rewards.cpu().numpy().astype(np.float64)
            adv = (r - r.mean()) / (r.std() + 1e-4)
            adv_local = torch.from_numpy(adv).to(rewards_local.device, dtype=torch.float32)
            return adv_local[rank * n_local:(rank + 1) * n_local]

        all_adv = self._stat_tracker.update(all_prompts, all_rewards.cpu().numpy())
        adv_local_np = all_adv[rank * n_local:(rank + 1) * n_local]
        return torch.from_numpy(adv_local_np).to(
            device=rewards_local.device, dtype=torch.float32,
        )

    @torch.no_grad()
    def _compute_advantages_weighted(
        self,
        prompts_local: list[str],
        rewards_local: torch.Tensor,
        per_reward_local: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """``weight_advantages=True``: per-reward advantages, then weight + normalize.

        Instead of normalizing the weighted-sum reward (see
        :meth:`_compute_advantages`), compute EACH reward's advantage
        independently -- via its own fresh per-prompt stat tracker, or global
        standardization when per-prompt tracking is off -- weighted-sum the
        advantages, then normalize the result to zero-mean / unit-std. Mirrors the
        DiffusionNFT's ``weight_advantages`` path. Returns this rank's slice.

        Falls back to :meth:`_compute_advantages` if no per-reward scores are
        available (e.g. every reward backend was skipped).
        """
        cfg = self._nft_cfg()
        rank = get_rank()
        n_local = rewards_local.shape[0]
        weights: dict[str, float] = dict(cfg.reward_fn or {})

        # Gather prompts once (+ the weighted-sum reward, used only to keep the
        # main stat tracker's summary stats meaningful in this mode).
        all_prompts, all_weighted = self._gather_prompts_and_rewards(
            prompts_local, rewards_local,
        )

        total_adv: np.ndarray | None = None
        for name, weight in weights.items():
            if name not in per_reward_local:
                continue
            r_local = per_reward_local[name].to(device=rewards_local.device).float()
            all_r = self._gather_local_tensor(r_local)
            r_np = all_r.cpu().numpy().astype(np.float64)
            if cfg.per_prompt_stat_tracking:
                tracker = PerPromptStatTracker(global_std=cfg.per_prompt_global_std)
                adv = np.asarray(tracker.update(all_prompts, r_np), dtype=np.float64)
            else:
                adv = (r_np - r_np.mean()) / (r_np.std() + 1e-4)
            adv = adv * float(weight)
            total_adv = adv if total_adv is None else total_adv + adv

        if total_adv is None:
            # No per-reward scores available -> behave like Mode 1.
            return self._compute_advantages(prompts_local, rewards_local)

        # Normalize the weighted-sum advantage (zero-mean / unit-std).
        total_adv = (total_adv - total_adv.mean()) / (total_adv.std() + 1e-4)

        # Populate the main tracker on the weighted-sum reward so the
        # DiffusionNFT-style summary stats in ``_collect_reward_stats`` stay valid
        # (the caller clears it afterwards).
        if self._stat_tracker is not None:
            self._stat_tracker.update(all_prompts, all_weighted.cpu().numpy())

        adv_local = np.ascontiguousarray(total_adv[rank * n_local:(rank + 1) * n_local])
        return torch.from_numpy(adv_local).to(
            device=rewards_local.device, dtype=torch.float32,
        )

    # ==================================================================
    # NFT loss kernel (single (x_0, t, eps, advantage) microbatch)
    # ==================================================================

    def _nft_loss(
        self,
        x_0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        advantages: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One NFT step (matches DiffusionNFT/scripts/train_nft_sd3.py exactly).

        ``t_raw`` is the per-sample timestep in raw [0, T] units (drawn by
        :meth:`_nft_train_phase` from the rollout's discrete scheduler.timesteps;
        no continuous sampling is done here).

        Algorithm:
            1. Forward noising: x_t = (1-t/T) * x_0 + (t/T) * eps,  v_t = eps - x_0.
            2. Build implicit policies in v-space:
                   v_theta_plus  = beta * v_theta + (1-beta) * v_old
                   v_theta_minus = (1+beta) * v_old - beta * v_theta
            3. Convert to x_0 space via the Tweedie identity for linear flow:
                   x0_plus  = x_t - t01 * v_theta_plus
                   x0_minus = x_t - t01 * v_theta_minus
            4. Adaptive per-sample weighting (DMD-style, computed in fp64):
                   w_pos = sg( mean(|x0_plus  - x_0|).clip(min=1e-5) )
                   w_neg = sg( mean(|x0_minus - x_0|).clip(min=1e-5) )
            5. Per-sample x0-space losses, normalized by their own weight:
                   pos_loss = mean( (x0_plus  - x_0)^2 / w_pos )
                   neg_loss = mean( (x0_minus - x_0)^2 / w_neg )
            6. r_hat = clip( advantages/adv_clip_max, -1, 1 ) / 2 + 0.5
            7. policy_loss = mean( r_hat * pos_loss / beta
                                 + (1-r_hat) * neg_loss / beta ) * adv_clip_max
            8. (Optional) KL: + kl_weight * mean( (v_theta - v_ref)^2 )
        """
        cfg = self._nft_cfg()
        dtype = x_0.dtype
        batch_size = x_0.shape[0]
        T = float(self.scheduler.config.num_train_timesteps)

        # 1. Forward noise (t01 in [0,1], t_for_model in [0,T]).
        t_raw = t_raw.to(device=x_0.device)
        t01 = (t_raw.to(dtype) / T)
        t_broadcast = t01.view(batch_size, *([1] * (x_0.ndim - 1)))
        eps = torch.randn_like(x_0)
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * eps
        t_for_model = t_raw.to(dtype=torch.float32)

        # 2. Velocity predictions. MeanFlowNFT uses ``x_0`` and ``r_raw`` for its
        # central-difference V_theta construction.
        r_for_model = r_raw.to(dtype=torch.float32) if r_raw is not None else None

        # Hook for subclasses to set up state shared across the 3 ``_compute_v``
        # calls below (e.g. ``MeanFlowNFTTrainer`` uses this to precompute one
        # central-difference dF/dt on ``self.old_model`` and reuse it for all
        # networks; see ``share_cd_with_old`` in MeanFlowNFTRLConfig). Default
        # no-op. Always paired with ``_post_compute_v_hook`` in a try/finally.
        self._pre_compute_v_hook(
            x_t=x_t, t_for_model=t_for_model, r_for_model=r_for_model,
            prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds,
            x_0=x_0,
        )
        try:
            v_theta = self._compute_v(
                self.generator, x_t, t_for_model, prompt_embeds, pooled_embeds,
                with_grad=True, x_0=x_0, r_raw=r_for_model,
            )
            v_old = self._compute_v(
                self.old_model, x_t, t_for_model, prompt_embeds, pooled_embeds,
                with_grad=False, x_0=x_0, r_raw=r_for_model,
            ).detach()
            v_ref = None
            if self.ref_model is not None and cfg.kl_weight > 0.0:
                v_ref = self._compute_v(
                    self.ref_model, x_t, t_for_model, prompt_embeds, pooled_embeds,
                    with_grad=False, x_0=x_0, r_raw=r_for_model,
                ).detach()
        finally:
            self._post_compute_v_hook()

        beta = float(cfg.beta)
        adv_clip_max = float(cfg.adv_clip_max)
        if adv_clip_max <= 0:
            raise ValueError(f"adv_clip_max must be > 0, got {adv_clip_max}")

        # 3-5. Build implicit policies, convert to x0 space, adaptive weight.
        v_theta_f = v_theta.float()
        v_old_f = v_old.float()
        x_t_f = x_t.float()
        x_0_f = x_0.float()
        t_b_f = t_broadcast.float()

        v_plus = beta * v_theta_f + (1.0 - beta) * v_old_f
        v_minus = (1.0 + beta) * v_old_f - beta * v_theta_f
        x0_plus = x_t_f - t_b_f * v_plus
        x0_minus = x_t_f - t_b_f * v_minus

        spatial = tuple(range(1, x_0.ndim))
        with torch.no_grad():
            w_pos = (
                (x0_plus.double() - x_0_f.double())
                .abs()
                .mean(dim=spatial, keepdim=True)
                .clip(min=1e-5)
            )
            w_neg = (
                (x0_minus.double() - x_0_f.double())
                .abs()
                .mean(dim=spatial, keepdim=True)
                .clip(min=1e-5)
            )
        # NOTE: do NOT cast w_pos / w_neg back to x0_plus.dtype here. The
        # fp32 numerator divided by fp64 weight_factor promotes to fp64,
        # then .mean() reduces in fp64 — this matches DiffusionNFT exactly.
        pos_loss = (((x0_plus - x_0_f) ** 2) / w_pos).mean(dim=spatial)
        neg_loss = (((x0_minus - x_0_f) ** 2) / w_neg).mean(dim=spatial)

        # 6-7. r_hat + scaled policy loss.
        adv = advantages.float().clamp(-adv_clip_max, adv_clip_max)
        r_hat = ((adv / adv_clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)
        unscaled = r_hat * pos_loss / beta + (1.0 - r_hat) * neg_loss / beta
        policy_loss = (unscaled * adv_clip_max).mean()
        loss = policy_loss

        info = {
            "loss_policy": float(policy_loss.detach()),
            "loss_unscaled": float(unscaled.mean().detach()),
            "t_mean": float(t_raw.mean().item()),
            "r_hat_mean": float(r_hat.mean().item()),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std().item()),
            "pos_loss_mean": float(pos_loss.mean().item()),
            "neg_loss_mean": float(neg_loss.mean().item()),
            "x0_norm_mean": float((x_0_f ** 2).mean().item()),
            "old_deviate": float(((v_theta_f - v_old_f) ** 2).mean().item()),
        }

        # 8. Optional KL regularization toward a frozen reference (v-space).
        if v_ref is not None:
            kl = ((v_theta_f - v_ref.float()) ** 2).mean(dim=spatial).mean()
            loss = loss + float(cfg.kl_weight) * kl
            info["loss_kl"] = float(kl.detach())
        info["loss_total"] = float(loss.detach())
        return loss, info

    # ==================================================================
    # Old model EMA decay (toward generator)
    # ==================================================================

    def _current_decay(self) -> float:
        """DiffusionNFT-aligned decay schedule.

        Mirrors ``DiffusionNFT/scripts/train_nft_sd3.py::return_decay`` line-
        by-line. The driving counter is ``self.global_step`` (inner gradient
        step count), NOT outer epochs — when num_batches_per_epoch *
        num_inner_epochs * (per-sample timesteps) == 1, the two coincide.
        """
        cfg = self._nft_cfg()
        step = int(self.global_step)
        if cfg.decay_type == 0:
            flat, uprate, uphold = 0, 0.0, 0.0
        elif cfg.decay_type == 1:
            flat, uprate, uphold = 0, 0.001, 0.5
        elif cfg.decay_type == 2:
            flat, uprate, uphold = 75, 0.0075, 0.999
        elif cfg.decay_type == 3:
            return 1.0
        else:
            raise ValueError(f"Unknown nft.decay_type={cfg.decay_type}")
        if step < flat:
            return 0.0
        return min(float((step - flat) * uprate), float(uphold))

    def _current_ema_decay(self) -> float:
        """DiffusionNFT EMA warmup schedule for ``generator_ema``.

        Uses DiffusionNFT's EMA warmup schedule:

            decay(s) = min((1 + s) / (10 + s), target_decay)

        where ``s = self.global_step`` (per-opt-step counter). Early steps
        use a smaller decay so EMA tracks the model quickly; the cap kicks
        in once ``(1+s)/(10+s) >= target_decay``.

        ``target_decay`` is read from ``train.ema_decay`` (DiffusionNFT
        default 0.9, but configurable).
        """
        s = int(self.global_step)
        target = float(self.config.train.ema_decay)
        return min((1.0 + float(s)) / (10.0 + float(s)), target)

    def _decay_old_model(self) -> float:
        cfg = self._nft_cfg()
        if cfg.nft_decay_interval <= 0:
            return 0.0
        if (self._nft_epoch + 1) % cfg.nft_decay_interval != 0:
            return 0.0
        decay = self._current_decay()
        # ema_update(src, tgt, decay): tgt = decay * tgt + (1 - decay) * src.
        # We want old <- decay * old + (1 - decay) * generator, so src=gen, tgt=old.
        self.ema_update(self.generator, self.old_model, decay)
        return decay

    # ==================================================================
    # Sampling phase: collect num_batches_per_epoch sampling rounds
    # ==================================================================

    @torch.no_grad()
    def _one_sampling_batch(self, prompts: list[str]) -> dict[str, Any]:
        """Roll out one K-repeat batch on this rank, return (x_0, embeds, rewards)."""
        cfg = self._nft_cfg()
        dtype = next(self.generator.parameters()).dtype

        with self._autocast():
            prompt_embeds, pooled_embeds = self._encode_prompts(prompts)
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        pooled_embeds = pooled_embeds.to(
            dtype=dtype if pooled_embeds.is_floating_point() else pooled_embeds.dtype,
        )

        num_steps = int(cfg.sampling_num_steps)
        sampler_model = (
            self.generator_ema if (cfg.nft_sample_use_ema and self.generator_ema is not None)
            else self.old_model
        )
        sampler_model.eval()

        x_0 = self._rollout_samples(
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            num_steps=num_steps,
            guidance_scale=cfg.sampling_guidance_scale,
            model=sampler_model,
        )
        # Capture the rollout's (t, r) jump schedule. DiffusionNFT stores
        # ``pipeline.scheduler.timesteps`` and ``next_timesteps``; we mirror
        # that pattern so MeanFlowNFT's flow-map central diff can use the exact
        # (t, r) pair the rollout used at inference.
        t_arr, r_arr = self._get_last_rollout_timestep_pairs()
        t_per_sample = t_arr.detach().cpu().repeat(x_0.shape[0], 1)
        r_per_sample = r_arr.detach().cpu().repeat(x_0.shape[0], 1)

        images = self._decode_latents_to_tensor(x_0)
        rewards, per_reward = self._score_rewards(images, prompts)

        return {
            "x_0": x_0.detach().cpu(),
            "prompt_embeds": prompt_embeds.detach().cpu(),
            "pooled_embeds": pooled_embeds.detach().cpu(),
            "timesteps": t_per_sample,
            "next_timesteps": r_per_sample,
            "prompts": list(prompts),
            "rewards": rewards.detach(),
            "per_reward": per_reward,
            "images_preview": images.detach().cpu() if cfg.log_sample_images > 0 else None,
        }

    def _get_last_rollout_timestep_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "MeanFlowNFTTrainer must expose its rollout timestep pairs."
        )

    def _nft_sampling_phase(self) -> tuple[dict[str, torch.Tensor], list[str]]:
        """Run ``num_batches_per_epoch`` sampling rounds + global advantage compute.

        Returns a concatenated dict of CPU tensors per rank
        (``x_0``, ``prompt_embeds``, ``pooled_embeds``, ``advantages``) plus
        the flat prompt list (for image logging).
        """
        cfg = self._nft_cfg()
        N = max(1, int(cfg.num_batches_per_epoch))
        self.generator.eval()

        batches: list[dict[str, Any]] = []
        first_preview = None
        for i in range(N):
            # Advance the K-repeat partition deterministically per global
            # sampling batch (matches DiffusionNFT's
            # `epoch * num_batches_per_epoch + i` seeding).
            sampler_epoch = self._nft_epoch * N + i
            self._nft_sampler.set_epoch(sampler_epoch)
            try:
                prompts = next(self._nft_sample_iter)
            except StopIteration:
                self._nft_sample_iter = iter(self._nft_dataloader)
                prompts = next(self._nft_sample_iter)
            assert isinstance(prompts, list) and len(prompts) > 0

            pack = self._one_sampling_batch(prompts)
            batches.append(pack)
            if first_preview is None and pack["images_preview"] is not None:
                first_preview = (pack["images_preview"], pack["prompts"])

        # Concatenate per-rank slices across the sampling phase.
        x_0_all = torch.cat([b["x_0"] for b in batches], dim=0)
        emb_all = torch.cat([b["prompt_embeds"] for b in batches], dim=0)
        pool_all = torch.cat([b["pooled_embeds"] for b in batches], dim=0)
        ts_all = torch.cat([b["timesteps"] for b in batches], dim=0)
        nts_all = torch.cat([b["next_timesteps"] for b in batches], dim=0)
        prompts_all: list[str] = [p for b in batches for p in b["prompts"]]
        rewards_all = torch.cat([b["rewards"] for b in batches], dim=0)

        # Per-reward concatenation across sampling rounds.
        per_reward_all: dict[str, torch.Tensor] = {}
        reward_keys = list(batches[0].get("per_reward", {}).keys())
        for k in reward_keys:
            per_reward_all[k] = torch.cat(
                [b["per_reward"][k] for b in batches], dim=0,
            )

        # Cross-rank gather + per-prompt advantage. After computing
        # advantages, capture stat-tracker summary stats (DiffusionNFT-style
        # logging), then clear so the next epoch's stats are derived only
        # from that epoch's K samples (pure on-policy normalization).
        if getattr(cfg, "weight_advantages", False):
            advantages = self._compute_advantages_weighted(
                prompts_all, rewards_all, per_reward_all,
            )
        else:
            advantages = self._compute_advantages(prompts_all, rewards_all)
        reward_stats = self._collect_reward_stats(prompts_all, rewards_all)
        if self._stat_tracker is not None:
            self._stat_tracker.clear()

        sample_pack = {
            "x_0": x_0_all,
            "prompt_embeds": emb_all,
            "pooled_embeds": pool_all,
            "timesteps": ts_all,
            "next_timesteps": nts_all,
            "prompts": prompts_all,
            "rewards": rewards_all,
            "per_reward": per_reward_all,
            "advantages": advantages,
            "reward_stats": reward_stats,
        }
        return sample_pack, first_preview

    @torch.no_grad()
    def _collect_reward_stats(
        self, prompts_all: list[str], rewards_all: torch.Tensor,
    ) -> dict[str, float]:
        """Capture per-prompt summary stats BEFORE ``stat_tracker.clear()``.

        Aligned with DiffusionNFT's wandb log block:
            group_size, trained_prompt_num
            zero_std_ratio, reward_std_mean
            mean_reward_{100, 75, 50, 25, 10}

        Returns a dict of CPU floats; safe to log from any rank (the
        tracker's per-rank stats already include all globally-gathered
        prompts/rewards via ``_compute_advantages``'s ``all_gather``).
        """
        stats: dict[str, float] = {}
        if self._stat_tracker is not None:
            group_size, trained_prompt_num = self._stat_tracker.get_stats()
            stats["group_size"] = float(group_size)
            stats["trained_prompt_num"] = float(trained_prompt_num)
            for pct in (100, 75, 50, 25, 10):
                stats[f"mean_reward_{pct}"] = float(
                    self._stat_tracker.get_mean_of_top_rewards(pct)
                )
        prompts_arr = np.array(prompts_all)
        rewards_np = rewards_all.detach().cpu().numpy().astype(np.float64)
        unique_prompts, inverse_indices, counts = np.unique(
            prompts_arr, return_inverse=True, return_counts=True,
        )
        if len(unique_prompts) > 0 and rewards_np.size > 0:
            grouped = rewards_np[np.argsort(inverse_indices)]
            split_indices = np.cumsum(counts)[:-1]
            reward_groups = np.split(grouped, split_indices)
            prompt_stds = np.array([np.std(g) for g in reward_groups])
            zero_std_count = int(np.count_nonzero(prompt_stds == 0))
            stats["zero_std_ratio"] = float(zero_std_count / len(prompt_stds))
            stats["reward_std_mean"] = float(prompt_stds.mean())
        return stats

    # ==================================================================
    # Inner train phase: num_inner_epochs * batches * num_timesteps updates
    # ==================================================================

    # ==================================================================
    # Training-time (t, r) hooks implemented by MeanFlowNFTTrainer
    # ==================================================================

    def _num_training_timesteps(self, n_ts_total: int) -> int:
        del n_ts_total
        return max(
            1, int(self._nft_cfg().num_training_timesteps_per_sample)
        )

    def _prepare_inner_epoch_tr_state(
        self,
        *,
        timesteps_all: torch.Tensor,
        next_timesteps_all: torch.Tensor,
        sample_perm: torch.Tensor,
        n_ts_total: int,
        n_train_ts: int,    # noqa: ARG002 - used by MeanFlowNFT override
        device: torch.device,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "MeanFlowNFTTrainer must prepare its (t, r) sampling state."
        )

    def _draw_training_tr(
        self,
        *,
        state: dict[str, Any],
        chunk_start: int,
        chunk_end: int,
        j_idx: int,
        inner_epoch_idx: int,   # noqa: ARG002 - used by MeanFlowNFT override
        device: torch.device,   # noqa: ARG002 - used by MeanFlowNFT override
        dtype: torch.dtype,     # noqa: ARG002 - used by MeanFlowNFT override
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "MeanFlowNFTTrainer must draw training (t, r) pairs."
        )

    def _nft_train_phase(
        self, sample_pack: dict[str, Any],
    ) -> dict[str, float]:
        """Run ``num_inner_epochs`` x ``num_batches`` x ``num_train_timesteps`` updates.

        Aligned with ``DiffusionNFT/scripts/train_nft_sd3.py`` (lines 869-1100)
        line-by-line:

            effective_grad_accum_steps = gradient_accumulation_steps * num_train_timesteps
            current_accumulated_steps = 0
            optimizer.zero_grad()

            for inner_epoch in range(num_inner_epochs):
                perm = randperm(B_local)
                # shuffle samples; per-sample independent timestep shuffle
                for k_batch in chunks(B_local, train_bs):
                    for j_idx in range(num_train_timesteps):
                        (t, r) <- _draw_training_tr(...)
                        loss = _nft_loss(sub.x_0, ..., t, r)
                        (loss / effective_grad_accum_steps).backward()
                        current_accumulated_steps += 1
                        if current_accumulated_steps % effective_grad_accum_steps == 0:
                            clip + optimizer.step + scheduler.step + zero_grad
                            ema.step  (per opt step, not per backward)
                            global_step += 1

        Notes:
        - ``global_step`` increments ONLY on opt.step (DiffusionNFT convention).
        - The loss is divided by ``effective_grad_accum_steps`` before backward
          so the accumulated gradient equals the mean of per-backward losses
          across the accumulation window.
        - ``num_train_timesteps`` is configured independently of rollout NFE.
        - MeanFlowNFT draws a fresh three-mode (t, r) partition per iteration.
        """
        cfg = self._nft_cfg()
        device = torch.device("cuda")

        x_0 = sample_pack["x_0"].to(device)
        prompt_embeds = sample_pack["prompt_embeds"].to(device)
        pooled_embeds = sample_pack["pooled_embeds"].to(device)
        advantages = sample_pack["advantages"].to(device)
        timesteps_all = sample_pack["timesteps"].to(device=device, dtype=torch.float32)
        next_timesteps_all = sample_pack["next_timesteps"].to(device=device, dtype=torch.float32)
        n_local, n_ts_total = timesteps_all.shape
        if n_local == 0:
            return {}

        n_train_ts = self._num_training_timesteps(n_ts_total)

        inner_bs = int(cfg.nft_inner_batch_size) if cfg.nft_inner_batch_size > 0 else n_local
        inner_bs = max(1, min(inner_bs, n_local))
        n_inner_epochs = max(1, int(cfg.num_inner_epochs))
        max_grad_norm = float(self.config.solver.generator.max_grad_norm)
        grad_accum = max(1, int(self.config.train.gradient_accumulation_steps))
        effective_grad_accum_steps = grad_accum * n_train_ts

        self.generator.train()
        accumulator: dict[str, float] = {}
        n_backward = 0       # cumulative backwards across the whole train phase
        n_opt_steps = 0
        current_accumulated_steps = 0
        self.optimizers["generator"].zero_grad(set_to_none=True)
        last_grad_norm: float = 0.0

        dtype = next(self.generator.parameters()).dtype

        for inner_epoch_idx in range(n_inner_epochs):
            # Per-sample shuffle (DiffusionNFT line 870-871).
            perm = torch.randperm(n_local, device=device)
            sx0 = x_0[perm]
            spe = prompt_embeds[perm]
            spp = pooled_embeds[perm]
            sadv = advantages[perm]
            # MeanFlowNFT draws fresh per-iteration (t, r) pairs.
            tr_state = self._prepare_inner_epoch_tr_state(
                timesteps_all=timesteps_all,
                next_timesteps_all=next_timesteps_all,
                sample_perm=perm,
                n_ts_total=n_ts_total,
                n_train_ts=n_train_ts,
                device=device,
            )

            for start in range(0, n_local, inner_bs):
                end = start + inner_bs
                x0_b = sx0[start:end]
                pe_b = spe[start:end]
                pp_b = spp[start:end]
                adv_b = sadv[start:end]
                for j_idx in range(n_train_ts):
                    t_raw, r_raw = self._draw_training_tr(
                        state=tr_state,
                        chunk_start=start,
                        chunk_end=end,
                        j_idx=j_idx,
                        inner_epoch_idx=inner_epoch_idx,
                        device=device,
                        dtype=dtype,
                    )
                    loss, info = self._nft_loss(
                        x_0=x0_b,
                        prompt_embeds=pe_b,
                        pooled_embeds=pp_b,
                        advantages=adv_b,
                        t_raw=t_raw,
                        r_raw=r_raw,
                    )
                    # Scale loss for grad accumulation; gradients accumulate
                    # across `effective_grad_accum_steps` backward calls.
                    (loss / float(effective_grad_accum_steps)).backward()
                    current_accumulated_steps += 1
                    n_backward += 1

                    for k, v in info.items():
                        accumulator[k] = accumulator.get(k, 0.0) + float(v)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        last_grad_norm = self._compute_grad_norm(self.generator)
                        if max_grad_norm > 0:
                            self._clip_grad_norm(self.generator, max_grad_norm)
                        self.optimizers["generator"].step()
                        self.schedulers["generator"].step()
                        self.optimizers["generator"].zero_grad(set_to_none=True)

                        # EMA per opt step (DiffusionNFT line 1088-1092).
                        # Uses the warmup-aware decay (matches
                        # EMAModuleWrapper.get_current_decay).
                        if (
                            self.generator_ema is not None
                            and self.global_step % self.config.train.ema_every == 0
                        ):
                            self.ema_update(
                                self.generator, self.generator_ema,
                                self._current_ema_decay(),
                            )

                        # ``global_step`` increments per OPT step (DiffusionNFT
                        # convention; drives ``return_decay(global_step, ...)``).
                        self.global_step += 1
                        n_opt_steps += 1
                        accumulator["grad_norm"] = (
                            accumulator.get("grad_norm", 0.0) + float(last_grad_norm)
                        )

        if n_backward == 0:
            return {}
        averaged = {k: v / n_backward for k, v in accumulator.items()}
        if n_opt_steps > 0:
            averaged["grad_norm"] = accumulator.get("grad_norm", 0.0) / n_opt_steps
        averaged["lr"] = self.schedulers["generator"].get_last_lr()[0]
        averaged["inner_backward_steps"] = float(n_backward)
        averaged["inner_opt_steps"] = float(n_opt_steps)
        averaged["num_train_timesteps"] = float(n_train_ts)
        averaged["effective_grad_accum_steps"] = float(effective_grad_accum_steps)
        return averaged

    # ==================================================================
    # One outer NFT epoch
    # ==================================================================

    def _run_nft_epoch(self) -> dict[str, dict[str, float]]:
        """One NFT epoch: sampling + inner update + old_model decay.

        Returns a metrics dict whose top-level keys are wandb section names:

        - ``nft_reward``    — reward summary + DiffusionNFT stat-tracker stats
        - ``nft_advantage`` — advantage / r_hat distribution
        - ``nft_loss``      — per-iteration policy / KL losses
        - ``nft_train``     — optimization counters (lr, grad_norm, decay, ...)
        - ``nft_diagnostic``— sanity-check signals (x0 norm, old/gen deviation)

        The base trainer's :meth:`_log_step` automatically adds a ``profile``
        section with GPU memory + timer phases.
        """
        cfg = self._nft_cfg()
        active = self._nft_epoch >= int(cfg.nft_start_epoch)

        with self.timer.measure("nft_sampling"):
            sample_pack, preview = self._nft_sampling_phase()

        # Build reward / advantage sections from the sampling phase output.
        # Use the global (cross-rank gathered) rewards/per_reward for stats so
        # wandb reports the same value on every rank; the per-rank tensors are
        # local slices and would otherwise differ.
        gathered_rewards_all = self._gather_local_tensor(sample_pack["rewards"])
        reward_section: dict[str, float] = {
            "reward_mean": float(gathered_rewards_all.mean().item()),
            "reward_std": float(gathered_rewards_all.std().item()),
            "reward_min": float(gathered_rewards_all.min().item()),
            "reward_max": float(gathered_rewards_all.max().item()),
        }
        # Per-reward breakdown (DiffusionNFT logs each scorer separately).
        for name, vals in sample_pack.get("per_reward", {}).items():
            gathered = self._gather_local_tensor(vals.to(sample_pack["rewards"].device))
            reward_section[f"{name}_mean"] = float(gathered.mean().item())
            reward_section[f"{name}_std"] = float(gathered.std().item())
            reward_section[f"{name}_min"] = float(gathered.min().item())
            reward_section[f"{name}_max"] = float(gathered.max().item())
        reward_section.update(sample_pack.get("reward_stats", {}))

        advantage_section: dict[str, float] = {
            "advantage_mean": float(sample_pack["advantages"].mean().item()),
            "advantage_std": float(sample_pack["advantages"].std().item()),
            "advantage_abs_mean": float(sample_pack["advantages"].abs().mean().item()),
        }

        loss_section: dict[str, float] = {}
        train_section: dict[str, float] = {}
        diagnostic_section: dict[str, float] = {
            "n_samples_local": float(sample_pack["x_0"].shape[0]),
        }

        if active:
            with self.timer.measure("nft_update"):
                upd = self._nft_train_phase(sample_pack)

            # Dispatch upd entries into the section dicts.
            for k in ("loss_policy", "loss_unscaled", "loss_kl",
                      "loss_total", "pos_loss_mean", "neg_loss_mean"):
                if k in upd:
                    loss_section[k] = float(upd[k])
            for k in ("lr", "grad_norm", "inner_backward_steps",
                      "inner_opt_steps", "num_train_timesteps",
                      "effective_grad_accum_steps", "t_mean"):
                if k in upd:
                    train_section[k] = float(upd[k])
            if "r_hat_mean" in upd:
                advantage_section["r_hat_mean"] = float(upd["r_hat_mean"])
            for k in ("x0_norm_mean", "old_deviate"):
                if k in upd:
                    diagnostic_section[k] = float(upd[k])

            with self.timer.measure("nft_old_decay"):
                decay_val = self._decay_old_model()
            train_section["old_decay"] = float(decay_val)

        metrics: dict[str, dict[str, float]] = {
            "nft_reward": reward_section,
            "nft_advantage": advantage_section,
        }
        if loss_section:
            metrics["nft_loss"] = loss_section
        if train_section:
            metrics["nft_train"] = train_section
        if diagnostic_section:
            metrics["nft_diagnostic"] = diagnostic_section

        # Optional preview image logging.
        if (
            cfg.log_sample_images > 0
            and is_main_process()
            and self.wandb_logger is not None
            and preview is not None
        ):
            imgs, prompts_preview = preview
            n = min(int(cfg.log_sample_images), imgs.shape[0])
            try:
                from PIL import Image as _PIL
                pil_imgs = [
                    _PIL.fromarray(
                        (img.permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    )
                    for img in imgs[:n]
                ]
                self.wandb_logger.log_images(
                    images=pil_imgs,
                    captions=prompts_preview[:n],
                    step=self.global_step,
                    key="media/nft_samples",
                )
            except Exception as e:
                logger.warning(f"NFT sample image logging failed: {e}")

        return metrics

    # ==================================================================
    # Main training loop override (epoch-based, aligned with DiffusionNFT)
    # ==================================================================

    def train_step(self, batch: Any) -> dict[str, dict[str, float]]:
        """Not used — :class:`NFTTrainer` overrides :meth:`_train_loop`."""
        raise NotImplementedError("NFTTrainer uses _train_loop() override")

    def _train_loop(self) -> None:
        """Epoch-based loop following DiffusionNFT's outer-loop structure."""
        cfg = self._nft_cfg()
        train_cfg = self.config.train

        # Build K-repeat dataloader / sampler / iterator.
        self._setup_nft_dataloader()

        # Restore RNG state if resuming.
        self._restore_rng_state()

        if is_main_process():
            logger.info(
                f"Starting NFT training: num_epochs={cfg.num_epochs}, "
                f"num_batches_per_epoch={cfg.num_batches_per_epoch}, "
                f"num_inner_epochs={cfg.num_inner_epochs}, "
                f"K={cfg.num_image_per_prompt}, "
                f"per_rank_bs={train_cfg.batch_size}"
            )

        while self._nft_epoch < int(cfg.num_epochs):
            # Eval at epoch boundary.
            eval_cfg = self.config.eval
            if (
                eval_cfg.enabled
                and eval_cfg.eval_interval > 0
                and self._nft_epoch % eval_cfg.eval_interval == 0
            ):
                with self.timer.measure("eval"):
                    self._evaluate()

            # Save at epoch boundary (skip step 0).
            if (
                train_cfg.save_interval > 0
                and self._nft_epoch % train_cfg.save_interval == 0
                and self._nft_epoch > 0
            ):
                self._save_checkpoint()

            with self.timer.measure("epoch"):
                metrics = self._run_nft_epoch()

            # Log epoch summary.
            if self._nft_epoch % train_cfg.log_interval == 0:
                self._log_step(metrics)

            self._nft_epoch += 1

        logger.info(
            f"NFT training complete: epoch={self._nft_epoch}, "
            f"global_step={self.global_step}"
        )
        self._save_checkpoint()
        self._save_final_transformer()

    # ==================================================================
    # Model initialization helper
    # ==================================================================

    def _load_model_init_from_path(
        self, model: nn.Module, init_path: str, role_name: str,
    ) -> None:
        if not init_path:
            return
        init_path = str(init_path).strip()
        if not init_path:
            return
        if os.path.isfile(init_path):
            state_dict = torch.load(init_path, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(
                f"  {role_name}: loaded init checkpoint {init_path} "
                f"(keys={len(state_dict)}, missing={len(missing)}, unexpected={len(unexpected)})"
            )
            return
        from meanflownft.utils.fast_init import fast_init
        model_dir = init_path
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            alt = os.path.join(model_dir, "transformer")
            if os.path.exists(os.path.join(alt, "config.json")):
                model_dir = alt
            else:
                raise FileNotFoundError(
                    f"{role_name}_init_path has no config.json at {init_path}"
                )
        with fast_init(torch.device("cpu")):
            loaded = model.__class__.from_pretrained(
                model_dir, torch_dtype=next(model.parameters()).dtype,
            )
        sd = loaded.state_dict()
        del loaded
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(
            f"  {role_name}: loaded init transformer dir {model_dir} "
            f"(keys={len(sd)}, missing={len(missing)}, unexpected={len(unexpected)})"
        )

    # ==================================================================
    # Gradient-norm helpers
    # ==================================================================

    @staticmethod
    def _compute_grad_norm(model: nn.Module) -> float:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            total = torch.zeros((), device=torch.cuda.current_device())
            for p in model.parameters():
                if p.grad is not None:
                    total = total + p.grad.detach().float().norm() ** 2
            total = total.clone()
            if dist.is_initialized():
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
            return float(total.sqrt().item())
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().float().norm().item()) ** 2
        return total ** 0.5

    @staticmethod
    def _clip_grad_norm(model: nn.Module, max_norm: float) -> None:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            model.clip_grad_norm_(max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # ==================================================================
    # Checkpoint hooks (per-rank stat tracker)
    # ==================================================================

    def _get_extra_checkpoint_state(self) -> dict:
        """Save NFT outer-loop counters + per-rank stat tracker."""
        extra = {"nft_epoch": self._nft_epoch}
        if self._stat_tracker is not None:
            extra["nft_stat_tracker"] = {
                "rank": get_rank(),
                "state": self._stat_tracker.state_dict(),
            }
        return extra

    def _restore_extra_checkpoint_state(self, extra: dict) -> None:
        """Restore NFT outer-loop counters + per-rank stat tracker."""
        self._nft_epoch = int(extra.get("nft_epoch", 0))
        if self._stat_tracker is not None:
            st = extra.get("nft_stat_tracker")
            if st:
                self._stat_tracker.load_state_dict(st.get("state", {}))

    def _get_lora_model_names(self) -> set[str]:
        names = super()._get_lora_model_names()
        if self.config.model.generator_lora.enabled:
            if self.old_model is not None:
                names.add("old_model")
            if self.ref_model is not None:
                names.add("ref_model")
        return names

    def _get_lora_configs(self) -> dict[str, Any]:
        cfgs = super()._get_lora_configs()
        if self.config.model.generator_lora.enabled:
            if self.old_model is not None:
                cfgs["old_model"] = self.config.model.generator_lora
            if self.ref_model is not None:
                cfgs["ref_model"] = self.config.model.generator_lora
        return cfgs
