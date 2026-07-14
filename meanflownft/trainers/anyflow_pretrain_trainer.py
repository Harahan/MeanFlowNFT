"""
AnyFlow Stage 1 (Forward Training) Trainer for MeanFlowNFT.

Implements the forward training stage of the AnyFlow paper
(https://arxiv.org/abs/2605.13724) inside the MeanFlowNFT framework.

Algorithm summary (per train_step):

1. Sample (t, r) per sample using a 3-mode partition of the *global* batch:
     - first ``round(diffusion_ratio * global_bsz)`` samples: r = t
       (degenerates to standard flow matching loss)
     - next ``round(consistency_ratio * global_bsz)`` samples: r = 0
       (endpoint consistency mapping)
     - the rest: r ~ U(0, t) (generic flow map transition)

2. Forward noise: ``z_t = (1 - t/T) * x_0 + (t/T) * noise``.

3. (No-grad) Compute the flow-map material derivative ``dF/dt`` along the
   ODE trajectory by central difference:

       z_{t±ε} = z_t ± v_pred * (ε / T)
       F_± = transformer(z_{t±ε}, t ± ε, r)
       dF/dt = (F_+ - F_-) / (2 ε * guidance)

   The tangent ``v_pred`` is selected by
   ``anyflow_pretrain.cd_velocity_source``:

     - ``"noise_minus_x0"`` (default): ``v_pred = noise - latents``
       (original AnyFlow / MeanFlow; unbiased but high-variance).
     - ``"u_self"`` (iMF arXiv:2512.02012 §4.1, opt-in):
       ``v_pred = teacher(z_t, t, t, c).detach()``, a single no-grad
       forward through a FROZEN copy of the raw base transformer at the
       boundary ``r = t``. Low-variance estimate of the marginal velocity
       without self-bootstrap drift (teacher is frozen). The on-policy
       trainer reuses its existing ``real_score`` as this teacher.

4. Loss target uses the PDE-residual identity for flow maps:

       target = v_target - (t - r) * dF/dt

   ``v_target`` is selected by ``anyflow_pretrain.v_target_source`` —
   **independent of** ``cd_velocity_source``:

     - ``"noise_minus_x0"`` (default): ``v_target = noise - x_0``
       (unbiased data-line target; standard flow matching).
     - ``"u_self"``: ``v_target = teacher(z_t, t, t, c).detach()``
       (Option C teacher distillation; the student is trained to match
       the frozen base teacher's marginal velocity, with the AnyFlow
       PDE residual correction).

   When r = t this collapses to ``target = v_target`` (standard FM /
   distillation respectively). Combining the two knobs (cd_velocity_source
   × v_target_source) gives four meaningful modes; see
   ``AnyFlowPretrainConfig.v_target_source`` for the cross-table.

5. (Optional) Reverse-CFG fusion: when ``forward_guidance_scale > 1``, the
   model's cond prediction is rewritten as
   ``(noise_pred - (1 - G) * uncond_pred) / G`` so that inference can run
   without CFG (the model's cond forward learns the CFG-applied score).

6. Loss = weighted MSE(noise_pred, target), with per-sample timestep weights
   (gaussian / beta08 / uniform) and a scale_weight rebalance that brings
   non-FM samples to the same magnitude as FM samples.

Aligned with ``AnyFlow/far/trainers/trainer_wan_anyflow_pretrain.py`` for
the bidirectional case (we don't replicate FAR causal frame chunking; image
diffusion is bidirectional by definition).

Inherits all base infrastructure (FSDP/DDP wrap, EMA, checkpoint, RNG, eval) from
:class:`BaseTrainer`. Loads the flow-map two-time embedding wrapper into
the transformer in ``_pre_wrap_models()`` so the new ``delta_embedder``
parameters are sharded by FSDP correctly.

Notes on resume / FSDP / HSDP:
- All per-batch random quantities (the (t, r) mode partition, the random
  text-drop mask) are derived from ``self.global_step`` via a fresh local
  RNG so resume is exact and all ranks see the same partition pattern.
- The dataset's DistributedSampler uses ``set_epoch(epoch)`` per the base
  trainer's pattern.
- Flow-map wrapper is set up BEFORE FSDP/DDP wrapping (in ``_pre_wrap_models``)
  so the new ``delta_embedder`` is sharded correctly.
"""

from __future__ import annotations

import copy
import logging
import os
import random
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler

from meanflownft.config import MeanFlowNFTConfig
from meanflownft.data.latent_prompt_dataset import create_latent_prompt_dataloader
from meanflownft.models.sd35 import (
    encode_prompts_sd35,
    load_sd35_models,
)
from meanflownft.models.sd35_flowmap import (
    predict_noise_sd35_flowmap,
    setup_flowmap_for_sd3,
)
from meanflownft.parallel.utils import (
    fsdp_wrap_model,
    get_rank,
    get_transformer_wrap_policy,
    get_world_size,
)
from meanflownft.schedulers.flowmap_scheduler import FlowMapScheduler
from meanflownft.trainers.base_trainer import BaseTrainer
from meanflownft.utils.lora import setup_lora

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AnyFlowPretrainTrainer(BaseTrainer):
    """AnyFlow forward training stage.

    See the module docstring for the algorithm. Stage 1 only needs the
    trainable flow-map generator and an optional frozen boundary teacher.
    """

    def __init__(self, config: MeanFlowNFTConfig):
        super().__init__(config)
        self.model_type = str(self.config.model.model_type).lower()
        if self.model_type != "sd35":
            raise ValueError(
                "AnyFlowPretrainTrainer only supports model_type='sd35'; "
                f"got {self.config.model.model_type!r}."
            )

        # Validate flowmap config.
        fm = self.config.flowmap_model
        if not fm.enabled:
            raise ValueError(
                "AnyFlowPretrainTrainer requires flowmap_model.enabled=True."
            )

        # Validate three-mode ratios.
        ap = self.config.anyflow_pretrain
        if not (0.0 <= ap.diffusion_ratio):
            raise ValueError(f"diffusion_ratio must be >= 0, got {ap.diffusion_ratio}")
        if not (0.0 <= ap.consistency_ratio):
            raise ValueError(f"consistency_ratio must be >= 0, got {ap.consistency_ratio}")
        if ap.diffusion_ratio + ap.consistency_ratio > 1.0 + 1e-6:
            raise ValueError(
                f"diffusion_ratio + consistency_ratio must be <= 1, got "
                f"{ap.diffusion_ratio + ap.consistency_ratio}"
            )

        # Model components (populated in setup_models)
        self.generator: Optional[nn.Module] = None
        self.generator_ema: Optional[nn.Module] = None
        self.text_encoders: list[nn.Module] = []
        self.tokenizers: list = []
        self.scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None
        self.flowmap_scheduler: Optional[FlowMapScheduler] = None
        self.vae = None
        self.latent_channels: Optional[int] = None
        self.latent_size: Optional[int] = None

        # u_self teacher (frozen base transformer + flow-map wrap). Created
        # lazily by ``_pre_wrap_models`` when EITHER
        # ``cd_velocity_source == 'u_self'`` (iMF JVP tangent) OR
        # ``v_target_source == 'u_self'`` (Option-C distillation target);
        # one copy serves both knobs.
        # Subclasses (e.g. ``AnyFlowOnPolicyTrainer``) may override
        # ``_should_create_u_self_teacher`` to skip creation and reuse an
        # existing frozen base teacher (e.g. ``real_score``).
        self._u_self_teacher: Optional[nn.Module] = None

        # Cached unconditional embeddings for forward CFG fusion + text drop.
        # Lazy-loaded from anyflow_pretrain.negative_embedding_path or computed
        # from the empty-prompt the first time we need them.
        self._uncond_prompt_embeds: Optional[torch.Tensor] = None
        self._uncond_pooled_embeds: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # SD3.5 flow-map helpers
    # ------------------------------------------------------------------

    def _setup_flowmap_wrapper(self) -> None:
        fm = self.config.flowmap_model
        setup_flowmap_for_sd3(
            self.generator,
            gate_value=fm.gate_value,
            deltatime_type=fm.deltatime_type,
        )

    def _predict_noise_flowmap(
        self,
        model: nn.Module,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_embeds: torch.Tensor,
        r_timesteps: torch.Tensor,
        guidance_scale: float = 1.0,
        uncond_text_embeddings: Optional[torch.Tensor] = None,
        uncond_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict an SD3.5 flow-map velocity with optional CFG."""
        return predict_noise_sd35_flowmap(
            model=model,
            noisy_latents=noisy_latents,
            text_embeddings=prompt_embeds,
            timesteps=timesteps,
            pooled_prompt_embeds=pooled_embeds,
            r_timesteps=r_timesteps,
            guidance_scale=guidance_scale,
            uncond_text_embeddings=uncond_text_embeddings,
            uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds,
        )

    # ------------------------------------------------------------------
    # u_self teacher helpers (frozen base reference for iMF / distillation)
    #
    # The teacher is a FROZEN copy of the raw base transformer (flow-map
    # wrapped exactly like the generator, but never updated). It can serve
    # two INDEPENDENT purposes, each controlled by its own config knob:
    #
    #   1. CD tangent direction (``cd_velocity_source == "u_self"``):
    #      replace the original AnyFlow ``e - x`` JVP tangent with the
    #      teacher's marginal v prediction at ``r = t``. Lower variance
    #      than the conditional ``e - x``, no self-bootstrap drift (frozen).
    #      Motivated by iMF (arXiv:2512.02012 §4.1).
    #
    #   2. Regression target (``v_target_source == "u_self"``): replace
    #      the original AnyFlow ``noise - latents`` target with the same
    #      teacher boundary forward. Turns pretrain into TEACHER DISTILLATION
    #      + PDE residual (the student learns to match the teacher's
    #      marginal velocity field, with the AnyFlow PDE-residual correction).
    #
    # ``_should_create_u_self_teacher`` returns True if EITHER knob is
    # ``"u_self"``; one teacher copy serves both, and
    # :meth:`_compute_forward_training_loss` only runs the teacher forward
    # once per train_step regardless of how many knobs consume the result.
    #
    # The teacher uses the same flow-map wrap as the generator, but since
    # ``delta_embedder = timestep_embedder.deepcopy()`` at init AND the
    # teacher is frozen, at ``r = t`` the rt_emb formula collapses to
    # ``rt_emb = (1 - gate) * t_emb + gate * delta_emb == t_emb`` exactly,
    # so the teacher behaves as the raw base SD3.5 transformer.
    # ------------------------------------------------------------------

    def _should_create_u_self_teacher(self) -> bool:
        """Whether to allocate a frozen base teacher in ``_pre_wrap_models``.

        Returns True if EITHER the CD tangent (``cd_velocity_source``) or
        the regression target (``v_target_source``) is set to ``"u_self"``,
        since both knobs share the same frozen teacher. One transformer
        copy serves both; ``_compute_forward_training_loss`` only runs
        the teacher forward once per train_step regardless of how many
        knobs consume the result.

        Subclasses with an existing frozen base copy (e.g.
        :class:`AnyFlowOnPolicyTrainer` already creates ``real_score``)
        should override this to return False and override
        :meth:`_predict_v_base_at_boundary` to redirect to that copy.
        """
        ap = self.config.anyflow_pretrain
        return ap.cd_velocity_source == "u_self" or ap.v_target_source == "u_self"

    def _predict_v_base_at_boundary(
        self,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        t: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Single no-grad forward through the frozen base teacher at r=t.

        Returns the marginal conditional velocity ``v_marg(z, t, c)`` from
        the pretrained base, with autocast applied so the dtype matches
        the training-time precision. Subclasses may override to redirect
        to an alternative frozen reference (e.g. ``real_score``).
        """
        if self._u_self_teacher is None:
            raise RuntimeError(
                "_predict_v_base_at_boundary called but self._u_self_teacher "
                "is None. Did _pre_wrap_models / _should_create_u_self_teacher "
                "get overridden without overriding this method?"
            )
        with torch.no_grad(), self._autocast():
            v_base = self._predict_noise_flowmap(
                self._u_self_teacher,
                noisy_latents, prompt_embeds, t, pooled_embeds, t,  # r = t
                guidance_scale=1.0,
            )
        return v_base

    # ------------------------------------------------------------------
    # Model and scheduler setup
    # ------------------------------------------------------------------

    def setup_models(self) -> None:
        model_cfg = self.config.model
        dist_cfg = self.config.distributed

        logger.info("=" * 60)
        logger.info("Setting up AnyFlow Pretrain models")
        logger.info("=" * 60)

        components = load_sd35_models(model_cfg)
        base_transformer = components["transformer"]
        self.text_encoders = components["text_encoders"]
        self.tokenizers = components["tokenizers"]
        self.scheduler = components["scheduler"]
        self.vae = components["vae"]

        # Gradient checkpointing.
        if model_cfg.gradient_checkpointing:
            base_transformer.enable_gradient_checkpointing()

        # Generator = trainable copy of the pretrained transformer.
        self.generator = copy.deepcopy(base_transformer)
        # Cache the RAW base transformer (no flow-map wrapper, no LoRA) for
        # downstream subclasses. AnyFlow Stage 2 onpolicy needs raw teacher
        # weights for the on-policy real score and discriminator initialization,
        # mirroring AnyFlow's separate teacher checkpoint. Pretrain
        # itself doesn't use this; cached here so onpolicy override of
        # _pre_wrap_models can deepcopy a clean teacher without re-loading
        # SD3.5 from disk. Released at the end of setup_models.
        self._base_transformer_snapshot = base_transformer
        del base_transformer

        # Apply role-specific initialization (allows starting Stage 1 from a
        # custom checkpoint, e.g. an existing flow-map model).
        self._load_model_init_from_path(
            self.generator,
            model_cfg.generator_init_path,
            "generator",
        )

        # Install the flow-map (two-time) wrapper BEFORE setup_lora so that:
        #   (a) generator has `delta_embedder` when setup_lora's inject_lora
        #       calls load_state_dict(state_dict, strict=False) — Stage 1 LoRA
        #       checkpoints saved by base_trainer's incremental save contain
        #       `delta_embedder.*` keys (via the AnyFlow-aware
        #       _filter_lora_state_dict). Without this ordering, those keys are
        #       silently dropped and Stage 1's r-conditioning learning is lost.
        #   (b) setup_lora's "freeze all non-LoRA params" step uniformly covers
        #       delta_embedder; we re-enable it explicitly below so AnyFlow's
        #       r-dependent flow-map keeps training (full-FT this small MLP).
        # OnPolicy subclass overrides _pre_wrap_models to build real_score /
        # fake_score_net here from self._base_transformer_snapshot.
        self._pre_wrap_models()

        # LoRA (optional). For pretrain we usually do full FT.
        self.generator_lora_params = []
        if model_cfg.generator_lora.enabled:
            self.generator_lora_params = setup_lora(self.generator, model_cfg.generator_lora)
            # Re-unfreeze delta_embedder (setup_lora freezes everything except
            # LoRA A/B). AnyFlow's r-dependent flow-map relies on this small
            # MLP being trainable; without re-enabling, gate=0.25 mixing
            # degenerates to (1-0.25)*timestep_emb + 0.25*dead_delta_emb,
            # erasing the r-conditioning learning signal.
            for name, param in self.generator.named_parameters():
                if "delta_embedder" in name:
                    param.requires_grad = True
            logger.info(
                f"  Generator: LoRA injected (rank={model_cfg.generator_lora.rank}) "
                f"+ delta_embedder kept trainable"
            )
        else:
            self.generator.train()
            logger.info("  Generator: trainable copy created (full weight)")

        # Freeze text encoders + VAE.
        for enc in self.text_encoders:
            enc.requires_grad_(False)
            enc.eval()
        self.vae.requires_grad_(False)
        self.vae.eval()
        device = torch.device("cuda")
        self.vae.to(device)
        for enc in self.text_encoders:
            enc.to(device)

        # SD3.5 latent dimensions.
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_channels = self.generator.config.in_channels
        latent_divisor = vae_scale_factor
        if model_cfg.image_resolution % latent_divisor != 0:
            raise ValueError(
                f"image_resolution={model_cfg.image_resolution} must be divisible by "
                f"the SD3.5 VAE scale factor {latent_divisor}."
            )
        self.latent_size = model_cfg.image_resolution // latent_divisor

        # FlowMap scheduler uses SD3.5's static shift.
        sched_cfg = self.scheduler.config
        self.flowmap_scheduler = FlowMapScheduler(
            num_train_timesteps=int(sched_cfg.num_train_timesteps),
            shift=float(getattr(sched_cfg, "shift", 1.0)),
            weight_type=self.config.anyflow_pretrain.weight_type,
        )

        # Note: _pre_wrap_models() is now called earlier (before setup_lora)
        # so Stage 1 LoRA checkpoints can correctly load delta_embedder weights.

        # Release the cached raw base transformer; onpolicy subclass has
        # already used it inside the earlier _pre_wrap_models() call to build
        # real_score / fake_score_net.
        if hasattr(self, "_base_transformer_snapshot"):
            del self._base_transformer_snapshot

        # EMA generator (optional). MUST be deepcopy'd BEFORE FSDP wrap:
        # FSDP-wrapped modules hold references to ProcessGroup / DeviceMesh
        # objects (modules / C extensions) that pickle/deepcopy can't handle,
        # so deepcopy(generator) after FSDP raises
        # "TypeError: cannot pickle 'module' object". Deep-copy here, then
        # wrap both generator and EMA below.
        if self.config.train.use_ema:
            self.generator_ema = copy.deepcopy(self.generator)
            self.generator_ema.requires_grad_(False)
            self.generator_ema.eval()
            logger.info(
                f"  Generator EMA: created (decay={self.config.train.ema_decay}, "
                f"every={self.config.train.ema_every})"
            )

        # Wrap with FSDP / DDP (wraps generator + EMA if present).
        self._wrap_models(dist_cfg)

        self.models = {"generator": self.generator}
        if self.generator_ema is not None:
            self.models["generator_ema"] = self.generator_ema

        logger.info("AnyFlow Pretrain model setup complete")
        logger.info("=" * 60)

    def _pre_wrap_models(self) -> None:
        """Install flow-map two-time embedding wrapper. Runs after model
        creation but before FSDP/DDP wrap so the new ``delta_embedder`` is
        a normal child of the wrapped model.

        Also creates the frozen u_self teacher (used as the iMF JVP
        tangent source and/or the Option-C distillation target, depending
        on which of ``cd_velocity_source`` / ``v_target_source`` is set to
        ``"u_self"``). The teacher is a deepcopy of the RAW base
        transformer (cached at ``self._base_transformer_snapshot``), with
        its OWN flow-map wrap installed and then fully frozen.
        """
        self._setup_flowmap_wrapper()

        if self._should_create_u_self_teacher():
            if not hasattr(self, "_base_transformer_snapshot"):
                raise RuntimeError(
                    "u_self mode requires self._base_transformer_snapshot "
                    "(cached by setup_models). Did the setup_models ordering "
                    "change?"
                )
            base = self._base_transformer_snapshot
            self._u_self_teacher = copy.deepcopy(base)
            # Apply flow-map wrap so we can call _predict_noise_flowmap on
            # the teacher (same r/t API as the student). With
            # delta_embedder = timestep_embedder.deepcopy() at init AND the
            # teacher frozen, at r=t the rt_emb mixing collapses to t_emb
            # exactly, i.e. the teacher equals the raw SD3.5 forward.
            fm = self.config.flowmap_model
            setup_flowmap_for_sd3(
                self._u_self_teacher,
                gate_value=fm.gate_value,
                deltatime_type=fm.deltatime_type,
            )
            self._u_self_teacher.requires_grad_(False)
            self._u_self_teacher.eval()
            logger.info(
                "  u_self teacher: frozen base transformer (flow-map wrapped) created"
            )

    def _wrap_models(self, dist_cfg) -> None:
        strategy = dist_cfg.strategy
        fsdp_precision = self.config.model.dtype

        if strategy == "fsdp":
            try:
                from diffusers.models.transformers.transformer_sd3 import JointTransformerBlock
                wrap_policy = get_transformer_wrap_policy(JointTransformerBlock)
            except ImportError:
                wrap_policy = None
                logger.warning(
                    "Could not import transformer block class for FSDP wrap policy, "
                    "falling back to size-based wrapping"
                )
            self.generator = fsdp_wrap_model(
                self.generator,
                sharding_strategy=dist_cfg.fsdp_sharding,
                fsdp_precision=fsdp_precision,
                auto_wrap_policy=wrap_policy,
            )
            logger.info(
                f"  Generator wrapped with FSDP (sharding={dist_cfg.fsdp_sharding}, "
                f"fsdp_precision={fsdp_precision})"
            )
            if self.generator_ema is not None:
                self.generator_ema = fsdp_wrap_model(
                    self.generator_ema,
                    sharding_strategy=dist_cfg.fsdp_sharding,
                    fsdp_precision=fsdp_precision,
                    auto_wrap_policy=wrap_policy,
                )
                logger.info(
                    f"  Generator EMA wrapped with FSDP (sharding={dist_cfg.fsdp_sharding})"
                )
            if self._u_self_teacher is not None:
                self._u_self_teacher = fsdp_wrap_model(
                    self._u_self_teacher,
                    sharding_strategy=dist_cfg.fsdp_sharding,
                    fsdp_precision=fsdp_precision,
                    auto_wrap_policy=wrap_policy,
                )
                logger.info(
                    f"  u_self teacher wrapped with FSDP (sharding={dist_cfg.fsdp_sharding})"
                )
        elif strategy == "ddp":
            from meanflownft.parallel.utils import ddp_wrap_model
            device = torch.device("cuda")
            self.generator = self.generator.to(device)
            has_lora = self.config.model.generator_lora.enabled
            self.generator = ddp_wrap_model(
                self.generator, find_unused_parameters=has_lora,
            )
            logger.info("  Generator wrapped with DDP")
            # EMA stays as a regular nn.Module (no grad, no DDP needed); just
            # move it to the same GPU as the generator.
            if self.generator_ema is not None:
                self.generator_ema = self.generator_ema.to(device)
                logger.info("  Generator EMA moved to GPU (no DDP wrap)")
            if self._u_self_teacher is not None:
                # Frozen — no DDP wrap needed, just move to GPU.
                self._u_self_teacher = self._u_self_teacher.to(device)
                logger.info("  u_self teacher moved to GPU (no DDP wrap, frozen)")
        else:
            raise ValueError(f"Unknown distributed strategy: {strategy}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

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
            f"params={sum(p.numel() for p in gen_params) / 1e6:.1f} M)"
        )

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------

    def _create_dataloader(self):
        """Use pre-generated SD3.5 latent data."""
        return create_latent_prompt_dataloader(
            data_root=self.config.prompt_path,
            batch_size=self.config.train.batch_size,
            distributed=(get_world_size() > 1),
            seed=self.config.train.seed,
            load_pooled=True,
        )

    def _prepare_consistency_batch(
        self,
        batch: dict[str, Any],
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move a precomputed SD3.5 latent/embedding batch to the GPU."""
        required = {"latents", "prompt_embeds", "pooled_embeds"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(
                f"Consistency batch is missing {sorted(missing)}; "
                f"got keys {sorted(batch)}."
            )
        return (
            batch["latents"].to(device=device, dtype=model_dtype),
            batch["prompt_embeds"].to(device=device, dtype=model_dtype),
            batch["pooled_embeds"].to(device=device, dtype=model_dtype),
        )

    # ------------------------------------------------------------------
    # Init helpers (loading custom init checkpoints)
    # ------------------------------------------------------------------

    def _load_model_init_from_path(
        self, model: nn.Module, init_path: str, role_name: str,
    ) -> None:
        """Load an optional role-specific initialization checkpoint."""
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
        # diffusers transformer dir
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

    # ------------------------------------------------------------------
    # Negative embedding loader (for forward CFG fusion + text drop)
    # ------------------------------------------------------------------

    def _ensure_uncond_embeds(self, batch_dtype: torch.dtype, batch_device: torch.device) -> None:
        """Load or compute the empty-prompt embedding once and cache."""
        if self._uncond_prompt_embeds is not None:
            return
        path = self.config.anyflow_pretrain.negative_embedding_path or ""
        if path:
            data = torch.load(path, map_location="cpu", weights_only=False)
            self._uncond_prompt_embeds = data["prompt_embeds"].to(
                device=batch_device, dtype=batch_dtype,
            )
            self._uncond_pooled_embeds = data["pooled_embeds"].to(
                device=batch_device, dtype=batch_dtype,
            )
            logger.info(f"Loaded precomputed negative embeddings from {path}")
            return
        # Sequence length 256 matches SD3.5 data generation.
        uncond_emb, uncond_pool = encode_prompts_sd35(
            [""], self.text_encoders, self.tokenizers, batch_device,
            max_sequence_length=256,
        )
        self._uncond_prompt_embeds = uncond_emb.to(dtype=batch_dtype)
        self._uncond_pooled_embeds = uncond_pool.to(dtype=batch_dtype)
        logger.info("Computed negative embeddings on-the-fly (set anyflow_pretrain.negative_embedding_path to cache)")

    def _expand_uncond_to_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeat cached negative embeds to match the batch."""
        emb = self._uncond_prompt_embeds
        pool = self._uncond_pooled_embeds
        if emb.shape[0] != 1:
            emb = emb[:1]
            pool = pool[:1]
        if emb.dim() == 3:
            emb = emb.expand(batch_size, -1, -1).contiguous()
        elif emb.dim() == 2:
            emb = emb.expand(batch_size, -1).contiguous()
        if pool.dim() == 3:
            pool = pool.expand(batch_size, -1, -1).contiguous()
        elif pool.dim() == 2:
            pool = pool.expand(batch_size, -1).contiguous()
        elif pool.dim() == 1:
            pool = pool.expand(batch_size, -1).contiguous() if pool.shape == (batch_size,) else pool.unsqueeze(0).expand(batch_size, -1).contiguous()
        return emb, pool

    # ------------------------------------------------------------------
    # Three-mode timestep sampling (aligned with AnyFlow sample_timestep)
    # ------------------------------------------------------------------

    def _sample_three_mode_tr(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-rank sample of (t, r) using the three-mode partition.

        Aligned exactly with AnyFlow ``trainer_wan_anyflow_pretrain.sample_timestep``:

        - sample t1, t2 ~ U(0, 1) per local sample
        - t = max, r = min
        - first ``round(diffusion_ratio * global_bsz)`` global samples: r = t
        - next  ``round(consistency_ratio * global_bsz)`` global samples: r = 0
        - rest: r ~ U(0, t) (already from the min(t1, t2))

        This deterministic-by-global-index assignment means **no broadcast is
        needed** — every rank computes its own slice using the same global
        partition rule. That keeps FSDP collectives in lock-step.

        Returns:
            (t, r, is_diffusion) all in [0, 1]; ``is_diffusion`` is bool[B].
        """
        ap = self.config.anyflow_pretrain
        global_start_idx = get_rank() * batch_size
        global_bsz = get_world_size() * batch_size

        # Use a per-step seeded RNG so resume reproduces the same partition.
        # Note: t1, t2 are local-rank-specific (different ranks see different
        # noise), but the *partition* (which sample is diffusion / consistency
        # / generic) is a deterministic function of global_idx.
        gen = torch.Generator(device=device)
        gen.manual_seed(self.config.train.seed + self.global_step + get_rank() * 7919)
        t1 = torch.rand(batch_size, dtype=dtype, device=device, generator=gen)
        t2 = torch.rand(batch_size, dtype=dtype, device=device, generator=gen)
        t = torch.maximum(t1, t2)
        r = torch.minimum(t1, t2)

        n_diffusion = round(ap.diffusion_ratio * global_bsz)
        n_consistency = round(ap.consistency_ratio * global_bsz)
        is_diffusion = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for b in range(batch_size):
            g_idx = global_start_idx + b
            if g_idx < n_diffusion:
                r[b] = t[b]
                is_diffusion[b] = True
            elif g_idx < n_diffusion + n_consistency:
                r[b] = 0.0
            # else: keep r ~ U(0, t)
        return t, r, is_diffusion

    # ------------------------------------------------------------------
    # Central difference dF/dt (no-grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_central_difference(
        self,
        noisy_latents: torch.Tensor,
        latents: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        guidance: float,
        fwd_guidance: float = 1.0,
        v_pred_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate ``dF/dt`` along the ODE trajectory by central difference.

        ``fwd_guidance`` is the guidance scalar passed to the two SD3.5 model
        forwards. ``guidance`` is the reverse-CFG divisor.

        Aligned with AnyFlow ``compute_central_difference``:

            v_pred = noise - latents                       (noise_minus_x0)
                   = teacher(z_t, t, t, c).detach()        (u_self; caller passes
                                                            it via v_pred_override)
            t± = t ± epsilon
            z_t± = z_t ± v_pred * (epsilon / num_train_timesteps)
            return (F(z_+, t_+, r) - F(z_-, t_-, r)) / (2 * eps * guidance)

        ``v_pred`` is selected by the caller via ``v_pred_override``:

          - ``None`` (caller uses ``cd_velocity_source == "noise_minus_x0"``):
            the original AnyFlow tangent ``e - x``, an unbiased but
            high-variance estimate of the marginal velocity.
          - ``Tensor`` (caller uses ``cd_velocity_source == "u_self"``):
            a pre-computed tangent — the frozen u_self teacher's marginal v
            prediction at the boundary ``r = t``. Lower variance than
            ``e - x`` (iMF arXiv:2512.02012 §4.1) and frozen so there's
            no self-bootstrap drift. Caller obtains it from
            :meth:`_predict_v_base_at_boundary`.

        NOTE: this function only controls the CD **tangent direction**;
        the regression ``v_target`` is decided independently by the caller
        (``v_target_source``, see :meth:`_compute_forward_training_loss`).
        The two knobs are orthogonal — you can mix-and-match e.g.
        teacher CD tangent + data-line v_target.

        The ``/ guidance`` at the end converts the central diff of the
        student's cond head ``f_c`` (which under reverse-CFG fusion
        equals ``G * g + (1-G) * f_u``) into the central diff of the
        fused quantity ``g``, since ``df_c/dt ≈ G * dg/dt`` when ``f_u``
        is slowly varying. This is required for the PDE-residual target
        ``g = v_target - (t-r) * dg/dt`` to make sense.
        """
        ap = self.config.anyflow_pretrain
        T = float(self.scheduler.config.num_train_timesteps)
        eps = float(ap.epsilon)

        if v_pred_override is not None:
            v_pred = v_pred_override.to(latents.dtype)
        elif ap.cd_velocity_source == "noise_minus_x0":
            v_pred = noise - latents
        else:
            # u_self should reach this function with v_pred_override set by
            # the caller (_compute_forward_training_loss). If we got here
            # without an override, it's a programming bug.
            raise ValueError(
                f"_compute_central_difference: anyflow_pretrain.cd_velocity_source="
                f"{ap.cd_velocity_source!r} expects caller to pass v_pred_override "
                f"(teacher boundary velocity); got None."
            )

        t_plus = t + eps
        z_plus = noisy_latents + v_pred * (eps / T)
        F_plus = self._predict_noise_flowmap(
            self.generator, z_plus, prompt_embeds, t_plus, pooled_embeds, r,
            guidance_scale=fwd_guidance,
        )

        t_minus = t - eps
        z_minus = noisy_latents - v_pred * (eps / T)
        F_minus = self._predict_noise_flowmap(
            self.generator, z_minus, prompt_embeds, t_minus, pooled_embeds, r,
            guidance_scale=fwd_guidance,
        )

        return (F_plus - F_minus) / (2.0 * eps * guidance)

    # ------------------------------------------------------------------
    # Forward training loss (aligned with AnyFlow train_bidirection)
    # ------------------------------------------------------------------

    def _compute_forward_training_loss(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the forward-training PDE-residual loss.

        See module docstring for the full formula. Returns scalar mean loss
        plus an info dict for logging.
        """
        ap = self.config.anyflow_pretrain
        batch_size = latents.shape[0]
        device = latents.device
        dtype = latents.dtype
        T = float(self.scheduler.config.num_train_timesteps)

        # 1. Sample (t, r) per the three-mode partition, then apply SD3.5's
        # static timestep shift to both endpoints.
        t01, r01, is_diffusion = self._sample_three_mode_tr(batch_size, dtype, device)
        t = (self.flowmap_scheduler.apply_shift(t01) * T).to(device)
        r = (self.flowmap_scheduler.apply_shift(r01) * T).to(device)

        # 2. Forward-noise to t.
        noise = torch.randn_like(latents)
        noisy_latents = self.flowmap_scheduler.scale_noise(latents, t, noise)

        # 3. Forward + optional reverse-CFG fusion.
        guidance = float(ap.forward_guidance_scale)
        with self._autocast():
            noise_pred = self._predict_noise_flowmap(
                self.generator, noisy_latents, prompt_embeds, t, pooled_embeds, r,
                guidance_scale=1.0,
            )
        if guidance > 1.0:
            self._ensure_uncond_embeds(prompt_embeds.dtype, prompt_embeds.device)
            uncond_emb, uncond_pool = self._expand_uncond_to_batch(batch_size)
            with torch.no_grad():
                with self._autocast():
                    noise_pred_uncond = self._predict_noise_flowmap(
                        self.generator, noisy_latents, uncond_emb, t, uncond_pool, r,
                        guidance_scale=1.0,
                    )
            noise_pred = (noise_pred - (1.0 - guidance) * noise_pred_uncond) / guidance

        # 4. Decide the CD tangent (v_pred) and the regression target (v_target).
        # These are TWO INDEPENDENT knobs:
        #   ap.cd_velocity_source ∈ {"noise_minus_x0", "u_self"}  →  v_pred
        #   ap.v_target_source    ∈ {"noise_minus_x0", "u_self"}  →  v_target
        # If either is "u_self", we call the frozen base teacher ONCE at r=t
        # and reuse the result. See AnyFlowPretrainConfig docstrings for the
        # four combinations and their semantics.
        cd_uses_teacher = ap.cd_velocity_source == "u_self"
        tgt_uses_teacher = ap.v_target_source == "u_self"
        v_base = None
        if cd_uses_teacher or tgt_uses_teacher:
            v_base = self._predict_v_base_at_boundary(
                noisy_latents, prompt_embeds, t, pooled_embeds,
            ).to(latents.dtype)

        v_pred_override = v_base if cd_uses_teacher else None
        v_target = v_base if tgt_uses_teacher else (noise - latents)

        # 5. Central-difference dF/dt (no-grad).
        # Use the same CFG-free SD3.5 forward and reverse-CFG divisor.
        dF_dt = self._compute_central_difference(
            noisy_latents=noisy_latents, latents=latents, noise=noise,
            t=t, r=r, prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds,
            guidance=guidance,
            fwd_guidance=1.0,
            v_pred_override=v_pred_override,
        )

        # 6. PDE-residual target.
        # broadcast (t - r) over the spatial dims of the latent tensor
        diff = (t - r).view(batch_size, *([1] * (latents.ndim - 1)))
        target = v_target - diff * dF_dt

        # 7. Per-sample MSE then weighted mean.
        per_sample = (noise_pred.float() - target.float()) ** 2
        per_sample = per_sample.reshape(batch_size, -1).mean(dim=-1)
        weights = self.flowmap_scheduler.get_train_weight(t).to(device).flatten()
        weighted = per_sample * weights

        # 8. Scale-weight rebalance: bring non-diffusion samples to the same
        # global magnitude as diffusion samples (aligned with AnyFlow Wan
        # `trainer_wan_anyflow_pretrain.py:314-319` exactly):
        #   scale_weight = global_diff_mean / (local_nondiff_weighted + 1e-5)
        # The numerator and denominator both use the WEIGHTED loss, and the
        # scale is a per-sample vector (one factor per non-diffusion sample),
        # not a single scalar. Each non-diffusion sample is normalized to the
        # global diffusion mean, equalizing per-mode contribution to the batch
        # gradient regardless of intrinsic loss magnitude.
        scale = None
        with torch.no_grad():
            if dist.is_initialized() and get_world_size() > 1:
                # Every rank must enter these collectives, including ranks whose
                # local partition is homogeneous.
                global_weighted = torch.cat(
                    dist.nn.all_gather(weighted), dim=0
                )
                global_mask = torch.cat(
                    dist.nn.all_gather(is_diffusion), dim=0
                )
            else:
                global_weighted = weighted
                global_mask = is_diffusion

            if (
                global_mask.any()
                and (~global_mask).any()
                and (~is_diffusion).any()
            ):
                diff_mean = global_weighted[global_mask].mean()
                # Only local non-diffusion samples need a scale; homogeneous
                # diffusion ranks still participated in the global gathers.
                local_nondiff_weighted = weighted[~is_diffusion]
                scale = diff_mean / (local_nondiff_weighted + 1e-5)
        if scale is not None:
            weighted = weighted.clone()
            weighted[~is_diffusion] = weighted[~is_diffusion] * scale

        loss = weighted.mean()

        # Three-mode partition is deterministic by GLOBAL index (matches AnyFlow
        # Wan original), so a single rank's batch is usually homogeneous
        # (all-diffusion / all-consistency / all-generic depending on which
        # global slice the rank holds). All-reduce the partition counts so the
        # logged metric reflects the WORLD-level distribution (which is the
        # one configured by diffusion_ratio / consistency_ratio).
        n_d = is_diffusion.sum()
        n_c = ((r == 0) & (~is_diffusion)).sum()
        n_g = ((r > 0) & (r < t)).sum()
        if dist.is_initialized() and get_world_size() > 1:
            counts = torch.stack([n_d, n_c, n_g]).to(device=device, dtype=torch.long)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            n_d, n_c, n_g = counts[0], counts[1], counts[2]

        info = {
            "loss_forward": float(loss.detach()),
            "n_diffusion": int(n_d.item()),
            "n_consistency": int(n_c.item()),
            "n_generic": int(n_g.item()),
            "t_mean": float(t.mean().item()),
            "r_mean": float(r.mean().item()),
            "weight_mean": float(weights.mean().item()),
        }
        return loss, info

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict[str, Any]) -> dict[str, dict[str, float]]:
        """One AnyFlow forward training iteration."""
        ap = self.config.anyflow_pretrain
        device = torch.device("cuda")
        model_dtype = next(self.generator.parameters()).dtype

        latents, prompt_embeds, pooled_embeds = self._prepare_consistency_batch(
            batch, device, model_dtype,
        )

        # Random text-drop (CFG dropout): replace prompts with the negative
        # embedding with probability drop_text_ratio. Use a per-step seeded
        # local RNG so resume reproduces the mask exactly.
        if ap.drop_text_ratio > 0.0:
            self._ensure_uncond_embeds(prompt_embeds.dtype, prompt_embeds.device)
            # Per-rank seed (added get_rank()) so each rank gets an
            # independent CFG-dropout mask, matching AnyFlow Wan (which uses
            # unseeded torch.rand → independent across ranks). Without the
            # rank term, all ranks would drop the SAME global-batch positions,
            # which biases the CFG-dropout statistics under FSDP.
            gen = torch.Generator(device=device)
            gen.manual_seed(
                self.config.train.seed + self.global_step * 31337 + get_rank() * 7919
            )
            mask = torch.rand(latents.shape[0], device=device, generator=gen) < ap.drop_text_ratio
            if mask.any():
                uncond_emb, uncond_pool = self._expand_uncond_to_batch(latents.shape[0])
                prompt_embeds = prompt_embeds.clone()
                pooled_embeds = pooled_embeds.clone()
                prompt_embeds[mask] = uncond_emb.to(prompt_embeds.dtype)[mask]
                pooled_embeds[mask] = uncond_pool.to(pooled_embeds.dtype)[mask]

        self.generator.train()

        with self.timer.measure("forward_training"):
            loss, info = self._compute_forward_training_loss(
                latents, prompt_embeds, pooled_embeds,
            )

        self.optimizers["generator"].zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._compute_grad_norm(self.generator)
        if self.config.solver.generator.max_grad_norm > 0:
            self._clip_grad_norm(self.generator, self.config.solver.generator.max_grad_norm)
        self.optimizers["generator"].step()
        self.schedulers["generator"].step()

        # EMA with optional warmup (AnyFlow Wan style):
        # During warmup (global_step < ema_warmup_steps), use decay=0 so the
        # EMA buffer is replaced by the current source weights every update.
        # This makes EMA cold-start at the end of warmup and avoids dragging
        # in noisy early-training weights. After warmup, switch to the fixed
        # ``train.ema_decay``. This mirrors AnyFlow's ``ShardEMA.get_decay``.
        if (
            self.generator_ema is not None
            and (self.global_step + 1) % self.config.train.ema_every == 0
        ):
            warmup = int(self.config.anyflow_pretrain.ema_warmup_steps)
            decay = (
                0.0 if self.global_step < warmup else self.config.train.ema_decay
            )
            self.ema_update(
                src_model=self.generator,
                tgt_model=self.generator_ema,
                decay=decay,
            )

        info.update({
            "lr": self.schedulers["generator"].get_last_lr()[0],
            "grad_norm": grad_norm,
        })
        return {"forward": info}

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_grad_norm(model: nn.Module, max_norm: float) -> float:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(model, FSDP):
            norm = model.clip_grad_norm_(max_norm)
            return float(norm.item() if isinstance(norm, torch.Tensor) else norm)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return float(norm.item() if isinstance(norm, torch.Tensor) else norm)

    @staticmethod
    def _compute_grad_norm(model: nn.Module) -> float:
        return AnyFlowPretrainTrainer._clip_grad_norm(model, float("inf"))

    # ------------------------------------------------------------------
    # Eval helpers required by BaseTrainer._evaluate
    #
    # BaseTrainer._evaluate's local _batch_generate hard-calls three hooks:
    #     prompt_embeds, pooled_embeds = self._encode_prompts(prompts)
    #     uncond_embeds, uncond_pooled = self._get_uncond_embeds(N)
    #     latents, _ = self._generate_latents(prompt_embeds, pooled_embeds,
    #                                          uncond_embeds, uncond_pooled,
    #                                          num_steps=..., model=...)
    #     images = self._decode_latents_to_tensor(latents)
    # The hooks below implement the SD3.5 AnyFlow rollout path.
    # ------------------------------------------------------------------

    def _encode_prompts(
        self, prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode evaluation prompts with the SD3.5 text encoders."""
        device = torch.device("cuda")
        model_dtype = next(self.generator.parameters()).dtype
        prompt_embeds, pooled_embeds = encode_prompts_sd35(
            prompts, self.text_encoders, self.tokenizers, device,
            max_sequence_length=256,
        )
        prompt_embeds = prompt_embeds.to(dtype=model_dtype)
        pooled_embeds = pooled_embeds.to(dtype=model_dtype)
        return prompt_embeds, pooled_embeds

    def _get_uncond_embeds(
        self, batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached empty-prompt embeddings expanded to ``batch_size``."""
        device = torch.device("cuda")
        model_dtype = next(self.generator.parameters()).dtype
        self._ensure_uncond_embeds(model_dtype, device)
        return self._expand_uncond_to_batch(batch_size)

    def _generate_latents(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,  # noqa: ARG002 - CFG-free flow-map inference (cfg=1.0)
        uncond_pooled: torch.Tensor,  # noqa: ARG002 - same as above
        num_steps: int = 4,
        gradient_truncation: bool = False,  # noqa: ARG002 - eval path is no_grad
        random_stop: bool = False,  # noqa: ARG002 - eval uses fixed num_steps
        model: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, int]:
        """Generate latents via simple flow-map Euler rollout (one transformer
        forward per step). Used by BaseTrainer._evaluate; AnyFlow inference
        is CFG-free (guidance_scale=1.0) because reverse-CFG fusion already
        baked the CFG signal into the model during training.

        ``uncond_embeds`` / ``uncond_pooled`` / ``gradient_truncation`` /
        ``random_stop`` are accepted for the shared evaluation interface.
        """
        if model is None:
            model = self.generator_ema or self.generator

        device = prompt_embeds.device
        model_dtype = next(model.parameters()).dtype
        batch_size = prompt_embeds.shape[0]

        # Build the SD3.5 flow-map timestep grid with its static shift.
        sched_cfg = self.scheduler.config
        sched = FlowMapScheduler(
            num_train_timesteps=int(sched_cfg.num_train_timesteps),
            shift=float(getattr(sched_cfg, "shift", 1.0)),
            weight_type="uniform",
        )
        sched.set_timesteps(num_steps, device=device)
        timesteps = sched.timesteps  # [N+1]

        # Pure-noise init.
        x_t = torch.randn(
            batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=device, dtype=model_dtype,
        )

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for i in range(num_steps):
                    t = timesteps[i].expand(batch_size).to(device=device, dtype=torch.float32)
                    r = timesteps[i + 1].expand(batch_size).to(device=device, dtype=torch.float32)
                    with self._autocast():
                        v = self._predict_noise_flowmap(
                            model, x_t, prompt_embeds, t, pooled_embeds, r,
                            guidance_scale=1.0,
                        )
                    x_t = sched.step(v, x_t, t, r)
        finally:
            if was_training:
                model.train()

        return x_t, num_steps
