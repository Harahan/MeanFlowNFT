"""
AnyFlow Stage 2 (On-Policy DMD with Flow Map Backward Simulation) Trainer.

Implements the on-policy distillation stage of the AnyFlow paper
(https://arxiv.org/abs/2605.13724) inside the MeanFlowNFT framework. Built on
top of :class:`AnyFlowPretrainTrainer` so we inherit the flow-map model
wrapper, central-difference forward training (used as ``cotrain_forward_kl``),
and SD3.5 flow-map helpers.

Algorithm summary (per train_step):

1. **Generator update**:
     - Random ``sample_step ~ choice(num_inference_steps_list)`` (shared
       across ranks via broadcast).
     - Random ``grad_timestep ~ U[0, sample_step)`` (shared across ranks).
     - Run :func:`_shortcut_rollout` through the **student** transformer:
       3 flow-map jumps (prev / current / post) totaling 3 transformer
       forwards regardless of ``sample_step``. This is the "Flow Map
       Backward Simulation" trick — much cheaper than the N forwards a
       standard rollout would need, and the gradient still flows through
       all three jumps.
     - DMD loss on the rollout output: gradient = ``(pred_fake - pred_real) / |p_real|.mean()``
       with the real and fake scores computed by jumping the noisy rollout
       latent to ``z_0`` via the flow-map step (r=0).
     - (Optional) ``cotrain_forward_kl`` adds the Stage 1 forward training
       loss on the same prompt's GT latents as a stabilizing anchor.

2. **Discriminator (fake score) update**:
     - Run rollout no_grad to produce fake video latents.
     - Standard flow-matching loss with logit-normal timestep sampling.
     - Discriminator forwards use ``r_timestep = t`` (degenerate flow map,
       i.e. instantaneous velocity), aligned with AnyFlow.

Resume / multi-rank correctness:
- ``sample_step`` and ``grad_timestep`` are per-batch random integers that
  must be the same on every rank to keep FSDP collectives in sync. We use
  ``dist.broadcast(src=0)``, mirroring AnyFlow.
- All other random quantities (logit-normal t for discriminator, etc.) are
  per-rank independent (they don't trigger asymmetric collectives).
- ``self.global_step`` is the FT step counter (one ``train_step`` = one
  generator update + ``discriminator_update_ratio`` discriminator updates).

FSDP/DDP wrapping reuses the parent's helpers; the only addition is the
discriminator (= fake score net), which is also
flow-map-wrapped so it can use the flow-map ``step(t -> r=0)`` jump in DMD.

Note on iMF (arXiv:2512.02012, "Improved Mean Flows"):
    iMF's contribution is to replace the conditional velocity ``e - x``
    used as the central-difference tangent direction with the network's
    own reverse-CFG-fused boundary prediction
    ``(u_c - (1 - G) * sg(u_uncond)) / G`` at ``r = t`` (§4.1
    boundary-condition variant; we use the fused form to match the loss-
    side reverse-CFG fusion so the tangent corresponds to the marginal
    velocity the fused model traces). This applies wherever a flow-map
    ``dF/dt`` is estimated. Apart from the optional ``cotrain_forward_kl``
    path (which IS a Stage-1 forward-training loss and DOES respect
    ``anyflow_pretrain.cd_velocity_source``), this on-policy trainer does
    NOT estimate ``dF/dt``: the student is updated via shortcut rollout +
    DMD KL gradient (or CA+DM x0-distill), and the discriminator
    (fake score net) is trained with a standard flow-matching v-loss
    (``target = noise - latents`` at ``r = t``, no central diff).
    Therefore there is no on-policy-specific ``cd_velocity_source`` knob —
    set the iMF variant via ``anyflow_pretrain.cd_velocity_source`` and
    it will be picked up by the cotrain forward-training path automatically.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import compute_density_for_timestep_sampling

from meanflownft.config import MeanFlowNFTConfig
from meanflownft.models.sd35_flowmap import setup_flowmap_for_sd3
from meanflownft.parallel.utils import (
    fsdp_wrap_model,
    get_rank,
    get_transformer_wrap_policy,
    get_world_size,
)
from meanflownft.trainers.anyflow_pretrain_trainer import AnyFlowPretrainTrainer
from meanflownft.utils.lora import setup_lora

logger = logging.getLogger(__name__)


class AnyFlowOnPolicyTrainer(AnyFlowPretrainTrainer):
    """AnyFlow on-policy DMD distillation stage.

    Adds two extra models on top of the pretrain trainer:

    - ``self.real_score`` (alias: teacher): frozen pretrained flow-map model,
      used as the DMD "real score". Initialize from a Stage 1 checkpoint.
    - ``self.fake_score_net`` (alias: discriminator): online-updated
      flow-map model, learns the student's current distribution. Initialize
      from the same Stage 1 checkpoint as the generator.
    """

    def __init__(self, config: MeanFlowNFTConfig):
        super().__init__(config)
        self.real_score: Optional[nn.Module] = None
        self.fake_score_net: Optional[nn.Module] = None
        # FT (discriminator) step counter; resets implicitly on resume since
        # we derive it from global_step.
        # generator step counter is global_step itself (1 train_step = 1 gen
        # step + N discriminator steps).

        # Log the active rollout mode once. The full-step <-> detach dependency
        # is hard-enforced by AnyFlowOnPolicyConfig.__post_init__ (raises if
        # rollout_full_steps is on without rollout_detach_between_jumps).
        oc = self.config.anyflow_onpolicy
        if get_rank() == 0:
            logger.info(
                "[AnyFlowOnPolicy] rollout=%s, detach_between_jumps=%s",
                "full-step (all scheduler steps)" if oc.rollout_full_steps
                else "3-jump shortcut",
                oc.rollout_detach_between_jumps,
            )

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def setup_models(self) -> None:
        # Reuse the pretrain trainer's setup_models almost as-is — it builds
        # generator + flow-map wrap. We then add real_score + fake_score_net
        # in _pre_wrap_models (overridden below) so they get the same wrapper
        # AND get FSDP-sharded together with the generator.
        super().setup_models()

        # After super().setup_models() returns, self.generator is FSDP/DDP-wrapped
        # and self.real_score / self.fake_score_net have been attached.
        # Register them in self.models for checkpointing.
        if self.real_score is not None:
            # Real score is frozen; we don't save it (init_path-driven instead),
            # but exposing in self.models lets eval print sanity info.
            pass
        if self.fake_score_net is not None:
            self.models["fake_score_net"] = self.fake_score_net

    # ------------------------------------------------------------------
    # u_self teacher override: reuse real_score (no extra frozen copy).
    #
    # The on-policy trainer already creates ``real_score`` (a frozen
    # deepcopy of the raw base + flow-map wrap) for the DMD real-score
    # path. That same model is structurally identical to what the parent
    # ``AnyFlowPretrainTrainer`` would create as ``_u_self_teacher``, so
    # we skip the redundant allocation and redirect the boundary forward
    # to ``real_score``.
    # ------------------------------------------------------------------

    def _should_create_u_self_teacher(self) -> bool:
        # We already have self.real_score (frozen base + flow-map wrap);
        # reuse it instead of allocating a second copy.
        return False

    def _predict_v_base_at_boundary(
        self,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        t: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Boundary forward through ``real_score`` (frozen, never updated).

        Mirrors :meth:`AnyFlowPretrainTrainer._predict_v_base_at_boundary`
        but uses ``self.real_score`` (already a frozen base + flow-map
        wrap) so we don't double-allocate memory.
        """
        if self.real_score is None:
            raise RuntimeError(
                "AnyFlowOnPolicyTrainer._predict_v_base_at_boundary requires "
                "self.real_score (built in _pre_wrap_models); is u_self mode "
                "being called too early in setup?"
            )
        with torch.no_grad(), self._autocast():
            return self._predict_noise_flowmap(
                self.real_score,
                noisy_latents, prompt_embeds, t, pooled_embeds, t,  # r = t
                guidance_scale=1.0,
            )

    def _pre_wrap_models(self) -> None:
        """Build real_score + fake_score_net BEFORE the FSDP wrap call.

        Aligned with AnyFlow's `real_cfg.pretrained_path` (teacher checkpoint) /
        `discriminator_cfg.pretrained_path` (teacher checkpoint) separation:

          - real_score = deepcopy(raw base transformer) + flow-map wrapper,
            frozen. This is the DMD-defined "real distribution score" — for
            SD3.5's raw multi-step pretrained transformer is the teacher
            (no Stage 0 teacher training is needed).
            We do NOT inject LoRA here; real_score stays at raw weights.
            (Also reused as the u_self teacher when
            ``anyflow_pretrain.cd_velocity_source == "u_self"``.)

          - fake_score_net = deepcopy(raw base transformer) + flow-map wrapper
            + optional LoRA, trainable. Online-updated to track the current
            student output distribution. Initializing from raw base (rather
            than from the Stage 1 trained generator) gives the score model
            a clean starting point uninfluenced by Stage 1 LoRA, matching
            AnyFlow Wan's recipe of initializing the discriminator from the
            teacher checkpoint.

          - generator is already in place from the parent setup (raw base +
            Stage 1 LoRA chain via generator_lora.load_path + flow-map
            wrapper installed by super()._pre_wrap_models()).

        Resume safety: real_score / fake_score init don't depend on the
        current generator state, so resume reproduces identical models
        regardless of how much Stage 2 training has happened (generator is
        reloaded from Stage 2 ckpt by _maybe_resume AFTER this method runs,
        which is harmless since real_score / fake_score are already
        deepcopy'd from the snapshot).
        """
        # 1. Install flow-map wrapper on generator (parent behavior). After
        # this, self.generator has a delta_embedder (deepcopy'd from its
        # original timestep_embedder).
        super()._pre_wrap_models()

        model_cfg = self.config.model
        fm = self.config.flowmap_model

        # The raw base transformer is cached on the parent during setup_models.
        # If it's been released (shouldn't happen at this call site), fail
        # explicitly rather than silently fall back to deepcopy(generator),
        # which would re-introduce the Stage 1 student leakage bug.
        if not hasattr(self, "_base_transformer_snapshot"):
            raise RuntimeError(
                "AnyFlowOnPolicyTrainer._pre_wrap_models needs "
                "self._base_transformer_snapshot (cached by parent setup_models). "
                "Did the setup_models ordering change?"
            )
        base = self._base_transformer_snapshot

        # 2. Build real_score from RAW base.
        self.real_score = copy.deepcopy(base)
        self._load_model_init_from_path(
            self.real_score, model_cfg.teacher_init_path, "real_score",
        )
        # Install flow-map wrapper so _predict_noise_flowmap can call against
        # real_score with an r_timestep argument. The wrapper's delta_embedder
        # is a fresh deepcopy of real_score's own timestep_embedder, so at
        # r=t fallback the forward equals standard SD3 forward up to the
        # (1-gate)*timestep_emb + gate*delta_emb mix — and since
        # delta_embedder == timestep_embedder.deepcopy() at init, the
        # residual is effectively zero. real_score therefore behaves as the
        # raw multi-step teacher, which is what DMD expects.
        setup_flowmap_for_sd3(
            self.real_score, gate_value=fm.gate_value,
            deltatime_type=fm.deltatime_type,
        )
        # Fully frozen — DMD's real_score must not change during training.
        self.real_score.requires_grad_(False)
        self.real_score.eval()

        # 3. Build fake_score_net from RAW base (not from generator).
        self.fake_score_net = copy.deepcopy(base)
        self._load_model_init_from_path(
            self.fake_score_net, model_cfg.fake_score_init_path, "fake_score_net",
        )
        setup_flowmap_for_sd3(
            self.fake_score_net, gate_value=fm.gate_value,
            deltatime_type=fm.deltatime_type,
        )
        if model_cfg.fake_score_lora.enabled:
            setup_lora(self.fake_score_net, model_cfg.fake_score_lora)
            # Re-unfreeze delta_embedder (same rationale as generator).
            for name, param in self.fake_score_net.named_parameters():
                if "delta_embedder" in name:
                    param.requires_grad = True
            logger.info(
                f"  Fake score: LoRA injected (rank={model_cfg.fake_score_lora.rank}) "
                f"+ delta_embedder kept trainable"
            )
        else:
            self.fake_score_net.train()
            logger.info("  Fake score: trainable copy created (full weight)")

        logger.info(
            "  AnyFlow OnPolicy: built real_score (raw teacher, frozen) + "
            "fake_score_net (raw base + LoRA, trainable), both flow-map-wrapped."
        )

    def _wrap_models(self, dist_cfg) -> None:
        """Wrap generator (already done by parent), then also wrap
        real_score + fake_score_net with the same strategy.
        """
        # Parent wraps self.generator.
        super()._wrap_models(dist_cfg)

        strategy = dist_cfg.strategy
        fsdp_precision = self.config.model.dtype

        if strategy == "fsdp":
            try:
                from diffusers.models.transformers.transformer_sd3 import JointTransformerBlock
                wrap_policy = get_transformer_wrap_policy(JointTransformerBlock)
            except ImportError:
                wrap_policy = None
            self.real_score = fsdp_wrap_model(
                self.real_score,
                sharding_strategy=dist_cfg.fsdp_sharding,
                fsdp_precision=fsdp_precision,
                auto_wrap_policy=wrap_policy,
            )
            self.fake_score_net = fsdp_wrap_model(
                self.fake_score_net,
                sharding_strategy=dist_cfg.fsdp_sharding,
                fsdp_precision=fsdp_precision,
                auto_wrap_policy=wrap_policy,
            )
        elif strategy == "ddp":
            from meanflownft.parallel.utils import ddp_wrap_model
            device = torch.device("cuda")
            self.real_score = self.real_score.to(device)
            self.fake_score_net = self.fake_score_net.to(device)
            has_lora = self.config.model.fake_score_lora.enabled
            self.fake_score_net = ddp_wrap_model(
                self.fake_score_net, find_unused_parameters=has_lora,
            )
            # real_score is frozen; no DDP wrap needed.
        else:
            raise ValueError(f"Unknown distributed strategy: {strategy}")

    # ------------------------------------------------------------------
    # Optimizer (generator + discriminator)
    # ------------------------------------------------------------------

    def setup_optimizers(self) -> None:
        # Parent creates generator optimizer.
        super().setup_optimizers()

        fake_cfg = self.config.solver.fake_score
        fake_params = [p for p in self.fake_score_net.parameters() if p.requires_grad]
        self.optimizers["fake_score"] = torch.optim.AdamW(
            fake_params,
            lr=fake_cfg.lr,
            betas=(fake_cfg.beta1, fake_cfg.beta2),
            eps=fake_cfg.eps,
            weight_decay=fake_cfg.weight_decay,
        )
        self.schedulers["fake_score"] = self.create_warmup_constant_scheduler(
            self.optimizers["fake_score"], fake_cfg.warmup_steps,
        )
        logger.info(
            f"Discriminator (fake_score) optimizer created "
            f"(lr={fake_cfg.lr}, params={sum(p.numel() for p in fake_params) / 1e6:.1f} M)"
        )

    # ------------------------------------------------------------------
    # Broadcast helpers (sample_step / grad_timestep must be rank-consistent)
    # ------------------------------------------------------------------

    def _sample_int_synced(self, low: int, high: int) -> int:
        """Sample a single int in [low, high) on rank 0 and broadcast."""
        device = torch.device("cuda")
        val = torch.tensor([random.randrange(low, high)], dtype=torch.long, device=device)
        if dist.is_initialized() and get_world_size() > 1:
            dist.broadcast(val, src=0)
        return int(val.item())

    def _choice_int_synced(self, choices: list[int]) -> int:
        device = torch.device("cuda")
        val = torch.tensor([random.choice(choices)], dtype=torch.long, device=device)
        if dist.is_initialized() and get_world_size() > 1:
            dist.broadcast(val, src=0)
        return int(val.item())

    # ------------------------------------------------------------------
    # Flow Map Backward Simulation (3-segment shortcut rollout)
    # ------------------------------------------------------------------

    def _shortcut_rollout(
        self,
        init_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        num_inference_steps: int,
        grad_timestep: int,
        guidance_scale: float = 1.0,
        with_grad: bool = True,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """3-segment shortcut rollout aligned with AnyFlow ``training_rollout``.

        Splits the standard ``num_inference_steps``-step Euler rollout into
        three flow-map jumps:

            prev:    timesteps[0]              -> timesteps[grad_timestep]
            current: timesteps[grad_timestep]  -> timesteps[grad_timestep+1]
            post:    timesteps[grad_timestep+1] -> timesteps[-1] (= 0)

        Each jump is a single transformer forward, so the total cost is
        always 3 forwards regardless of ``num_inference_steps``. When
        ``with_grad=True`` the grad flows through all three jumps; when
        False the entire rollout is no-grad.
        """
        if model is None:
            model = self.generator

        # Build the timestep grid with SD3.5's static shift.
        sched = self.flowmap_scheduler
        sched.set_timesteps(num_inference_steps, device=init_latents.device)
        timesteps = sched.timesteps  # length N+1

        if grad_timestep < 0 or grad_timestep >= num_inference_steps:
            raise ValueError(
                f"grad_timestep={grad_timestep} out of [0, {num_inference_steps})"
            )

        batch_size = init_latents.shape[0]

        oc = self.config.anyflow_onpolicy
        # Feature 1 (rollout_detach_between_jumps): when on (and this is a grad
        # rollout), detach ONLY the velocity/model-forward input of each jump.
        # The scheduler step below keeps the LIVE running latent ``x``, so the
        # rollout accumulates additively as
        #     x = noise - sum_i (t_i - r_i)/T * v_i ,  v_i = F(x_{i-1}.detach(); θ)
        # Every v_i therefore stays in the final graph through its OWN additive
        # term (so the final DMD / CA-DM loss trains θ via every jump), but v_i is
        # computed from a detached x_{i-1} so it does NOT backprop through its
        # input into the earlier jumps. No-op when with_grad is False (the
        # discriminator rollout is already fully no_grad).
        detach_model_input = bool(oc.rollout_detach_between_jumps) and with_grad

        def _one_jump(x: torch.Tensor, t_scalar: torch.Tensor, r_scalar: torch.Tensor) -> torch.Tensor:
            """Single (t, r) flow-map jump on a batch."""
            if t_scalar.item() == r_scalar.item():
                # Degenerate (zero-width) segment — skip.
                return x
            t = t_scalar.expand(batch_size).to(device=x.device, dtype=torch.float32)
            r = r_scalar.expand(batch_size).to(device=x.device, dtype=torch.float32)
            # Detach ONLY the velocity input (Feature 1). The scheduler step uses
            # the live ``x`` so the additive running sum keeps every previous v_j.
            model_in = x.detach() if detach_model_input else x
            with self._autocast():
                v = self._predict_noise_flowmap(
                    model, model_in, prompt_embeds, t, pooled_embeds, r,
                    guidance_scale=guidance_scale,
                )
            return sched.step(v, x, t, r)

        # Build the list of (t, r) flow-map jumps:
        #   - full-step mode (Feature 2): every scheduler step
        #     timesteps[i] -> timesteps[i+1], matching the eval / inference grid.
        #     grad_timestep is unused here.
        #   - shortcut mode (default): the 3-segment prev / current / post split,
        #     3 forwards regardless of num_inference_steps.
        if oc.rollout_full_steps:
            jump_pairs = [
                (timesteps[i], timesteps[i + 1]) for i in range(num_inference_steps)
            ]
        else:
            jump_pairs = [
                (timesteps[0], timesteps[grad_timestep]),                  # prev
                (timesteps[grad_timestep], timesteps[grad_timestep + 1]),  # current (fine)
                (timesteps[grad_timestep + 1], timesteps[-1]),            # post (-> 0)
            ]

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            x = init_latents
            for t_scalar, r_scalar in jump_pairs:
                x = _one_jump(x, t_scalar, r_scalar)
        return x

    # ------------------------------------------------------------------
    # DMD KL gradient (fake - real, with normalization)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_dmd_kl_grad(
        self,
        noisy_latent: torch.Tensor,
        pred_video: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        normalization: bool = True,
    ) -> torch.Tensor:
        """Compute the DMD distribution-matching gradient on x_0 space.

        Aligned with AnyFlow ``_compute_kl_grad``:

            pred_fake_video = flow-map step(noise_pred_fake, noisy, t -> 0)
            pred_real_video = flow-map step(noise_pred_real, noisy, t -> 0) [+ optional CFG]
            grad = (pred_fake_video - pred_real_video) / |pred_video - pred_real_video|.mean()
        """
        oc = self.config.anyflow_onpolicy
        zeros = torch.zeros_like(timesteps)

        # Fake score prediction.
        with self._autocast():
            noise_pred_fake = self._predict_noise_flowmap(
                self.fake_score_net, noisy_latent, prompt_embeds, timesteps,
                pooled_embeds, timesteps,  # discriminator uses r=t (instantaneous)
                guidance_scale=1.0,
            )
        pred_fake_video = self.flowmap_scheduler.step(
            noise_pred_fake, noisy_latent, timesteps, zeros,
        )

        # Real score prediction (conditional, plus optional explicit CFG).
        with self._autocast():
            noise_pred_real = self._predict_noise_flowmap(
                self.real_score, noisy_latent, prompt_embeds, timesteps,
                pooled_embeds, timesteps,
                guidance_scale=1.0,
            )
        pred_real_video = self.flowmap_scheduler.step(
            noise_pred_real, noisy_latent, timesteps, zeros,
        )

        if oc.real_guidance_scale and oc.real_guidance_scale > 0.0:
            self._ensure_uncond_embeds(prompt_embeds.dtype, prompt_embeds.device)
            uncond_emb, uncond_pool = self._expand_uncond_to_batch(prompt_embeds.shape[0])
            with self._autocast():
                noise_pred_real_uncond = self._predict_noise_flowmap(
                    self.real_score, noisy_latent, uncond_emb, timesteps,
                    uncond_pool, timesteps, guidance_scale=1.0,
                )
            pred_real_video_uncond = self.flowmap_scheduler.step(
                noise_pred_real_uncond, noisy_latent, timesteps, zeros,
            )
            guidance_scale = float(oc.real_guidance_scale)
            pred_real_video = pred_real_video_uncond + guidance_scale * (
                pred_real_video - pred_real_video_uncond
            )

        grad = pred_fake_video - pred_real_video
        if normalization:
            p_real = pred_video - pred_real_video
            normalizer = p_real.abs().mean(
                dim=tuple(range(1, p_real.ndim)), keepdim=True,
            )
            grad = grad / (normalizer + 1e-8)
        grad = torch.nan_to_num(grad)
        return grad

    # ------------------------------------------------------------------
    # Generator update (DMD + optional cotrain forward training)
    # ------------------------------------------------------------------

    def _generator_loss(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        latents_for_cotrain: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """AnyFlow on-policy DMD generator loss with optional co-train forward training."""
        oc = self.config.anyflow_onpolicy

        # AnyFlow on-policy DMD path.
        device = torch.device("cuda")
        model_dtype = next(self.generator.parameters()).dtype
        T = float(self.scheduler.config.num_train_timesteps)

        # Choose (sample_step, grad_timestep) — both rank-broadcast.
        sample_step = self._choice_int_synced(oc.num_inference_steps_list)
        grad_timestep = self._sample_int_synced(0, sample_step)

        # Random init noise for rollout.
        batch_size = prompt_embeds.shape[0]
        init_latents = torch.randn(
            batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=device, dtype=model_dtype,
        )

        # Run student rollout (with grad).
        self.generator.train()
        self.real_score.eval()
        self.fake_score_net.eval()
        pred_video = self._shortcut_rollout(
            init_latents=init_latents,
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            num_inference_steps=sample_step,
            grad_timestep=grad_timestep,
            with_grad=True,
            model=self.generator,
        )

        # DMD gradient on a noisy version of the rollout.
        with torch.no_grad():
            t01 = torch.rand(batch_size, dtype=model_dtype, device=device)
            timesteps = (self.flowmap_scheduler.apply_shift(t01) * T).to(device)
            timesteps = timesteps.clamp(oc.dmd_min_timestep, oc.dmd_max_timestep)
            noisy_latent = self.flowmap_scheduler.scale_noise(
                pred_video, timesteps, torch.randn_like(pred_video),
            ).detach()
            grad = self._compute_dmd_kl_grad(
                noisy_latent=noisy_latent,
                pred_video=pred_video,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
                normalization=oc.gradient_normalization,
            )

        dmd_loss = oc.dmd_weight * F.mse_loss(
            pred_video.double(),
            (pred_video.double() - grad.double()).detach(),
            reduction="mean",
        )

        info = {
            "loss_dmd": float(dmd_loss.detach()),
            "sample_step": float(sample_step),
            "grad_timestep": float(grad_timestep),
            "dmd_t_mean": float(timesteps.mean().item()),
        }

        # Optional co-train Stage 1 forward training loss as anchor.
        # Inherits AnyFlowPretrainTrainer._create_dataloader unchanged: the
        # dataloader serves (prompt_embeds, latents) pairs so the cotrain
        # branch always has latents available when this gate is enabled.
        if oc.cotrain_forward_kl and latents_for_cotrain is not None:
            forward_loss, fwd_info = self._compute_forward_training_loss(
                latents_for_cotrain, prompt_embeds, pooled_embeds,
            )
            total = dmd_loss + forward_loss
            info["loss_forward"] = fwd_info["loss_forward"]
            info["loss_total"] = float(total.detach())
            return total, info

        info["loss_total"] = float(dmd_loss.detach())
        return dmd_loss, info

    # ------------------------------------------------------------------
    # Discriminator update (fake score flow matching loss)
    # ------------------------------------------------------------------

    def _discriminator_loss(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Train fake_score_net on student rollout via flow matching loss."""
        oc = self.config.anyflow_onpolicy
        device = torch.device("cuda")
        model_dtype = next(self.fake_score_net.parameters()).dtype
        T = float(self.scheduler.config.num_train_timesteps)

        sample_step = self._choice_int_synced(oc.num_inference_steps_list)
        grad_timestep = self._sample_int_synced(0, sample_step)

        batch_size = prompt_embeds.shape[0]
        init_latents = torch.randn(
            batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=device, dtype=model_dtype,
        )

        self.generator.eval()
        self.real_score.eval()
        self.fake_score_net.train()

        # No-grad rollout to get fake samples.
        with torch.no_grad():
            pred_video = self._shortcut_rollout(
                init_latents=init_latents,
                prompt_embeds=prompt_embeds,
                pooled_embeds=pooled_embeds,
                num_inference_steps=sample_step,
                grad_timestep=grad_timestep,
                with_grad=False,
                model=self.generator,
            )
            latents = pred_video

        # Logit-normal timestep sampling followed by SD3.5's static shift.
        t01 = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size, logit_mean=0.0, logit_std=1.0,
        ).to(device=device, dtype=model_dtype)
        timesteps = (self.flowmap_scheduler.apply_shift(t01) * T).to(device)
        timesteps = timesteps.clamp(oc.dmd_min_timestep, oc.dmd_max_timestep)

        noise = torch.randn_like(latents)
        noisy_latents = self.flowmap_scheduler.scale_noise(latents, timesteps, noise)

        with self._autocast():
            noise_pred = self._predict_noise_flowmap(
                self.fake_score_net, noisy_latents, prompt_embeds, timesteps,
                pooled_embeds, timesteps,  # r = t (instantaneous)
                # Match the conditioning used in _compute_dmd_kl_grad so the
                # fake score is trained and queried at the same guidance point.
                guidance_scale=1.0,
            )
        target = noise - latents
        per_sample = (noise_pred.float() - target.float()) ** 2
        per_sample = per_sample.reshape(batch_size, -1).mean(dim=-1)
        loss = per_sample.mean()

        return loss, {
            "loss_disc": float(loss.detach()),
            "sample_step": float(sample_step),
            "grad_timestep": float(grad_timestep),
            "disc_t_mean": float(timesteps.mean().item()),
        }

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict[str, Any]) -> dict[str, dict[str, float]]:
        device = torch.device("cuda")
        model_dtype = next(self.generator.parameters()).dtype
        latents, prompt_embeds, pooled_embeds = self._prepare_consistency_batch(
            batch, device, model_dtype,
        )
        oc = self.config.anyflow_onpolicy

        metrics: dict[str, dict[str, float]] = {}

        # Optional: text-drop applied uniformly to both generator and
        # discriminator paths so the discriminator sees the same conditioning
        # distribution as the generator.
        ap = self.config.anyflow_pretrain
        if ap.drop_text_ratio > 0.0:
            self._ensure_uncond_embeds(prompt_embeds.dtype, prompt_embeds.device)
            gen = torch.Generator(device=device)
            gen.manual_seed(self.config.train.seed + self.global_step * 31337)
            mask = torch.rand(latents.shape[0], device=device, generator=gen) < ap.drop_text_ratio
            if mask.any():
                uncond_emb, uncond_pool = self._expand_uncond_to_batch(latents.shape[0])
                prompt_embeds = prompt_embeds.clone()
                pooled_embeds = pooled_embeds.clone()
                prompt_embeds[mask] = uncond_emb.to(prompt_embeds.dtype)[mask]
                pooled_embeds[mask] = uncond_pool.to(pooled_embeds.dtype)[mask]

        # Optional sub-batch for DMD branch (rollout is memory-heavy).
        dmd_bs = oc.dmd_batch_size if oc.dmd_batch_size > 0 else latents.shape[0]
        gen_prompt = prompt_embeds[:dmd_bs]
        gen_pooled = pooled_embeds[:dmd_bs]
        gen_latents = latents[:dmd_bs] if oc.cotrain_forward_kl else None

        # 1. Generator update.
        with self.timer.measure("generator_update"):
            gen_loss, gen_info = self._generator_loss(
                gen_prompt, gen_pooled, latents_for_cotrain=gen_latents,
            )
            self.optimizers["generator"].zero_grad(set_to_none=True)
            gen_loss.backward()
            gen_grad = self._compute_grad_norm(self.generator)
            if self.config.solver.generator.max_grad_norm > 0:
                self._clip_grad_norm(self.generator, self.config.solver.generator.max_grad_norm)
            self.optimizers["generator"].step()
            self.schedulers["generator"].step()
            gen_info["lr"] = self.schedulers["generator"].get_last_lr()[0]
            gen_info["grad_norm"] = gen_grad
            metrics["generator"] = gen_info

        # 2. Discriminator update(s).
        with self.timer.measure("discriminator_update"):
            disc_info = {}
            for _ in range(max(1, oc.discriminator_update_ratio)):
                disc_loss, disc_info_step = self._discriminator_loss(
                    prompt_embeds[:dmd_bs], pooled_embeds[:dmd_bs],
                )
                self.optimizers["fake_score"].zero_grad(set_to_none=True)
                disc_loss.backward()
                disc_grad = self._compute_grad_norm(self.fake_score_net)
                if self.config.solver.fake_score.max_grad_norm > 0:
                    self._clip_grad_norm(
                        self.fake_score_net,
                        self.config.solver.fake_score.max_grad_norm,
                    )
                self.optimizers["fake_score"].step()
                self.schedulers["fake_score"].step()
                disc_info = disc_info_step
                disc_info["lr"] = self.schedulers["fake_score"].get_last_lr()[0]
                disc_info["grad_norm"] = disc_grad
            metrics["discriminator"] = disc_info

        # 3. EMA on generator (AnyFlow Wan style: decay=0 during warmup, then
        # switch to fixed decay; AnyFlow Wan onpolicy uses ema_warmup_step=200
        # with ema_decay=0.99). Mirrors anyflow_pretrain_trainer above.
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

        return metrics
