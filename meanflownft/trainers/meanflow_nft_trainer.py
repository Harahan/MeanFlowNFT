"""
MeanFlowNFT Trainer for MeanFlowNFT.

Combines a pretrained flow-map (``u_theta(x_t, r, t)``) parameterization
with DiffusionNFT (arXiv 2509.16117) RL fine-tuning. Achieves strict policy
improvement in instantaneous-velocity space by deriving::

    V_theta(x_t, t) = u_theta(x_t, r, t) + (t - r) * dF/dt
    dF/dt = central_diff_t( u_theta ; v_t )

so that NFT can be applied in v-space (where its theory holds) while the
network remains MeanFlow-parameterized (preserving few-step generation).

The central difference (no-grad) is the stop-gradient approximation of
the analytic ``dF/dt`` in the derivation, mirroring Stage 1 pretraining's
existing ``_compute_central_difference``. It is FSDP / HSDP / DDP-
compatible.

The tangent direction ``v_dir`` driving the central difference is selected by
``MeanFlowNFTRLConfig.cd_velocity_source``:

  - "noise_minus_x0" (default): ``v_dir = eps - x_0`` (MeanFlow baseline).
  - "u_self" (iMF, arXiv:2512.02012 §4.1 Algorithm 2; opt-in): replace the
    conditional velocity ``e - x`` with each network's own BARE
    conditional boundary prediction at ``r = t``:

        v_dir = model(z_t, t, t, c).detach()   ≈  v_cfg

    No reverse-CFG fusion here. Reason: NFT supervises ``v_theta``
    (= ``f_c`` itself + central-diff PDE term), NOT a derived fused
    quantity. The trajectory the model traces at inference is along
    ``f_c = v_cfg`` (baked in by the pretrained reverse-CFG-fused LoRA),
    so the MeanFlow identity for this trajectory requires the JVP
    tangent to be ``f_c(z, t, t) = v_cfg`` — exactly mirroring iMF
    Algorithm 2's ``v_c = fn(z, t, t, w, c)`` bare-cond tangent.
    NFT loss target itself is UNCHANGED.

NOTE: This differs from ``anyflow_pretrain``'s u_self mode, which DOES
apply reverse-CFG fusion (because pretrain's loss is on the fused ``g``
quantity → needs ``v_marg`` direction). NFT's loss is on ``f_c`` directly
→ needs ``v_cfg`` direction → bare. See ``_compute_v`` docstring inline.

The rollout uses MeanFlowNFT's eval-time N-step Euler flow-map sampler
(same as ``AnyFlowPretrainTrainer._generate_latents``), so on-policy
samples reflect the actual inference path.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn as nn

from meanflownft.config import MeanFlowNFTConfig
from meanflownft.models.sd35_flowmap import (
    predict_noise_sd35_flowmap,
    setup_flowmap_for_sd3,
)
from meanflownft.parallel.utils import get_rank, get_world_size
from meanflownft.schedulers.flowmap_scheduler import FlowMapScheduler
from meanflownft.trainers.nft_trainer import NFTTrainer

logger = logging.getLogger(__name__)


class MeanFlowNFTTrainer(NFTTrainer):
    """MeanFlowNFT reinforcement learning for flow-map generators."""

    def __init__(self, config: MeanFlowNFTConfig):
        super().__init__(config)
        if not self.config.flowmap_model.enabled:
            raise ValueError(
                "MeanFlowNFTTrainer requires flowmap_model.enabled=True."
            )

        # Flow-map scheduler is initialized before LoRA injection.
        self.flowmap_scheduler: Optional[FlowMapScheduler] = None

        # Shared central-difference cache. When
        # ``MeanFlowNFTRLConfig.share_cd_with_old`` is True we run the CD
        # once per ``_nft_loss`` step on ``self.old_model`` (in
        # :meth:`_pre_compute_v_hook`) and store the result here; the
        # three ``_compute_v`` calls (generator / old / ref) all read
        # from this slot instead of running their own CD. Cleared by
        # :meth:`_post_compute_v_hook`. See the field docstring on
        # ``MeanFlowNFTRLConfig.share_cd_with_old`` for the rationale +
        # error analysis.
        self._shared_dF_dt: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Flow-map setup hooks (before/after LoRA injection).
    # ------------------------------------------------------------------

    def _pre_setup_lora(self) -> None:
        """Install flow-map (two-time) wrapper BEFORE LoRA is injected.

        This matches AnyFlowPretrainTrainer's ordering: delta_embedder must
        exist before setup_lora freezes everything (so resumed LoRA
        checkpoints' delta_embedder weights load correctly).
        """
        fm = self.config.flowmap_model
        setup_flowmap_for_sd3(
            self.generator, gate_value=fm.gate_value, deltatime_type=fm.deltatime_type,
        )

        # Build the FlowMap scheduler with SD3.5's static shift.
        sched_cfg = self.scheduler.config
        self.flowmap_scheduler = FlowMapScheduler(
            num_train_timesteps=int(sched_cfg.num_train_timesteps),
            shift=float(getattr(sched_cfg, "shift", 1.0)),
            weight_type="uniform",
        )

    def _post_setup_lora(self) -> None:
        """Re-enable delta_embedder gradients after setup_lora freezes everything."""
        if not self.config.model.generator_lora.enabled:
            return
        for name, param in self.generator.named_parameters():
            if "delta_embedder" in name:
                param.requires_grad = True

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def setup_models(self) -> None:
        super().setup_models()
        if get_rank() == 0:
            logger.info(
                "[MeanFlowNFT] nft_velocity_mode=%s",
                self._nft_cfg().nft_velocity_mode,
            )

    # ------------------------------------------------------------------
    # SD3.5 flow-map velocity prediction
    # ------------------------------------------------------------------

    def _predict_noise_flowmap(
        self,
        model: nn.Module,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_embeds: torch.Tensor,
        r_timesteps: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        return predict_noise_sd35_flowmap(
            model=model,
            noisy_latents=noisy_latents,
            text_embeddings=prompt_embeds,
            timesteps=timesteps,
            pooled_prompt_embeds=pooled_embeds,
            r_timesteps=r_timesteps,
            guidance_scale=guidance_scale,
        )

    # ------------------------------------------------------------------
    # Sampling rollout: MeanFlowNFT N-step Euler flow-map (eval-style).
    # ------------------------------------------------------------------

    def _get_last_rollout_timestep_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(t_arr, r_arr)`` from the MeanFlowNFT flow-map rollout.

        ``flowmap_scheduler.timesteps`` has length ``N + 1`` after the
        last rollout (``[t_0, t_1, ..., t_N=0]``); the actual flow-map
        jumps the rollout performed are ``(t_i, t_{i+1})`` for ``i in [0, N)``.
        Return those ``N`` pairs as ``t_arr = timesteps[:-1]`` and
        ``r_arr = timesteps[1:]`` — both length ``N``.
        """
        ts = self.flowmap_scheduler.timesteps.detach().to(dtype=torch.float32)
        return ts[:-1], ts[1:]

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
        """N-step flow-map Euler rollout (one transformer fwd per step).

        Mirrors :meth:`AnyFlowPretrainTrainer._generate_latents` exactly,
        which is the eval-time inference path. The flow-map rollout is CFG-free
        (guidance_scale=1.0); the ``guidance_scale`` arg is accepted for
        signature compatibility but ignored.
        """
        del guidance_scale  # Flow-map inference is CFG-free.
        device = prompt_embeds.device
        dtype = next(model.parameters()).dtype
        batch_size = prompt_embeds.shape[0]

        self.flowmap_scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.flowmap_scheduler.timesteps  # [N+1]

        x_t = torch.randn(
            batch_size, self.latent_channels, self.latent_size, self.latent_size,
            device=device, dtype=dtype,
        )
        for i in range(num_steps):
            t = timesteps[i].expand(batch_size).to(device=device, dtype=torch.float32)
            r = timesteps[i + 1].expand(batch_size).to(device=device, dtype=torch.float32)
            with self._autocast():
                v = self._predict_noise_flowmap(
                    model, x_t, prompt_embeds, t, pooled_embeds, r,
                    guidance_scale=1.0,
                )
            x_t = self.flowmap_scheduler.step(v, x_t, t, r)
        return x_t

    # ------------------------------------------------------------------
    # dF/dt estimator: central difference (no-grad)
    # ------------------------------------------------------------------

    def _central_diff_dudt(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        v_dir: torch.Tensor,
        t_raw: torch.Tensor,
        r_raw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute (F(z+, t+eps_cd, r) - F(z-, t-eps_cd, r)) / (2 * eps_cd).

        Aligned with :meth:`AnyFlowPretrainTrainer._compute_central_difference`
        (same step formula, same epsilon scaling): perturbations to ``x_t``
        are in [0,1] units (eps_cd / T), perturbations to ``t`` are in raw
        timestep units (eps_cd), and the denominator is ``2 * eps_cd``.

        Always invoked under :func:`torch.no_grad`.
        """
        T = float(self.scheduler.config.num_train_timesteps)
        eps_cd = float(self._nft_cfg().central_diff_epsilon)

        t_plus = t_raw + eps_cd
        t_minus = t_raw - eps_cd
        z_plus = x_t + v_dir * (eps_cd / T)
        z_minus = x_t - v_dir * (eps_cd / T)

        with self._autocast():
            f_plus = self._predict_noise_flowmap(
                model, z_plus, prompt_embeds, t_plus, pooled_embeds, r_raw,
                guidance_scale=1.0,
            )
            f_minus = self._predict_noise_flowmap(
                model, z_minus, prompt_embeds, t_minus, pooled_embeds, r_raw,
                guidance_scale=1.0,
            )
        return (f_plus - f_minus) / (2.0 * eps_cd)

    # ------------------------------------------------------------------
    # Shared dF/dt helper (extracted so it can be reused by both
    # ``_pre_compute_v_hook`` (precompute on old_model) and ``_compute_v``
    # (legacy per-network path).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_dF_dt_for_model(
        self,
        *,
        cd_model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r_raw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """Run one central-difference dF/dt on ``cd_model``.

        ``v_dir`` is selected by ``meanflow_nft.cd_velocity_source``:

          - ``"noise_minus_x0"``: recover ``eps`` from (x_t, x_0, t) and use
            ``v_dir = eps - x_0`` (MeanFlow baseline; model-independent).
          - ``"u_self"``: ``v_dir = cd_model(x_t, t, t, c).detach()``
            (iMF Algorithm 2 bare-cond tangent; depends on ``cd_model``).

        Always runs inside :func:`torch.no_grad`.
        """
        cfg = self._nft_cfg()
        T = float(self.scheduler.config.num_train_timesteps)

        if cfg.cd_velocity_source == "noise_minus_x0":
            # Recover eps from x_t / x_0 / t (deterministic;
            # eps = (x_t - (1 - t01) * x_0) / t01).
            t01 = (t.to(x_0.dtype) / T).view(x_0.shape[0], *([1] * (x_0.ndim - 1)))
            eps = (x_t - (1.0 - t01) * x_0) / t01.clamp(min=1.0 / T)
            v_dir = eps - x_0
        elif cfg.cd_velocity_source == "u_self":
            # iMF (arXiv:2512.02012 §4.1, Algorithm 2) boundary-condition
            # variant: tangent = ``cd_model``'s bare conditional output at
            # ``r = t``. When ``share_cd_with_old`` is True we get the
            # SAME ``v_dir`` (= old_model's u_c) for all three networks'
            # CD computations, which is exactly what algebraically
            # cancels the (CD_θ - CD_old) term out of v_θ - v_old.
            with self._autocast():
                v_dir = self._predict_noise_flowmap(
                    cd_model, x_t, prompt_embeds, t, pooled_embeds, t,  # r = t
                    guidance_scale=1.0,
                )
            v_dir = v_dir.to(x_t.dtype)
        else:
            raise ValueError(
                f"Unknown meanflow_nft.cd_velocity_source={cfg.cd_velocity_source!r}; "
                "expected 'noise_minus_x0' or 'u_self'."
            )

        return self._central_diff_dudt(
            model=cd_model,
            x_t=x_t.detach(),
            v_dir=v_dir.detach(),
            t_raw=t,
            r_raw=r_raw,
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
        )

    # ------------------------------------------------------------------
    # Pre / post hooks (override of NFTTrainer no-ops): when
    # ``share_cd_with_old`` is True, precompute ONE central-difference
    # dF/dt on ``self.old_model`` before the three ``_compute_v`` calls
    # and cache it in ``self._shared_dF_dt``. The cache is cleared in
    # ``_post_compute_v_hook`` (always, via try/finally in the parent).
    # ------------------------------------------------------------------

    def _pre_compute_v_hook(
        self,
        *,
        x_t: torch.Tensor,
        t_for_model: torch.Tensor,
        r_for_model: Optional[torch.Tensor],
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        x_0: torch.Tensor,
    ) -> None:
        cfg = self._nft_cfg()
        if cfg.nft_velocity_mode == "direct_u":
            return
        if not cfg.share_cd_with_old:
            return
        # Fast path (r == t for all samples → (t-r)·dF/dt = 0): no CD needed.
        # Matches the fast-path guard inside ``_compute_v`` exactly.
        if float(cfg.diffusion_ratio) >= 1.0 - 1e-8:
            return
        if self.old_model is None:
            # ref-only setup or pre-setup; degrade gracefully to legacy CD.
            return
        if r_for_model is None:
            # Defensive guard: MeanFlowNFT always supplies a flow-map (t, r) pair.
            return
        self._shared_dF_dt = self._compute_dF_dt_for_model(
            cd_model=self.old_model,
            x_t=x_t, t=t_for_model, r_raw=r_for_model,
            prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds,
            x_0=x_0,
        )

    def _post_compute_v_hook(self) -> None:
        self._shared_dF_dt = None

    # ------------------------------------------------------------------
    # V_theta construction with central difference.
    #
    # NFTTrainer._nft_loss calls
    #     _compute_v(model, x_t, t, ..., with_grad, x_0=x_0, r_raw=r_raw)
    # with t / r in raw timestep units (paired from the rollout's discrete
    # (t_i, t_{i+1}) jumps). We build:
    #
    #   V = u(x_t, r, t) + (t - r) * dF/dt
    #
    # where dF/dt is the no-grad central difference of u along ``v_dir``.
    # ``v_dir`` is selected by ``meanflow_nft.cd_velocity_source``:
    #   - "noise_minus_x0" (default): v_dir = eps - x_0 (eps recovered
    #     from x_t / x_0 / t). Original AnyFlow / MeanFlow.
    #   - "u_self"        (iMF, opt-in): bare conditional boundary forward
    #         v_dir = model(x_t, t, t, c).detach()  ≈  v_cfg
    #     Mirrors iMF Algorithm 2's ``v_c`` (bare cond, no fusion).
    #     Tangent = trajectory = v_cfg, fully self-consistent. NFT's
    #     pretrained LoRA already has f_c ≈ v_cfg from step 0, so no
    #     bootstrap dead-lock (unlike pretrain's u_self).
    # The leading u(x_t, r, t) carries the gradient when with_grad=True.
    #
    # NOTE: pretrain's u_self DOES apply (u_c - (1-G)*u_uncond)/G because
    # pretrain's loss is on the fused ``g`` quantity (needs v_marg dir).
    # NFT's loss is on f_c directly (needs v_cfg dir → bare).
    # ------------------------------------------------------------------

    def _compute_u(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        r_raw: torch.Tensor,
        *,
        with_grad: bool,
    ) -> torch.Tensor:
        """Leading MeanFlowNFT flow-map output u(x_t, r, t)."""
        ctx = nullcontext() if with_grad else torch.no_grad()
        with ctx:
            with self._autocast():
                return self._predict_noise_flowmap(
                    model, x_t, prompt_embeds, t, pooled_embeds, r_raw,
                    guidance_scale=1.0,
                )

    def _compute_v(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        *,
        with_grad: bool,
        x_0: Optional[torch.Tensor] = None,
        r_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_0 is None or r_raw is None:
            raise ValueError(
                "MeanFlowNFTTrainer._compute_v requires x_0 and r_raw kwargs "
                "(passed by NFTTrainer._nft_loss). Got "
                f"x_0={'set' if x_0 is not None else 'None'}, "
                f"r_raw={'set' if r_raw is not None else 'None'}."
            )

        cfg = self._nft_cfg()
        T = float(self.scheduler.config.num_train_timesteps)

        # Direct-u ablation: do not convert u -> V with the MeanFlow identity.
        # The parent NFT loss still sees the returned tensor as "v", while
        # MeanFlowNFT's training-time (t, r) sampler continues to draw r/t from
        # diffusion_ratio and consistency_ratio.
        if cfg.nft_velocity_mode == "direct_u":
            return self._compute_u(
                model, x_t, t, prompt_embeds, pooled_embeds, r_raw,
                with_grad=with_grad,
            )

        # ----------------------------------------------------------------
        # Fast path: when the (t, r) sampler forces r == t for every sample
        # (``diffusion_ratio == 1.0`` — pure-diffusion mode, partition rule
        # in :meth:`_sample_training_tr_three_mode`), the PDE-residual term
        # ``(t - r) * dF/dt`` is identically zero, so ``V_theta`` reduces to
        # just the leading ``u_theta(x_t, r=t, t)``. We can skip:
        #   - the central difference (2 no-grad fwds), AND
        #   - the iMF u_self boundary fwd (1 no-grad fwd when enabled).
        # Net saving: 2-3 fwds per ``_compute_v`` call (×3 networks per
        # ``_nft_loss``: generator + old + optional ref), i.e. 3-4× total
        # speedup for the default SD3.5 yaml (``diffusion_ratio: 1.0``).
        #
        # This is BIT-IDENTICAL to the full path: ``u + 0 * dF_dt == u`` in
        # IEEE 754 (and strictly safer — avoids any NaN/inf propagation from
        # ``0 * NaN`` if the central diff is numerically unstable). Matches
        # iMF (arXiv:2512.02012) Fig.3 caption ("t=r samples have JVP=0") and
        # MeanFlow's ``target = v_target - (t-r)*dF/dt`` form. The check is
        # a pure-Python scalar comparison (no GPU-CPU sync).
        # ----------------------------------------------------------------
        if float(cfg.diffusion_ratio) >= 1.0 - 1e-8:
            return self._compute_u(
                model, x_t, t, prompt_embeds, pooled_embeds, r_raw,
                with_grad=with_grad,
            )

        # dF/dt: either (a) reuse the shared-CD slot precomputed once on
        # ``self.old_model`` in :meth:`_pre_compute_v_hook` (when
        # ``share_cd_with_old`` is True), or (b) compute per-network CD
        # on ``model`` (legacy behavior).
        if self._shared_dF_dt is not None:
            dF_dt = self._shared_dF_dt
        else:
            dF_dt = self._compute_dF_dt_for_model(
                cd_model=model, x_t=x_t, t=t, r_raw=r_raw,
                prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds,
                x_0=x_0,
            )

        # Leading u_theta(x_t, r, t): under grad iff `with_grad`.
        u = self._compute_u(
            model, x_t, t, prompt_embeds, pooled_embeds, r_raw,
            with_grad=with_grad,
        )

        # (t - r) in raw timestep units; dF_dt is also per-raw-timestep
        # (denominator 2*eps_cd in raw units), so the product is dimensionless.
        # Matches Stage 1's PDE-residual target form
        # ``target = v_target - (t - r) * dF/dt`` exactly.
        diff = (t - r_raw).view(x_t.shape[0], *([1] * (x_t.ndim - 1)))
        return u + diff.to(u.dtype) * dF_dt.to(u.dtype)

    # ==================================================================
    # Training-time (t, r) sampling — MeanFlowNFT, decoupled from rollout
    #
    # Mirrors AnyFlow pretrain's `_sample_three_mode_tr` (Pattern A) for r and
    # the on-policy DMD generator-loss t-clamp (Pattern B) for t. Each inner
    # backward draws a fresh (t, r) per chunk; FSDP-symmetric via a per-step
    # seeded RNG keyed by (global_step, _nft_epoch, inner_epoch_idx,
    # chunk_start, j_idx, rank). Partition assignment is deterministic by
    # global index across ranks (every rank computes its own slice using the
    # same global rule), so no broadcast is needed.
    # ==================================================================

    def _num_training_timesteps(self, n_ts_total: int) -> int:
        """Use fresh MeanFlowNFT pairs, decoupled from the rollout grid."""
        n = int(self._nft_cfg().num_training_timesteps_per_sample)
        if n <= 0:
            raise ValueError(
                "meanflow_nft.num_training_timesteps_per_sample must be positive."
            )
        # One-time alignment log so a misconfigured ratio is obvious on rank 0.
        if not getattr(self, "_three_mode_log_emitted", False):
            cfg = self._nft_cfg()
            total_r = float(cfg.diffusion_ratio) + float(cfg.consistency_ratio)
            if not 0.0 <= total_r <= 1.0 + 1e-6:
                raise ValueError(
                    f"meanflow_nft.diffusion_ratio ({cfg.diffusion_ratio}) + "
                    f"consistency_ratio ({cfg.consistency_ratio}) must lie in "
                    f"[0, 1] (got {total_r:.4f})."
                )
            if get_rank() == 0:
                logger.info(
                    "[MeanFlowNFT] (t, r) sampling: three-mode partition "
                    "(diffusion_ratio=%.3f, consistency_ratio=%.3f, "
                    "generic_ratio=%.3f), t clamp=[%d, %d], "
                    "iters/inner-batch=%d (decoupled from rollout N=%d), "
                    "nft_velocity_mode=%s.",
                    cfg.diffusion_ratio, cfg.consistency_ratio,
                    1.0 - total_r,
                    cfg.nft_min_timestep, cfg.nft_max_timestep,
                    n, n_ts_total, cfg.nft_velocity_mode,
                )
            self._three_mode_log_emitted = True
        return max(1, n)

    def _prepare_inner_epoch_tr_state(self, **_kwargs: Any) -> dict[str, Any]:
        """MeanFlowNFT draws fresh per iter — no pre-shuffle needed."""
        return {}

    def _draw_training_tr(
        self,
        *,
        state: dict[str, Any],   # noqa: ARG002 - unused; we draw fresh
        chunk_start: int,
        chunk_end: int,
        j_idx: int,
        inner_epoch_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-iter (t, r) draw: three-mode partition + Pattern B clamp.

        See class docstring above for the full recipe; returns a CPU-free
        ``(t_raw, r_raw)`` pair in float32 raw-timestep units (callers do
        their own dtype casting).
        """
        chunk_size = max(0, chunk_end - chunk_start)
        if chunk_size == 0:
            empty = torch.empty(0, device=device, dtype=torch.float32)
            return empty, empty.clone()
        return self._sample_training_tr_three_mode(
            batch_size=chunk_size,
            chunk_start=chunk_start,
            j_idx=j_idx,
            inner_epoch_idx=inner_epoch_idx,
            device=device,
            dtype=dtype,
        )

    def _make_training_tr_rng(
        self,
        *,
        inner_epoch_idx: int,
        chunk_start: int,
        j_idx: int,
        device: torch.device,
        stream_offset: int = 0,
    ) -> torch.Generator:
        """Per-iteration seeded RNG (FSDP-symmetric partition; per-rank noise).

        Each call inside one inner train loop step must use a distinct seed so
        that consecutive ``num_training_timesteps_per_sample`` (t, r) draws are
        independent (``global_step`` only ticks on opt.step, so it is constant
        across all draws in one optimizer cycle). The seed mixes:
          - ``train.seed``                : run-level determinism
          - ``self._nft_epoch``           : outer NFT epoch
          - ``self.global_step``          : opt step inside the epoch
          - ``inner_epoch_idx``           : inner epoch within sampling slice
          - ``chunk_start``               : chunk position
          - ``j_idx``                     : iteration index within chunk
          - ``get_rank()``                : per-rank noise (different t1/t2)
        """
        gen = torch.Generator(device=device)
        seed = (
            int(self.config.train.seed)
            + int(self._nft_epoch) * 1_000_003
            + int(self.global_step) * 10_007
            + int(inner_epoch_idx) * 101
            + int(chunk_start) * 17
            + int(j_idx) * 3
            + get_rank() * 7919
            + int(stream_offset)
        )
        gen.manual_seed(int(seed % (2**63 - 1)))
        return gen

    def _sample_training_tr_three_mode(
        self,
        *,
        batch_size: int,
        chunk_start: int,
        j_idx: int,
        inner_epoch_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one (t, r) batch per MeanFlowNFT partition + shift/clamp.

        Recipe (aligned with
        :meth:`AnyFlowPretrainTrainer._sample_three_mode_tr` for r and
        :meth:`AnyFlowOnPolicyTrainer._generator_loss` Pattern B for t):

        1. ``t1, t2 ~ U(0, 1)`` per local sample (per-rank seeded; rank in seed
           makes ranks see different noise but the partition rule is global).
        2. ``t01 = max(t1, t2)``, ``r01 = min(t1, t2)``.
        3. Global-index partition (``global_idx = rank * batch_size + b``):
            - first ``round(diffusion_ratio * global_bsz)``  : ``r01 = t01`` (r = t)
            - next  ``round(consistency_ratio * global_bsz)``: ``r01 = 0`` (r = 0)
            - rest                                           : keep ``r01``.
        4. Apply SD3.5's static shift to both endpoints and scale to raw
           timestep units.
        5. ``t_raw.clamp_(nft_min_timestep, nft_max_timestep)``. Then
           re-clamp ``r_raw`` to ``[0, t_raw]`` so the flow-map jump direction
           stays valid (``r <= t``). Finally enforce the partition equalities
           exactly (``r = t`` on the diffusion slice, ``r = 0`` on the
           consistency slice) — the clamp does not perturb those samples.

        FSDP-symmetric: the partition is a deterministic function of global
        index; the per-rank RNG seed mixes rank so t1/t2 differ across ranks
        but the partition rule (which sample goes to which mode) is identical.
        No broadcast or all_reduce needed.
        """
        nc = self._nft_cfg()
        T = float(self.scheduler.config.num_train_timesteps)

        ws = max(1, get_world_size())
        rank = get_rank()
        global_start_idx = rank * batch_size
        global_bsz = ws * batch_size

        # Per-iteration seeded RNG (FSDP-symmetric partition; per-rank noise).
        gen = self._make_training_tr_rng(
            inner_epoch_idx=inner_epoch_idx,
            chunk_start=chunk_start,
            j_idx=j_idx,
            device=device,
        )
        t1 = torch.rand(batch_size, dtype=dtype, device=device, generator=gen)
        t2 = torch.rand(batch_size, dtype=dtype, device=device, generator=gen)
        t01 = torch.maximum(t1, t2)
        r01 = torch.minimum(t1, t2)

        # Three-mode partition by global index. We mark which samples land in
        # each slice; mask vectors avoid the python loop in
        # AnyFlowPretrain._sample_three_mode_tr (which works fine but the
        # vectorized form is identical and a touch faster).
        n_diffusion = round(float(nc.diffusion_ratio) * global_bsz)
        n_consistency = round(float(nc.consistency_ratio) * global_bsz)
        g_idx = torch.arange(
            global_start_idx, global_start_idx + batch_size,
            device=device, dtype=torch.long,
        )
        is_diffusion = g_idx < n_diffusion
        is_consistency = (g_idx >= n_diffusion) & (g_idx < n_diffusion + n_consistency)
        r01 = torch.where(is_diffusion, t01, r01)
        r01 = torch.where(is_consistency, torch.zeros_like(r01), r01)

        # Shift and scale both endpoints with SD3.5's static schedule.
        t_shifted = self.flowmap_scheduler.apply_shift(t01) * T
        r_shifted = self.flowmap_scheduler.apply_shift(r01) * T

        # Clamp t to the NFT-specific range; then re-clamp r to [0, t] so the
        # flow-map jump direction stays valid (r <= t). Finally re-pin the
        # partition equalities (r = t for diffusion, r = 0 for consistency)
        # against any clamp drift.
        t_min = float(nc.nft_min_timestep)
        t_max = float(nc.nft_max_timestep)
        t_raw = t_shifted.clamp(min=t_min, max=t_max)
        r_raw = r_shifted.clamp(min=0.0).minimum(t_raw)
        r_raw = torch.where(is_diffusion, t_raw, r_raw)
        r_raw = torch.where(is_consistency, torch.zeros_like(r_raw), r_raw)

        return t_raw.to(dtype=torch.float32), r_raw.to(dtype=torch.float32)
