"""
Wan2.1-T2V AnyFlow + DiffusionNFT Trainer for MeanFlowNFT.

Video counterpart of :class:`MeanFlowNFTTrainer`. Reuses the *entire* MeanFlowNFT
algorithm unchanged — the central-difference ``V_theta = u_theta(x_t, r, t) +
(t - r) * dF/dt`` derivation, three-mode ``(t, r)`` sampling, the DiffusionNFT
loss kernel, old_model EMA decay, and the checkpoint / resume machinery are all
inherited (they are tensor-rank agnostic). Only the video-specific surface is
overridden:

  * model loading      : ``WanAnyFlowTransformer3DModel`` + ``AutoencoderKLWan`` +
                         UMT5 with native two-time conditioning.
  * latent geometry    : 5D ``[B, F, C, H, W]`` (frames-before-channels).
  * rollout            : 5D AnyFlow flow-map Euler, CFG-free.
  * text conditioning  : UMT5 sequence embeds, NO pooled (a benign zero
                         ``[B, 1]`` placeholder is threaded through the parent's
                         pooled-embeds plumbing; the Wan forward ignores it).
  * VAE decode         : video ``[B, F, 3, H, W]``.
  * rewards / eval     : video-aware ``VideoMultiScorer`` + a video ``_evaluate``.

The per-sample training ``(t, r)`` are sampled as scalars ``[B]`` (one timestep
per video, shared across frames) and broadcast to the per-frame ``[B, F]`` form
the Wan transformer expects only at the model call (in
:func:`meanflownft.models.wan.predict_noise_wan`). This keeps the inherited
three-mode sampler and NFT loss exactly as-is.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from meanflownft.config import MeanFlowNFTConfig
from meanflownft.models.wan import (
    decode_wan_latents,
    encode_prompts_wan,
    predict_noise_wan,
)
from meanflownft.parallel.utils import (
    ddp_wrap_model,
    fsdp_wrap_model,
    get_rank,
    get_transformer_wrap_policy,
    get_world_size,
    is_main_process,
)
from meanflownft.rewards.video_multi_scorer import VideoMultiScorer
from meanflownft.schedulers.flowmap_scheduler import FlowMapScheduler
from meanflownft.trainers.meanflow_nft_trainer import MeanFlowNFTTrainer
from meanflownft.trainers.nft_utils import PerPromptStatTracker
from meanflownft.utils.lora import setup_lora

logger = logging.getLogger(__name__)


class WanMeanFlowNFTTrainer(MeanFlowNFTTrainer):
    """DiffusionNFT for the Wan2.1-T2V AnyFlow flow-map (video) generator."""

    def __init__(self, config: MeanFlowNFTConfig):
        super().__init__(config)
        if not getattr(self, "_is_wan", False):
            raise ValueError(
                "WanMeanFlowNFTTrainer requires model.model_type='wan'."
            )
        # Video latent geometry (populated in setup_models).
        self.latent_frames: Optional[int] = None
        # Eval latent frame count (decoupled from training via video.eval_num_frames;
        # equals latent_frames when eval_num_frames<=0).
        self.eval_latent_frames: Optional[int] = None
        self.latent_h: Optional[int] = None
        self.latent_w: Optional[int] = None
        # Separate video eval scorer (the base ``_eval_scorer`` is image-only).
        self._eval_video_scorer: Optional[VideoMultiScorer] = None

    # ------------------------------------------------------------------
    # The Wan model is natively two-time; only the flow-map scheduler is needed.
    # ------------------------------------------------------------------

    def _pre_setup_lora(self) -> None:
        self._use_dynamic_shifting = False
        self._image_seq_len = 0
        self.flowmap_scheduler = FlowMapScheduler(
            num_train_timesteps=int(self.scheduler.config.num_train_timesteps),
            shift=float(getattr(self.scheduler.config, "shift", 5.0)),
            weight_type="uniform",
        )

    # ``_post_setup_lora`` is inherited: it unfreezes any parameter whose name
    # contains "delta_embedder", which matches the Wan model's
    # ``condition_embedder.delta_embedder.*`` (full-FT of the r-conditioning MLP,
    # and persisted alongside LoRA by the checkpoint code.

    # ------------------------------------------------------------------
    # Model setup (mirrors NFTTrainer.setup_models, video-flavored)
    # ------------------------------------------------------------------

    def setup_models(self) -> None:
        # Delay large model imports until trainer construction.
        from meanflownft.models.wan import load_wan_models

        model_cfg = self.config.model
        dist_cfg = self.config.distributed

        logger.info("=" * 60)
        logger.info("Setting up Wan MeanFlowNFT models")
        logger.info("=" * 60)

        components = load_wan_models(model_cfg)
        base_transformer = components["transformer"]
        self.text_encoders = components["text_encoders"]
        self.tokenizers = components["tokenizers"]
        self.scheduler = components["scheduler"]
        self.vae = components["vae"]

        if model_cfg.gradient_checkpointing:
            base_transformer.enable_gradient_checkpointing()

        self.generator = copy.deepcopy(base_transformer)
        del base_transformer

        self._load_model_init_from_path(
            self.generator, model_cfg.generator_init_path, "generator",
        )

        # Build the scheduler; the model already contains two-time embeddings.
        self._pre_setup_lora()

        if model_cfg.generator_lora.enabled:
            setup_lora(self.generator, model_cfg.generator_lora)
            logger.info(
                f"  Generator: LoRA injected (rank={model_cfg.generator_lora.rank})"
            )
        else:
            self.generator.train()
            logger.info("  Generator: full-weight trainable")

        # Unfreeze delta_embedder (r-conditioning) after LoRA freezes everything.
        self._post_setup_lora()

        for enc in self.text_encoders:
            enc.requires_grad_(False)
            enc.eval()
        self.vae.requires_grad_(False)
        self.vae.eval()
        device = torch.device("cuda")
        self.vae.to(device)
        for enc in self.text_encoders:
            enc.to(device)

        # --- Video latent geometry ---
        self.latent_channels = self.generator.config.in_channels
        vae_t = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        vae_s = int(getattr(self.vae.config, "scale_factor_spatial", 8))
        vcfg = self.config.video
        if (vcfg.num_frames - 1) % vae_t != 0:
            raise ValueError(
                f"video.num_frames-1 ({vcfg.num_frames - 1}) must be divisible by "
                f"vae temporal factor {vae_t} (e.g. 81 frames for factor 4)."
            )
        if vcfg.height % vae_s != 0 or vcfg.width % vae_s != 0:
            raise ValueError(
                f"video.height/width ({vcfg.height}x{vcfg.width}) must be divisible "
                f"by vae spatial factor {vae_s}."
            )
        self.latent_frames = (vcfg.num_frames - 1) // vae_t + 1
        self.latent_h = vcfg.height // vae_s
        self.latent_w = vcfg.width // vae_s
        # Eval frame count: decoupled from training when video.eval_num_frames>0,
        # else falls back to training num_frames (latent_frames). Eval rollouts
        # use self.eval_latent_frames (see _evaluate); training rollouts keep
        # self.latent_frames.
        eval_nf = int(getattr(vcfg, "eval_num_frames", 0) or 0)
        if eval_nf > 0:
            if (eval_nf - 1) % vae_t != 0:
                raise ValueError(
                    f"video.eval_num_frames-1 ({eval_nf - 1}) must be divisible by "
                    f"vae temporal factor {vae_t} (e.g. 81/49/33 for factor 4)."
                )
            self.eval_latent_frames = (eval_nf - 1) // vae_t + 1
        else:
            self.eval_latent_frames = self.latent_frames
        # Benign value for any inherited reference to ``latent_size`` (unused on
        # the video path: rollout / central-diff are all explicitly 5D below).
        self.latent_size = self.latent_h
        # The flow-map rollout uses ``flowmap_scheduler``; the base x0_scheduler
        # (consistency/euler) is never used on the video path.
        self.x0_scheduler = None

        if self.config.train.use_ema:
            self.generator_ema = copy.deepcopy(self.generator)
            self.generator_ema.requires_grad_(False)
            self.generator_ema.eval()

        self.old_model = copy.deepcopy(self.generator)
        self.old_model.requires_grad_(False)
        self.old_model.eval()

        if self._nft_cfg().kl_weight > 0.0:
            self.ref_model = copy.deepcopy(self.generator)
            self.ref_model.requires_grad_(False)
            self.ref_model.eval()

        # FSDP flattens params per unit and requires a UNIFORM dtype within each
        # flat group. The vendored Wan transformer keeps some modules in fp32 via
        # ``_keep_in_fp32_modules`` (norms / time_embedder / scale_shift_table),
        # so after ``from_pretrained(torch_dtype=bf16)`` those params stay fp32
        # while the rest are bf16 -> FSDP raises "Must flatten tensors with
        # uniform dtype". Cast every model (base + delta_embedder + LoRA) to the
        # FSDP param dtype before wrapping. (FP32LayerNorm still computes in fp32
        # internally, so this is numerically safe for bf16 training.)
        _dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        _fsdp_dtype = _dtype_map[model_cfg.dtype]
        for _m in (self.generator, self.generator_ema, self.old_model, self.ref_model):
            if _m is not None:
                _m.to(_fsdp_dtype)

        self._wrap_models(dist_cfg)

        self.models = {"generator": self.generator}
        if self.generator_ema is not None:
            self.models["generator_ema"] = self.generator_ema
        if self.old_model is not None:
            self.models["old_model"] = self.old_model
        if self.ref_model is not None:
            self.models["ref_model"] = self.ref_model

        if self._nft_cfg().per_prompt_stat_tracking:
            self._stat_tracker = PerPromptStatTracker(
                global_std=self._nft_cfg().per_prompt_global_std,
            )

        if get_rank() == 0:
            logger.info(
                "[WanMeanFlowNFT] latent geometry: frames=%d, C=%d, H=%d, W=%d",
                self.latent_frames, self.latent_channels, self.latent_h, self.latent_w,
            )
            logger.info(
                "[WanMeanFlowNFT] nft_velocity_mode=%s, cd_velocity_source=%s",
                self._nft_cfg().nft_velocity_mode, self._nft_cfg().cd_velocity_source,
            )
        logger.info("Wan MeanFlowNFT model setup complete")
        logger.info("=" * 60)

    def _wrap_models(self, dist_cfg) -> None:
        """FSDP/DDP wrap on Wan transformer-block boundaries."""
        strategy = dist_cfg.strategy
        fsdp_precision = self.config.model.dtype

        if strategy == "fsdp":
            from meanflownft.models.wan_transformer import WanTransformerBlock
            wrap_policy = get_transformer_wrap_policy(WanTransformerBlock)
            for name in ("generator", "generator_ema", "old_model", "ref_model"):
                mod = getattr(self, name)
                if mod is not None:
                    setattr(
                        self, name,
                        fsdp_wrap_model(
                            mod,
                            sharding_strategy=dist_cfg.fsdp_sharding,
                            fsdp_precision=fsdp_precision,
                            auto_wrap_policy=wrap_policy,
                        ),
                    )
        elif strategy == "ddp":
            device = torch.device("cuda")
            has_lora = self.config.model.generator_lora.enabled
            ddp_find_unused_cfg = self.config.distributed.ddp_find_unused_parameters
            find_unused = has_lora if ddp_find_unused_cfg is None else bool(ddp_find_unused_cfg)
            self.generator = ddp_wrap_model(
                self.generator.to(device), find_unused_parameters=find_unused,
            )
            for name in ("generator_ema", "old_model", "ref_model"):
                mod = getattr(self, name)
                if mod is not None:
                    setattr(self, name, mod.to(device))
        else:
            raise ValueError(f"Unknown distributed strategy: {strategy}")

    # ------------------------------------------------------------------
    # Text encoding (UMT5, no pooled -> benign zero placeholder)
    # ------------------------------------------------------------------

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device("cuda")
        embeds, _ = encode_prompts_wan(
            prompts, self.text_encoders[0], self.tokenizers[0], device,
            max_sequence_length=int(self.config.video.max_sequence_length),
        )
        # Benign pooled placeholder so the parent's pooled-embeds plumbing
        # (store/cat/index/.to) works; the Wan forward ignores it.
        dummy_pooled = torch.zeros(embeds.shape[0], 1, device=device, dtype=embeds.dtype)
        return embeds, dummy_pooled

    # ------------------------------------------------------------------
    # Velocity prediction dispatchers (video, flow-map native)
    # ------------------------------------------------------------------

    def _predict_noise_flowmap(
        self,
        model: nn.Module,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_embeds: torch.Tensor,  # noqa: ARG002 - Wan has no pooled
        r_timesteps: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        return predict_noise_wan(
            model=model,
            noisy_latents=noisy_latents,
            text_embeddings=prompt_embeds,
            timesteps=timesteps,
            r_timesteps=r_timesteps,
            guidance_scale=guidance_scale,
        )

    # ------------------------------------------------------------------
    # 5D AnyFlow flow-map rollout (CFG-free)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _rollout_samples(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        *,
        num_steps: int,
        guidance_scale: float,
        model: nn.Module,
        latent_frames: Optional[int] = None,
    ) -> torch.Tensor:
        del guidance_scale  # AnyFlow inference is CFG-free.
        device = prompt_embeds.device
        dtype = next(model.parameters()).dtype
        batch_size = prompt_embeds.shape[0]
        # Training rollouts use the training frame count (self.latent_frames);
        # eval passes self.eval_latent_frames to decouple eval clip length.
        n_frames = self.latent_frames if latent_frames is None else int(latent_frames)

        self.flowmap_scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.flowmap_scheduler.timesteps  # [N+1]

        x_t = torch.randn(
            batch_size, n_frames, self.latent_channels,
            self.latent_h, self.latent_w, device=device, dtype=dtype,
        )
        for i in range(num_steps):
            t = timesteps[i].expand(batch_size).to(device=device, dtype=torch.float32)
            r = timesteps[i + 1].expand(batch_size).to(device=device, dtype=torch.float32)
            with self._autocast():
                v = self._predict_noise_flowmap(
                    model, x_t, prompt_embeds, t, pooled_embeds, r, guidance_scale=1.0,
                )
            x_t = self.flowmap_scheduler.step(v, x_t, t, r)
        return x_t

    # ------------------------------------------------------------------
    # Video VAE decode -> [B, F, 3, H, W] in [0, 1]
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_latents_to_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        return decode_wan_latents(self.vae, latents)

    # ------------------------------------------------------------------
    # Reward scorer: video-aware MultiScorer
    # ------------------------------------------------------------------

    def _ensure_reward_scorer(self) -> None:
        if self._reward_scorer is not None:
            return
        cfg = self._nft_cfg()
        if not cfg.reward_fn:
            raise ValueError(
                "WanMeanFlowNFTTrainer requires meanflow_nft.reward_fn with at "
                "least one weighted video reward "
                "(hpsv3_general / hpsv3_percentile / videoalign_mq / videoalign_ta)."
            )
        self._reward_scorer = VideoMultiScorer(
            device=torch.device("cpu"),
            score_dict=cfg.reward_fn,
            allow_unavailable=False,
            **(cfg.reward_model_paths or {}),
        )

    # ------------------------------------------------------------------
    # Video evaluation (reuses base symmetric-batching + cross-rank aggregation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> None:
        """Stream test-set generation and scoring without retaining all videos."""
        eval_cfg = self.config.eval
        if not eval_cfg.reward_fn:
            return
        rank = get_rank()
        world_size = get_world_size()
        logger.info(
            "[Eval] Wan video eval at epoch %d (rank=%d)",
            self._nft_epoch,
            rank,
        )
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(),
        }
        try:
            if self._eval_video_scorer is None:
                self._eval_video_scorer = VideoMultiScorer(
                    device=torch.device("cpu"),
                    score_dict=eval_cfg.reward_fn,
                    allow_unavailable=True,
                    **(self._nft_cfg().reward_model_paths or {}),
                )
            scorer = self._eval_video_scorer
            if not scorer.active_reward_names:
                logger.warning("[Eval] No available video rewards; skipping.")
                return
            eval_seed = int(eval_cfg.eval_seed) + rank
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            torch.cuda.manual_seed_all(eval_seed)
            gen_model = self.models.get("generator_ema") or self.models.get(
                "generator"
            )
            was_training = gen_model.training
            gen_model.eval()
            all_prompts = self._get_cached_eval_prompts()
            rank_prompts = all_prompts[rank::world_size]
            batch_size = max(1, int(eval_cfg.eval_batch_size))
            max_per_rank = (len(all_prompts) + world_size - 1) // world_size
            max_batches = (max_per_rank + batch_size - 1) // batch_size
            sums: dict[str, float] = {}
            num_local = 0
            media: list[torch.Tensor] = []
            media_prompts: list[str] = []
            scorer.to(torch.device("cuda"))
            try:
                for batch_index in range(max_batches):
                    start_index = batch_index * batch_size
                    end_index = min(start_index + batch_size, len(rank_prompts))
                    if start_index < len(rank_prompts):
                        batch_prompts = rank_prompts[start_index:end_index]
                        actual_count = end_index - start_index
                        if len(batch_prompts) < batch_size:
                            batch_prompts = batch_prompts + [batch_prompts[-1]] * (
                                batch_size - len(batch_prompts)
                            )
                    else:
                        fallback = rank_prompts[0] if rank_prompts else all_prompts[0]
                        batch_prompts = [fallback] * batch_size
                        actual_count = 0
                    embeds, dummy = self._encode_prompts(batch_prompts)
                    latents = self._rollout_samples(
                        embeds,
                        dummy,
                        num_steps=int(eval_cfg.eval_num_steps),
                        guidance_scale=1.0,
                        model=gen_model,
                        latent_frames=self.eval_latent_frames,
                    )
                    videos = self._decode_latents_to_tensor(latents)
                    if actual_count:
                        valid_videos = videos[:actual_count]
                        valid_prompts = batch_prompts[:actual_count]
                        details, _ = scorer(
                            valid_videos,
                            valid_prompts,
                            metadata=None,
                            only_strict=False,
                        )
                        for name in (*scorer.active_reward_names, "mean"):
                            if name in details:
                                values = np.asarray(details[name], dtype=np.float64)
                                sums[name] = sums.get(name, 0.0) + float(values.sum())
                        if rank == 0 and len(media) < int(eval_cfg.num_media_images):
                            keep = min(
                                actual_count,
                                int(eval_cfg.num_media_images) - len(media),
                            )
                            media.extend(valid_videos[:keep].detach().cpu())
                            media_prompts.extend(valid_prompts[:keep])
                        num_local += actual_count
                    del embeds, dummy, latents, videos
                    torch.cuda.empty_cache()
            finally:
                scorer.to(torch.device("cpu"))
                torch.cuda.empty_cache()
            local_metrics = {
                name: value / num_local
                for name, value in sums.items()
                if num_local > 0
            }
            metrics = self._aggregate_eval_metrics_across_ranks(
                local_metrics,
                num_local,
            )
            if was_training:
                gen_model.train()
            if self.wandb_logger:
                self.wandb_logger.log(
                    metrics,
                    step=self.global_step,
                    section="eval",
                )
                media_tensor = torch.stack(media) if media else None
                self._log_eval_videos(media_tensor, media_prompts)
            if is_main_process():
                values = " | ".join(
                    f"{name}={value:.4g}" for name, value in metrics.items()
                )
                logger.info(
                    "[Eval] epoch=%d step=%d | %s",
                    self._nft_epoch,
                    self.global_step,
                    values,
                )
            self._maybe_run_vbench(gen_model)
        finally:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.random.set_rng_state(rng_state["torch_cpu"])
            torch.cuda.set_rng_state(rng_state["torch_cuda"])
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    def _log_eval_videos(
        self, videos_tensor: Optional[torch.Tensor], prompts_local: list[str],
    ) -> None:
        """Log a few eval videos to wandb as mp4 (GenRL-style).

        Encodes up to ``eval.num_media_images`` sample clips to mp4 (via
        ``WandbLogger.log_videos`` / imageio-ffmpeg) at the configured
        ``video.fps`` and logs them under ``media/eval_video``.
        """
        if (
            self.wandb_logger is None
            or not is_main_process()
            or self.config.eval.num_media_images <= 0
            or videos_tensor is None
            or videos_tensor.shape[0] == 0
        ):
            return
        try:
            n = min(int(self.config.eval.num_media_images), videos_tensor.shape[0])
            self.wandb_logger.log_videos(
                videos=[videos_tensor[i] for i in range(n)],
                captions=prompts_local[:n],
                step=self.global_step,
                key="media/eval_video",
                fps=int(self.config.video.fps),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Eval] video logging failed: {e}")

    def _log_sampling_preview(self, preview) -> None:
        """Log sampling-phase (rollout) previews as mp4 videos (GenRL-style).

        Overrides the image-based base hook. ``preview`` is
        ``(videos[B, F, 3, H, W], prompts)`` from the first NFT sampling batch;
        enabled via ``meanflow_nft.log_sample_images > 0``. Logs to
        ``media/nft_samples``.
        """
        cfg = self._nft_cfg()
        if (
            cfg.log_sample_images <= 0
            or not is_main_process()
            or self.wandb_logger is None
            or preview is None
        ):
            return
        vids, prompts_preview = preview
        n = min(int(cfg.log_sample_images), vids.shape[0])
        try:
            self.wandb_logger.log_videos(
                videos=[vids[i] for i in range(n)],
                captions=prompts_preview[:n],
                step=self.global_step,
                key="media/nft_samples",
                fps=int(self.config.video.fps),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[NFT] sample video logging failed: {e}")

    # ------------------------------------------------------------------
    # Standalone VBench eval hook (implemented in meanflownft.eval.vbench_eval).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _maybe_run_vbench(self, gen_model: nn.Module) -> None:
        vb = getattr(self.config.eval, "vbench", None)
        if vb is None or not vb.enabled:
            return
        every = int(getattr(vb, "eval_every_epochs", 0))
        evaluation_only = bool(getattr(self, "_evaluation_only", False))
        if not evaluation_only and (
            every <= 0 or (self._nft_epoch % every != 0)
        ):
            return
        try:
            from meanflownft.eval.vbench_eval import run_vbench_eval
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[VBench] not available ({e}); skipping.")
            return
        run_vbench_eval(self, gen_model)
