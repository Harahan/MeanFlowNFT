"""Wan2.1 inference for normal, AnyFlow, and MeanFlowNFT policies."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

from meanflownft.config import LoRAConfig, ModelConfig
from meanflownft.models.wan import (
    decode_wan_latents,
    encode_prompts_wan,
    load_wan_models,
    predict_noise_wan,
)
from meanflownft.utils.checkpoint import load_wan_meanflow_nft_adapter

logger = logging.getLogger(__name__)

INFERENCE_MODES = {"normal", "anyflow", "meanflow_nft"}


def _apply_overrides(data: dict, overrides: dict[str, Any]) -> dict:
    for key, value in overrides.items():
        if "." in key:
            raise ValueError(
                f"Inference overrides are flat; nested key is invalid: {key}"
            )
        data[key] = value
    return data


@dataclass
class InferenceConfig:
    mode: str = "meanflow_nft"
    normal_pretrained_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    anyflow_pretrained_path: str = (
        "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
    )
    meanflow_nft_path: str = ""
    dtype: str = "bf16"
    num_steps: int = 4
    normal_num_steps: int = 50
    guidance_scale: float = 1.0
    normal_guidance_scale: float = 5.0
    negative_prompt: str = ""
    num_frames: int = 81
    height: int = 480
    width: int = 832
    fps: int = 16
    max_sequence_length: int = 512
    seed: int = 12345
    batch_size: int = 1
    prompts: list[str] = field(default_factory=list)
    prompt_file: str = ""
    max_prompts: int = 0
    output_dir: str = "./inference_outputs/wan2.1"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "net.0.proj",
            "net.2",
        ]
    )

    @classmethod
    def from_yaml(
        cls,
        path: str,
        overrides: dict[str, Any] | None = None,
    ) -> "InferenceConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"Inference config must be a YAML mapping: {path}")
        if overrides:
            raw = _apply_overrides(raw, overrides)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown inference field(s): {unknown}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        self.mode = str(self.mode).lower()
        if self.mode not in INFERENCE_MODES:
            raise ValueError(
                f"mode must be one of {sorted(INFERENCE_MODES)}"
            )
        self.dtype = str(self.dtype).lower()
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if self.mode == "normal":
            if not str(self.normal_pretrained_path).strip():
                raise ValueError("normal_pretrained_path is required")
        else:
            if not str(self.anyflow_pretrained_path).strip():
                raise ValueError("anyflow_pretrained_path is required")
            if float(self.guidance_scale) != 1.0:
                raise ValueError("AnyFlow inference is CFG-free (guidance=1.0)")
        if self.mode == "meanflow_nft" and not str(
            self.meanflow_nft_path
        ).strip():
            raise ValueError("meanflow_nft mode requires meanflow_nft_path")
        if self.mode != "meanflow_nft" and self.meanflow_nft_path:
            raise ValueError(
                "meanflow_nft_path is valid only in meanflow_nft mode"
            )
        if self.num_steps < 1 or self.normal_num_steps < 1:
            raise ValueError("inference step counts must be positive")
        if self.num_frames < 1 or (self.num_frames - 1) % 4:
            raise ValueError("num_frames must satisfy (num_frames - 1) % 4 == 0")
        if self.height % 8 or self.width % 8:
            raise ValueError("height and width must be divisible by 8")
        if self.batch_size < 1 or self.max_prompts < 0:
            raise ValueError("batch_size must be positive; max_prompts >= 0")
        if not self.prompts and not str(self.prompt_file).strip():
            raise ValueError("Set prompts or prompt_file")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WanInference:
    _DTYPES = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device("cuda", self.local_rank)
        self.dtype = self._DTYPES[config.dtype]
        self.pipeline = None
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.adapter_info: dict[str, Any] | None = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Wan inference requires a CUDA GPU")
        torch.cuda.set_device(self.device)
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl")
        if self.config.mode == "normal":
            self._setup_normal()
        else:
            self._setup_anyflow()

    def _setup_normal(self) -> None:
        from diffusers import WanPipeline

        self.pipeline = WanPipeline.from_pretrained(
            self.config.normal_pretrained_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=self.rank != 0)

    def _setup_anyflow(self) -> None:
        components = load_wan_models(
            ModelConfig(
                pretrained_path=self.config.anyflow_pretrained_path,
                model_type="wan",
                gradient_checkpointing=False,
                dtype=self.config.dtype,
                generator_lora=LoRAConfig(enabled=False),
            )
        )
        self.transformer = components["transformer"]
        self.vae = components["vae"]
        self.text_encoder = components["text_encoders"][0]
        self.tokenizer = components["tokenizers"][0]
        self.scheduler = components["scheduler"]
        if self.config.mode == "meanflow_nft":
            self.adapter_info = load_wan_meanflow_nft_adapter(
                self.transformer,
                self.config.meanflow_nft_path,
                LoRAConfig(
                    enabled=True,
                    rank=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=list(self.config.lora_target_modules),
                    init_lora_weights="gaussian",
                ),
            )
        self.transformer.requires_grad_(False).eval().to(self.device)
        self.text_encoder.requires_grad_(False).eval().to(self.device)
        self.vae.requires_grad_(False).eval().to(self.device)

    @staticmethod
    def _load_prompt_file(path: str) -> list[str]:
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        if path_obj.suffix == ".txt":
            prompts = path_obj.read_text(encoding="utf-8").splitlines()
        elif path_obj.suffix == ".json":
            raw = json.loads(path_obj.read_text(encoding="utf-8"))
            prompts = [
                item["prompt"] if isinstance(item, dict) else str(item)
                for item in raw
            ]
        elif path_obj.suffix == ".jsonl":
            prompts = []
            for line in path_obj.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    prompts.append(
                        item["prompt"] if isinstance(item, dict) else str(item)
                    )
        else:
            raise ValueError("Prompt file must be .txt, .json, or .jsonl")
        return [prompt.strip() for prompt in prompts if prompt.strip()]

    def _prompts(self) -> list[str]:
        prompts = (
            [str(prompt).strip() for prompt in self.config.prompts]
            if self.config.prompts
            else self._load_prompt_file(self.config.prompt_file)
        )
        prompts = [prompt for prompt in prompts if prompt]
        if self.config.max_prompts:
            prompts = prompts[: self.config.max_prompts]
        if not prompts:
            raise ValueError("No non-empty prompts were loaded")
        return prompts

    def _normal_video(self, prompt: str, index: int) -> np.ndarray:
        generator = torch.Generator(device=self.device).manual_seed(
            self.config.seed + index
        )
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=self.config.negative_prompt or None,
            height=self.config.height,
            width=self.config.width,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.normal_num_steps,
            guidance_scale=self.config.normal_guidance_scale,
            generator=generator,
            output_type="np",
            max_sequence_length=self.config.max_sequence_length,
        )
        frames = output.frames
        if isinstance(frames, list):
            frames = frames[0]
        array = np.asarray(frames)
        if array.ndim == 5:
            array = array[0]
        return array

    @torch.no_grad()
    def _anyflow_videos(
        self,
        prompts: list[str],
        indices: list[int],
    ) -> list[np.ndarray]:
        embeds, _ = encode_prompts_wan(
            prompts,
            self.text_encoder,
            self.tokenizer,
            self.device,
            max_sequence_length=self.config.max_sequence_length,
        )
        embeds = embeds.to(dtype=self.dtype)
        latent_frames = (self.config.num_frames - 1) // 4 + 1
        latent_h = self.config.height // 8
        latent_w = self.config.width // 8
        latents = torch.stack(
            [
                torch.randn(
                    latent_frames,
                    int(self.transformer.config.in_channels),
                    latent_h,
                    latent_w,
                    device=self.device,
                    dtype=self.dtype,
                    generator=torch.Generator(device=self.device).manual_seed(
                        self.config.seed + index
                    ),
                )
                for index in indices
            ],
            dim=0,
        )
        self.scheduler.set_timesteps(
            self.config.num_steps,
            device=self.device,
        )
        timesteps = self.scheduler.timesteps
        autocast_enabled = self.dtype in {torch.bfloat16, torch.float16}
        for step_index in range(self.config.num_steps):
            t = timesteps[step_index].expand(len(prompts)).float()
            r = timesteps[step_index + 1].expand(len(prompts)).float()
            with torch.autocast(
                device_type="cuda",
                dtype=self.dtype,
                enabled=autocast_enabled,
            ):
                velocity = predict_noise_wan(
                    model=self.transformer,
                    noisy_latents=latents,
                    text_embeddings=embeds,
                    timesteps=t,
                    r_timesteps=r,
                    guidance_scale=1.0,
                )
            latents = self.scheduler.step(velocity, latents, t, r)
        videos = decode_wan_latents(self.vae, latents).cpu().numpy()
        return [
            video.transpose(0, 2, 3, 1)
            for video in videos
        ]

    def _save_video(self, frames: np.ndarray, path: Path) -> None:
        import imageio.v2 as imageio

        array = np.asarray(frames)
        if array.dtype != np.uint8:
            array = (
                np.clip(array, 0.0, 1.0) * 255.0
            ).round().astype(np.uint8)
        imageio.mimsave(
            path,
            array,
            fps=self.config.fps,
            codec="libx264",
            format="FFMPEG",
        )

    def run(self) -> dict[str, Any]:
        prompts = self._prompts()
        output_dir = Path(self.config.output_dir) / self.config.mode
        if self.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        local_pairs = [
            (index, prompt)
            for index, prompt in enumerate(prompts)
            if index % self.world_size == self.rank
        ]
        local_records: list[dict[str, Any]] = []
        if self.config.mode == "normal":
            for index, prompt in local_pairs:
                frames = self._normal_video(prompt, index)
                filename = f"{index:05d}.mp4"
                self._save_video(frames, output_dir / filename)
                local_records.append(
                    {"index": index, "prompt": prompt, "file": filename}
                )
        else:
            for start in range(0, len(local_pairs), self.config.batch_size):
                chunk = local_pairs[start : start + self.config.batch_size]
                indices = [item[0] for item in chunk]
                chunk_prompts = [item[1] for item in chunk]
                videos = self._anyflow_videos(chunk_prompts, indices)
                for index, prompt, frames in zip(
                    indices, chunk_prompts, videos
                ):
                    filename = f"{index:05d}.mp4"
                    self._save_video(frames, output_dir / filename)
                    local_records.append(
                        {"index": index, "prompt": prompt, "file": filename}
                    )
                torch.cuda.empty_cache()
        if dist.is_initialized():
            gathered: list[list[dict[str, Any]] | None] = [
                None
            ] * self.world_size
            dist.all_gather_object(gathered, local_records)
            records = [
                record
                for part in gathered
                if part is not None
                for record in part
            ]
        else:
            records = local_records
        records.sort(key=lambda record: record["index"])
        result = {
            "mode": self.config.mode,
            "num_videos": len(records),
            "config": self.config.to_dict(),
            "adapter": self.adapter_info,
            "videos": records,
        }
        if self.rank == 0:
            (output_dir / "metadata.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Saved %d videos to %s", len(records), output_dir)
        return result

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
