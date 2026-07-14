#!/usr/bin/env python3
"""Generate SD3.5-Medium latent data for MeanFlowNFT Stage 1/2.

The script runs the frozen SD3.5 teacher on a prompt list and stores one
``.pt`` payload per prompt:

    {
        "latents": Tensor[C, H, W],
        "prompt_embeds": Tensor[sequence, hidden],
        "pooled_embeds": Tensor[hidden],
        "prompt": str,
    }

Defaults produce the release dataset:

    data/anyflow/sd35m_laion_aes_6p5_40step_cfg4.5_512

Generation is deterministic per prompt, distributed across ``torchrun`` ranks,
atomic, and resume-safe. Existing sample files are reused and included in the
rebuilt ``metadata.jsonl``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

# Make ``meanflownft`` importable when launched from the repository root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from meanflownft.utils.fast_init import fast_init  # noqa: E402


logger = logging.getLogger("generate_consistency_data")

DEFAULT_PRETRAINED_PATH = os.environ.get(
    "MEANFLOWNFT_SD35_PATH",
    "models/stable-diffusion-3.5-medium",
)
DEFAULT_PROMPT_FILE = "dataset/laion_aes_6p5/train.txt"
DEFAULT_OUTPUT_DIR = os.environ.get(
    "MEANFLOWNFT_DATA_DIR",
    "data/anyflow/sd35m_laion_aes_6p5_40step_cfg4.5_512",
)


def _setup_distributed() -> tuple[int, int]:
    """Initialize NCCL under torchrun and return ``(rank, world_size)``."""
    if (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and not dist.is_initialized()
    ):
        from datetime import timedelta

        dist.init_process_group(backend="nccl", timeout=timedelta(hours=6))
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return 0, 1


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _load_prompts(path: str, max_samples: int = -1) -> list[str]:
    """Load prompts from a JSON, JSONL, or one-prompt-per-line text file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Prompt source not found: {path}")

    if source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or not data:
            raise ValueError(f"Prompt JSON must contain a non-empty list: {path}")
        if isinstance(data[0], str):
            prompts = data
        elif isinstance(data[0], dict):
            prompts = [item.get("prompt", item.get("text", "")) for item in data]
        else:
            raise ValueError(f"Unsupported JSON entry type: {type(data[0])}")
    elif source.suffix.lower() == ".jsonl":
        prompts = []
        with source.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict):
                    prompts.append(item.get("prompt", item.get("text", "")))
                else:
                    raise ValueError(f"Unsupported JSONL entry: {item!r}")
    elif source.suffix.lower() in {".txt", ".text"}:
        with source.open("r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
    else:
        raise ValueError(f"Unsupported prompt file extension: {source.suffix}")

    prompts = [prompt for prompt in prompts if prompt]
    return prompts[:max_samples] if max_samples > 0 else prompts


def _sample_relpath(index: int) -> str:
    """Use two directory levels to avoid very large shared-FS directories."""
    return f"samples/{index // 10000:04d}/{index:08d}.pt"


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_dataset_manifest(
    args: argparse.Namespace,
    prompts: list[str],
    *,
    end_idx: int,
) -> dict[str, Any]:
    selected_digest = hashlib.sha256()
    for prompt in prompts:
        selected_digest.update(prompt.encode("utf-8"))
        selected_digest.update(b"\0")
    return {
        "format_version": 1,
        "model_type": "sd35",
        "pretrained_path": os.path.abspath(
            os.path.expanduser(args.pretrained_path)
        ),
        "prompt_file": os.path.abspath(os.path.expanduser(args.prompt_file)),
        "prompt_file_sha256": _sha256_file(args.prompt_file),
        "selected_prompts_sha256": selected_digest.hexdigest(),
        "selected_prompt_count": len(prompts),
        "max_samples": int(args.max_samples),
        "start_idx": int(args.start_idx),
        "end_idx": int(end_idx),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "image_resolution": int(args.image_resolution),
        "negative_prompt": str(args.negative_prompt),
        "seed": int(args.seed),
        "compute_dtype": str(args.dtype),
        "storage_dtype": "bf16",
        "text_max_sequence_length": 256,
        "output_format": "latent",
    }


def _validate_or_create_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    rank: int,
    adopt_legacy_output: bool,
) -> None:
    """Synchronously reject recipe changes before any rank loads the model."""
    error: list[str | None] = [None]
    if rank == 0:
        try:
            manifest_path = output_dir / "dataset_manifest.json"
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if existing != manifest:
                    differing = sorted(
                        key
                        for key in set(existing) | set(manifest)
                        if existing.get(key) != manifest.get(key)
                    )
                    raise RuntimeError(
                        "Dataset manifest mismatch; refusing unsafe resume. "
                        f"Differing fields: {differing}"
                    )
            else:
                shard_exists = any(output_dir.glob("_metadata.rank*.jsonl"))
                metadata_exists = (output_dir / "metadata.jsonl").exists()
                samples_dir = output_dir / "samples"
                sample_exists = (
                    samples_dir.exists()
                    and next(samples_dir.rglob("*.pt"), None) is not None
                )
                if (
                    shard_exists or metadata_exists or sample_exists
                ) and not adopt_legacy_output:
                    raise RuntimeError(
                        "Existing dataset has no dataset_manifest.json. "
                        "Refusing to trust file existence alone; validate the "
                        "legacy data, then rerun with --adopt_legacy_output."
                    )
                _atomic_json_save(manifest, manifest_path)
        except Exception as exc:  # noqa: BLE001
            error[0] = f"{type(exc).__name__}: {exc}"

    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])


def _prompt_seed(prompt: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"sd35\x00{prompt}\x00{base_seed}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _load_pipeline(
    pretrained_path: str,
    dtype: torch.dtype,
    device: torch.device,
):
    """Load a local frozen SD3.5 pipeline without redundant weight init."""
    from diffusers import StableDiffusion3Pipeline

    with fast_init(torch.device("cpu")):
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            pretrained_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
        )

    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)
    for module in (
        pipeline.transformer,
        pipeline.vae,
        pipeline.text_encoder,
        pipeline.text_encoder_2,
        pipeline.text_encoder_3,
    ):
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    return pipeline


@torch.no_grad()
def _generate_batch(
    pipeline,
    prompts: list[str],
    *,
    num_inference_steps: int,
    guidance_scale: float,
    image_resolution: int,
    seeds: list[int],
    device: torch.device,
    negative_prompt: str,
) -> list[dict[str, Any]]:
    """Generate one batch and move compact bf16 payloads to CPU."""
    generators = [
        torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds
    ]
    prompt_embeds, _, pooled_embeds, _ = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        prompt_3=prompts,
        do_classifier_free_guidance=False,
        device=device,
        max_sequence_length=256,
    )
    latents = pipeline(
        prompt=prompts,
        negative_prompt=[negative_prompt] * len(prompts),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=image_resolution,
        width=image_resolution,
        generator=generators,
        output_type="latent",
    ).images.detach()

    return [
        {
            "latents": latents[index]
            .to(dtype=torch.bfloat16, device="cpu")
            .contiguous(),
            "prompt_embeds": prompt_embeds[index]
            .to(dtype=torch.bfloat16, device="cpu")
            .contiguous(),
            "pooled_embeds": pooled_embeds[index]
            .to(dtype=torch.bfloat16, device="cpu")
            .contiguous(),
            "prompt": prompt,
        }
        for index, prompt in enumerate(prompts)
    ]


def _aggregate_metadata(output_dir: Path, world_size: int) -> int:
    """Merge rank-local metadata shards in sample-id order."""
    entries: list[dict[str, Any]] = []
    for rank in range(world_size):
        shard = output_dir / f"_metadata.rank{rank:02d}.jsonl"
        if not shard.exists():
            continue
        with shard.open("r", encoding="utf-8") as handle:
            entries.extend(
                json.loads(line) for line in handle if line.strip()
            )

    entries.sort(key=lambda item: int(item["id"]))
    target = output_dir / "metadata.jsonl"
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(temporary, target)
    return len(entries)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pretrained_path",
        default=DEFAULT_PRETRAINED_PATH,
        help="Local SD3.5-Medium diffusers checkpoint.",
    )
    parser.add_argument("--prompt_file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of prompts; negative means all.",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="Exclusive source index; negative means the end.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Regenerate samples even when their target files exist.",
    )
    parser.add_argument(
        "--adopt_legacy_output",
        action="store_true",
        help=(
            "Create a manifest for an existing pre-manifest dataset only after "
            "you have independently verified its recipe."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--fail_fast_after",
        type=int,
        default=3,
        help="Abort after this many consecutive failed batches (0 disables).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s [%(levelname)s] %(message)s",
    )

    rank, world_size = _setup_distributed()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cpu" and rank == 0:
        logger.warning("CUDA is unavailable; generation will be extremely slow.")
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]

    all_prompts = _load_prompts(args.prompt_file, max_samples=args.max_samples)
    end_idx = args.end_idx if args.end_idx >= 0 else len(all_prompts)
    end_idx = min(end_idx, len(all_prompts))
    if not 0 <= args.start_idx <= end_idx:
        raise ValueError(
            f"Invalid source range [{args.start_idx}, {end_idx}) for "
            f"{len(all_prompts)} prompts."
        )
    prompts = all_prompts[args.start_idx:end_idx]

    if rank == 0:
        logger.info(
            "Loaded %d prompts from %s (source range [%d, %d))",
            len(prompts),
            args.prompt_file,
            args.start_idx,
            end_idx,
        )
        logger.info(
            "SD3.5: %d steps, CFG=%.2f, %dx%d; output=%s; "
            "batch/GPU=%d; world_size=%d; resume=%s",
            args.num_inference_steps,
            args.guidance_scale,
            args.image_resolution,
            args.image_resolution,
            args.output_dir,
            args.batch_size,
            world_size,
            not args.no_resume,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _build_dataset_manifest(args, prompts, end_idx=end_idx)
    _validate_or_create_manifest(
        output_dir,
        manifest,
        rank=rank,
        adopt_legacy_output=bool(args.adopt_legacy_output),
    )
    shard_meta_path = output_dir / f"_metadata.rank{rank:02d}.jsonl"
    shard_meta_path.write_text("", encoding="utf-8")

    rank_indices = list(range(rank, len(prompts), world_size))
    pipeline = None
    if rank_indices:
        if rank == 0:
            logger.info("Loading the SD3.5 pipeline...")
        pipeline = _load_pipeline(args.pretrained_path, dtype, device)
        if rank == 0:
            logger.info("Pipeline loaded.")

    started_at = time.time()
    generated = 0
    skipped = 0
    failed_batches = 0
    consecutive_failures = 0

    with shard_meta_path.open("a", encoding="utf-8") as metadata_file:
        for batch_start in range(0, len(rank_indices), args.batch_size):
            local_indices = rank_indices[
                batch_start : batch_start + args.batch_size
            ]
            global_indices = [
                args.start_idx + local_index for local_index in local_indices
            ]
            batch_prompts = [prompts[index] for index in local_indices]
            target_paths = [
                output_dir / _sample_relpath(index) for index in global_indices
            ]

            pending = [
                index
                for index, target in enumerate(target_paths)
                if args.no_resume or not target.exists()
            ]
            skipped += len(target_paths) - len(pending)

            if pending:
                pending_prompts = [batch_prompts[index] for index in pending]
                pending_paths = [target_paths[index] for index in pending]
                seeds = [
                    _prompt_seed(prompt, args.seed) for prompt in pending_prompts
                ]
                try:
                    payloads = _generate_batch(
                        pipeline,
                        pending_prompts,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        image_resolution=args.image_resolution,
                        seeds=seeds,
                        device=device,
                        negative_prompt=args.negative_prompt,
                    )
                    for payload, target in zip(
                        payloads, pending_paths, strict=True
                    ):
                        _atomic_torch_save(payload, target)
                        generated += 1
                    consecutive_failures = 0
                except Exception as exc:
                    failed_batches += 1
                    consecutive_failures += 1
                    logger.exception(
                        "[rank %d] generation failed near sample %s: %r",
                        rank,
                        global_indices[0] if global_indices else "?",
                        exc,
                    )
                    if (
                        args.fail_fast_after > 0
                        and consecutive_failures >= args.fail_fast_after
                    ):
                        raise RuntimeError(
                            f"[rank {rank}] {consecutive_failures} consecutive "
                            "batches failed; aborting to avoid a silent partial "
                            "dataset."
                        ) from exc

            # Only index samples that actually exist. A failed batch therefore
            # cannot leave dangling metadata entries.
            for global_index, prompt, target in zip(
                global_indices, batch_prompts, target_paths, strict=True
            ):
                if not target.exists():
                    continue
                metadata_file.write(
                    json.dumps(
                        {
                            "id": global_index,
                            "file": str(target.relative_to(output_dir)),
                            "format": "latent",
                            "prompt": prompt,
                            "model_type": "sd35",
                            "num_inference_steps": args.num_inference_steps,
                            "guidance_scale": args.guidance_scale,
                            "image_resolution": args.image_resolution,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            metadata_file.flush()

            processed = batch_start + len(local_indices)
            if (
                (generated + skipped) % max(1, args.log_interval) == 0
                or processed >= len(rank_indices)
            ):
                elapsed = time.time() - started_at
                rate = generated / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[rank %d] %d/%d (generated=%d, skipped=%d, "
                    "failed_batches=%d, %.2f samples/s)",
                    rank,
                    processed,
                    len(rank_indices),
                    generated,
                    skipped,
                    failed_batches,
                    rate,
                )

    if pipeline is not None:
        del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _barrier()
    if rank == 0:
        total = _aggregate_metadata(output_dir, world_size)
        logger.info(
            "Wrote %d metadata entries to %s",
            total,
            output_dir / "metadata.jsonl",
        )
    _barrier()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
