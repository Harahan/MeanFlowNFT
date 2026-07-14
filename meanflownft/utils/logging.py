"""
Logging utilities for MeanFlowNFT.

Provides structured wandb logging with metric sections:
- dmd/*: Distribution Matching Distillation loss metrics (every step)
- fake_score/*: Fake score network training metrics (every step)
- nft/*: NFT teacher update metrics + reward scores (every nft_update_interval steps)
- nft_buffer/*: NFT sampling buffer statistics (every nft_update_interval steps)
- eval/*: Generator evaluation reward scores (every eval_interval steps)
- teacher_eval/*: Teacher evaluation reward scores (every eval_interval steps, DMD-NFT only)
- media/*: Generated image samples (every eval_interval steps)
- profile/*: GPU memory and timing metrics (every log_interval steps)
- debug/*: Optional diagnostics (e.g. DMD-NFT debug_log_fm_loss)

X-axis: All panels use wandb's built-in _step (= global_step) as X-axis.
_step is automatically set by the step= parameter in wandb.log() calls
and cannot be matched by glob patterns, avoiding circular references.
Section prefixes are registered explicitly via wandb.define_metric().
When adding a new section, add its prefix to the define_metric loop
in WandbLogger.__init__.

Also provides a simple timer utility for profiling training stages.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch

from meanflownft.parallel.utils import is_main_process

logger = logging.getLogger(__name__)


class WandbLogger:
    """Structured wandb logger with section-based metric organization.

    All metrics are logged with section prefixes (e.g., "dmd/loss_dm",
    "profile/gpu_mem_gb") to keep the wandb dashboard organized.

    Args:
        project: wandb project name.
        run_name: wandb run name.
        entity: wandb entity (team/user).
        config: Config dict to log as wandb config.
        tags: List of tags for the run.
        enabled: Whether to actually log (False for non-main processes or debug).
    """

    def __init__(
        self,
        project: str = "MeanFlowNFT",
        run_name: str = "",
        entity: str = "",
        config: dict | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
        output_dir: str = "",
    ):
        self.enabled = enabled and is_main_process()
        self._run = None

        if self.enabled:
            import wandb
            # Store wandb logs in the output directory to keep everything together
            wandb_dir = output_dir or None
            if wandb_dir:
                os.makedirs(wandb_dir, exist_ok=True)
            self._run = wandb.init(
                project=project,
                name=run_name or None,
                entity=entity or None,
                config=config,
                tags=tags,
                resume="allow",
                dir=wandb_dir,
            )
            # Set X-axis for all dashboard panels to training step.
            # Use wandb's built-in _step (set by step= param in log() calls,
            # which is always global_step). _step is an internal metric that
            # won't be matched by any glob pattern, avoiding circular references.
            # Use explicit section prefixes instead of "*" for the same reason.
            # NOTE: Add new section prefixes here when adding new metric groups.
            for prefix in ["dmd", "fake_score", "nft", "nft_buffer",
                           "eval", "teacher_eval", "media", "profile", "debug"]:
                wandb.define_metric(f"{prefix}/*", step_metric="_step")
            logger.info(f"wandb initialized: {wandb.run.url}")

    def log(self, metrics: dict[str, Any], step: int, section: str = "") -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dict of metric_name -> value.
            step: Global training step.
            section: Optional section prefix (e.g., "dmd", "fake_score").
                If provided, metrics are logged as "section/metric_name".
        """
        if not self.enabled:
            return

        if section:
            metrics = {f"{section}/{k}": v for k, v in metrics.items()}

        self._run.log(metrics, step=step)

    def log_multi_section(self, sections: dict[str, dict[str, Any]], step: int) -> None:
        """Log metrics from multiple sections in a single wandb call.

        Args:
            sections: Dict of section_name -> {metric_name: value}.
            step: Global training step.

        Example:
            logger.log_multi_section({
                "dmd": {"loss_dm": 0.5, "gradient_norm": 1.2},
                "fake_score": {"loss_fake": 0.3},
                "profile": {"gpu_mem_gb": 24.5},
            }, step=100)
        """
        if not self.enabled:
            return

        flat = {}
        for section, metrics in sections.items():
            for k, v in metrics.items():
                flat[f"{section}/{k}"] = v
        self._run.log(flat, step=step)

    def log_images(
        self,
        images: list,
        captions: list[str],
        step: int,
        key: str = "media/eval",
    ) -> None:
        """Log images to wandb as a media panel.

        Args:
            images: List of PIL.Image instances.
            captions: List of caption strings (one per image).
            step: Global training step.
            key: Wandb log key.
        """
        if not self.enabled:
            return
        import wandb
        wandb_images = [
            wandb.Image(img, caption=cap)
            for img, cap in zip(images, captions)
        ]
        self._run.log({key: wandb_images}, step=step)

    def finish(self) -> None:
        """Finish the wandb run."""
        if self.enabled and self._run is not None:
            self._run.finish()


class Timer:
    """Lightweight timer for profiling training stages.

    Usage:
        timer = Timer()
        with timer.measure("forward_pass"):
            output = model(input)
        print(timer.summary())  # "forward_pass: 0.123s"
    """

    def __init__(self):
        self._times: dict[str, list[float]] = defaultdict(list)
        self._start: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        """Context manager to measure wall-clock time of a code block.

        Args:
            name: Name of the operation being timed.
        """
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self._times[name].append(elapsed)

    def get_last(self, name: str) -> float:
        """Get the most recent measurement for a named operation."""
        if name not in self._times or len(self._times[name]) == 0:
            return 0.0
        return self._times[name][-1]

    def get_average(self, name: str) -> float:
        """Get the average time across all measurements for a named operation."""
        if name not in self._times or len(self._times[name]) == 0:
            return 0.0
        return sum(self._times[name]) / len(self._times[name])

    def get_metrics(self) -> dict[str, float]:
        """Get average times for all tracked operations, suitable for logging."""
        return {f"time_{k}_s": self.get_average(k) for k in self._times}

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._times.clear()


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format.

    Only the main process logs at INFO level; other ranks log at WARNING.
    Forces reconfiguration even if basicConfig was already called (e.g., by torchrun).
    """
    fmt = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    target_level = level if is_main_process() else logging.WARNING

    # Force reconfigure: basicConfig is a no-op if root logger already has handlers
    root = logging.getLogger()
    root.setLevel(target_level)
    # Remove existing handlers to avoid duplicate output
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setLevel(target_level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(handler)

    # Silence noisy third-party INFO logs while keeping warnings/errors visible.
    # Example: httpx request-line logs from UnifiedReward API calls.
    logging.getLogger("httpx").setLevel(logging.WARNING)
