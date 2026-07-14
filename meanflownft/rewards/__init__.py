"""
Reward scoring for MeanFlowNFT training and evaluation.

Adapted from DiffusionNFT's reward implementations.
Provides a unified MultiScorer interface for computing weighted reward metrics
during training evaluation.

Supported scorers:
- pickscore: PickScore v1 (text-image alignment)
- hpsv2: Human Preference Score v2
- hpsv3: Human Preference Score v3
- clipscore: CLIP-based text-image similarity
- aesthetic: LAION aesthetic predictor
- imagereward: ImageReward v1.0
- ocr: PaddleOCR-based text rendering accuracy
- geneval2: Qwen3-VL Soft-TIFA compositional evaluation
"""

from meanflownft.rewards.multi_scorer import MultiScorer

__all__ = ["MultiScorer"]
