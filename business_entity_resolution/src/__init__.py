"""Business entity resolution baseline utilities."""

from .features import build_pair_features
from .model import compare_validation_results, train_baseline_models

__all__ = ["build_pair_features", "train_baseline_models", "compare_validation_results"]
