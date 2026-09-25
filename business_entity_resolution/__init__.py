"""Business entity resolution package."""

from .src.features import build_pair_features
from .src.model import compare_validation_results, train_baseline_models
from .src.threshold import choose_threshold, decide_entity_match, entity_decision, keep_match, should_keep_match

__all__ = [
    "build_pair_features",
    "train_baseline_models",
    "compare_validation_results",
    "choose_threshold",
    "keep_match",
    "entity_decision",
    "decide_entity_match",
    "should_keep_match",
]
