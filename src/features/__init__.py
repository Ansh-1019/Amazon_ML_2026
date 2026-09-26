"""Feature extraction subpackage for Entity Resolution."""
from src.features.features import build_pair_features
from src.features.pairwise import FeatureExtractor

__all__ = [
    "build_pair_features",
    "FeatureExtractor",
]
