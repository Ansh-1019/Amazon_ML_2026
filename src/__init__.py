"""
Amazon ML Challenge 2026 - Business Entity Resolution Pipeline Package
"""

from .pipeline import EntityResolutionPipeline
from .predict import run_inference

__version__ = "0.1.0"
__all__ = ["EntityResolutionPipeline", "run_inference"]

