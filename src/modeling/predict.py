import logging
from typing import Any
import numpy as np
import pandas as pd


logger = logging.getLogger("pipeline.modeling.predict")


class ModelPredictor:
    """Predicts match scores/probabilities for candidate pairs."""

    def __init__(self, model: Any):
        self.model = model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns probability of candidate pair being a true match."""
        if X.empty:
            return np.array([])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        elif hasattr(self.model, "predict"):
            return self.model.predict(X)
        else:
            raise AttributeError("Model does not support predict_proba or predict method.")
