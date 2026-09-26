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
        if X is None or X.empty:
            return np.array([])

        X_feats = X.copy()
        if hasattr(self.model, "feature_names_") and self.model.feature_names_:
            expected = [c for c in self.model.feature_names_ if c in X_feats.columns]
            if len(expected) == len(self.model.feature_names_):
                X_feats = X_feats[self.model.feature_names_]
            else:
                X_feats = X_feats.select_dtypes(include=[np.number])
        else:
            meta_cols = {"source1_entity_id", "candidate_entity_id", "target_source", "s1_id", "s2_id", "s3_id", "target_id", "label"}
            feat_cols = [c for c in X_feats.columns if c not in meta_cols and np.issubdtype(X_feats[c].dtype, np.number)]
            if feat_cols:
                X_feats = X_feats[feat_cols]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_feats)
            if probs.ndim == 2 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        elif hasattr(self.model, "predict"):
            return self.model.predict(X_feats)
        else:
            raise AttributeError("Model does not support predict_proba or predict method.")

