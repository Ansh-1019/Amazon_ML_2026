import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


logger = logging.getLogger("pipeline.modeling.trainer")


class ModelTrainer:
    """Wrapper for training GBDT models (CatBoost / LightGBM) for match score prediction."""

    def __init__(self, model_type: str = "catboost", params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type.lower()
        self.params = params or {}
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> Any:
        """Fits the GBDT classifier on training feature matrix X_train and binary targets y_train."""
        logger.info(f"Training {self.model_type.upper()} model on shape {X_train.shape}...")

        if self.model_type == "catboost":
            if CatBoostClassifier is None:
                raise ImportError("CatBoost is not installed.")
            self.model = CatBoostClassifier(**self.params)
            eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=100)

        elif self.model_type == "lightgbm":
            if lgb is None:
                raise ImportError("LightGBM is not installed.")
            self.model = lgb.LGBMClassifier(**self.params)
            eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
            self.model.fit(X_train, y_train, eval_set=eval_set)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return self.model

    def save_model(self, save_path: Union[str, Path]) -> None:
        """Saves the trained model artifact."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model_type == "catboost" and hasattr(self.model, "save_model"):
            self.model.save_model(str(path))
        logger.info(f"Model saved to {path}")
