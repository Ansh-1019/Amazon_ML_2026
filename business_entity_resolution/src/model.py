from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .features import build_pair_features


def _split_train_valid(data: pd.DataFrame, target_col: str = "label", test_size: float = 0.2, random_state: int = 42):
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' is not present in the candidate-pair data.")
    y = data[target_col].astype(int)
    if y.nunique() < 2:
        raise ValueError("At least one positive and one negative example are required for a binary classification baseline.")

    safe_test_size = float(test_size)
    if len(data) < 10:
        safe_test_size = 0.5
    elif safe_test_size <= 0:
        safe_test_size = 0.2

    train_idx, valid_idx = train_test_split(
        data.index,
        test_size=safe_test_size,
        stratify=y,
        random_state=random_state,
    )
    return data.loc[train_idx].copy(), data.loc[valid_idx].copy()


def _evaluate_model(model, X_valid: pd.DataFrame, y_valid: pd.Series):
    y_prob = model.predict_proba(X_valid)[:, 1]
    return {
        "val_auc": float(roc_auc_score(y_valid, y_prob)),
        "val_avg_precision": float(average_precision_score(y_valid, y_prob)),
        "val_accuracy": float(accuracy_score(y_valid, (y_prob >= 0.5).astype(int))),
        "val_log_loss": float(log_loss(y_valid, y_prob, labels=[0, 1])),
        "predictions": y_prob,
    }


def train_catboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int = 42,
    **kwargs,
):
    params = {
        "iterations": kwargs.pop("iterations", 300),
        "depth": kwargs.pop("depth", 6),
        "learning_rate": kwargs.pop("learning_rate", 0.05),
        "loss_function": kwargs.pop("loss_function", "Logloss"),
        "random_seed": kwargs.pop("random_seed", random_state),
        "verbose": kwargs.pop("verbose", False),
        "allow_writing_files": kwargs.pop("allow_writing_files", False),
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int = 42,
    **kwargs,
):
    params = {
        "n_estimators": kwargs.pop("n_estimators", 300),
        "learning_rate": kwargs.pop("learning_rate", 0.05),
        "max_depth": kwargs.pop("max_depth", 6),
        "objective": kwargs.pop("objective", "binary"),
        "random_state": kwargs.pop("random_state", random_state),
        "verbosity": kwargs.pop("verbosity", -1),
        "subsample": kwargs.pop("subsample", 0.9),
        "colsample_bytree": kwargs.pop("colsample_bytree", 0.9),
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def compare_validation_results(
    candidate_pairs: pd.DataFrame,
    target_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
):
    train_df, valid_df = _split_train_valid(candidate_pairs, target_col=target_col, test_size=test_size, random_state=random_state)

    X_train = build_pair_features(train_df, target_col=target_col)
    X_valid = build_pair_features(valid_df, target_col=target_col)

    y_train = X_train.pop(target_col)
    y_valid = X_valid.pop(target_col)

    catboost_model = train_catboost_model(X_train, y_train, X_valid, y_valid, random_state=random_state)
    lightgbm_model = train_lightgbm_model(X_train, y_train, X_valid, y_valid, random_state=random_state)

    catboost_scores = _evaluate_model(catboost_model, X_valid, y_valid)
    lightgbm_scores = _evaluate_model(lightgbm_model, X_valid, y_valid)

    comparison = {
        "catboost": {"val_auc": catboost_scores["val_auc"], "val_avg_precision": catboost_scores["val_avg_precision"], "val_accuracy": catboost_scores["val_accuracy"], "val_log_loss": catboost_scores["val_log_loss"]},
        "lightgbm": {"val_auc": lightgbm_scores["val_auc"], "val_avg_precision": lightgbm_scores["val_avg_precision"], "val_accuracy": lightgbm_scores["val_accuracy"], "val_log_loss": lightgbm_scores["val_log_loss"]},
    }
    return comparison


def train_baseline_models(
    candidate_pairs: pd.DataFrame,
    target_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
):
    train_df, valid_df = _split_train_valid(candidate_pairs, target_col=target_col, test_size=test_size, random_state=random_state)

    X_train = build_pair_features(train_df, target_col=target_col)
    X_valid = build_pair_features(valid_df, target_col=target_col)

    y_train = X_train.pop(target_col)
    y_valid = X_valid.pop(target_col)

    catboost_model = train_catboost_model(X_train, y_train, X_valid, y_valid, random_state=random_state)
    lightgbm_model = train_lightgbm_model(X_train, y_train, X_valid, y_valid, random_state=random_state)

    comparison = {
        "catboost": _evaluate_model(catboost_model, X_valid, y_valid),
        "lightgbm": _evaluate_model(lightgbm_model, X_valid, y_valid),
    }

    return {
        "catboost": catboost_model,
        "lightgbm": lightgbm_model,
        "comparison": comparison,
    }


__all__ = [
    "train_catboost_model",
    "train_lightgbm_model",
    "compare_validation_results",
    "train_baseline_models",
]
