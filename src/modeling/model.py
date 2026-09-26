from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.features.features import build_pair_features


def _mutate_name(name: Any):
    if name is None or pd.isna(name):
        return None
    text = str(name).strip()
    if not text:
        return None

    tokens = text.split()
    suffix_map = {
        "corp": "inc",
        "corporation": "llc",
        "company": "group",
        "llc": "inc",
        "inc": "group",
        "limited": "solutions",
        "ltd": "holdings",
        "pvt": "private",
    }

    mutated = text
    for old, new in suffix_map.items():
        if text.lower().endswith(old):
            mutated = text[: -len(old)] + new
            break
    else:
        if len(tokens) > 1:
            idx = max(0, len(tokens) - 1)
            mutated = " ".join(tokens[:idx] + ["group"])
        else:
            mutated = f"{text} group"
    return mutated


def _mutate_address(address: Any):
    if address is None or pd.isna(address):
        return None
    text = str(address).strip()
    if not text:
        return None

    lowered = text.lower()
    replacements = [
        ("street", "avenue"),
        ("st", "ave"),
        ("road", "lane"),
        ("rd", "ln"),
        ("ave", "road"),
        ("lane", "drive"),
        ("drive", "street"),
    ]
    for old, new in replacements:
        if old in lowered:
            return re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.IGNORECASE)
    if "," in text:
        left, rest = text.split(",", 1)
        return f"{left.replace(' ', ' ')} 9999, {rest.strip()}"
    return f"{text} Suite 999"


def generate_hard_negatives(candidate_pairs: pd.DataFrame, target_col: str = "label", max_per_row: int = 2):
    """Create near-miss negatives that are intentionally similar but not matching.

    This is intentionally designed to model the competition's most damaging error:
    false merges between highly similar businesses. The negatives preserve the same
    high-level structure as real records while altering only the discriminative bits.
    """
    if candidate_pairs is None or candidate_pairs.empty:
        return pd.DataFrame()

    data = candidate_pairs.copy()
    generated = []
    for _, row in data.iterrows():
        base = row.to_dict()
        base[target_col] = 0

        name_variant = _mutate_name(base.get("right_name")) or _mutate_name(base.get("left_name"))
        address_variant = _mutate_address(base.get("right_address")) or _mutate_address(base.get("left_address"))

        if name_variant is not None:
            neg_name = dict(base)
            neg_name["right_name"] = name_variant
            if "left_name" in neg_name and "right_name" in neg_name and neg_name["left_name"] == neg_name["right_name"]:
                neg_name["right_name"] = f"{neg_name['right_name']} group"
            generated.append(neg_name)

        if address_variant is not None:
            neg_address = dict(base)
            neg_address["right_address"] = address_variant
            generated.append(neg_address)

        if len(generated) >= max_per_row * max(1, len(data)):
            break

    if not generated:
        empty = data.copy()
        empty[target_col] = 0
        return empty

    result = pd.DataFrame(generated)
    if target_col not in result.columns:
        result[target_col] = 0
    if result.empty:
        return result

    result[target_col] = 0
    result = result.reindex(columns=data.columns.union(result.columns, sort=False))
    if target_col not in result.columns:
        result[target_col] = 0
    return result


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


def predict_pair_probabilities(
    model: Any,
    features_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Produces the standardized prediction output:
    source1_entity_id, candidate_entity_id, target_source, probability
    """
    if features_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "target_source", "probability"])

    # Determine ID/metadata columns
    s1_col = "source1_entity_id" if "source1_entity_id" in features_df.columns else "s1_id"
    cand_col = "candidate_entity_id" if "candidate_entity_id" in features_df.columns else ("target_id" if "target_id" in features_df.columns else "s2_id")
    src_col = "target_source" if "target_source" in features_df.columns else "source"

    meta_cols = {
        "source1_entity_id", "candidate_entity_id", "target_source",
        "s1_id", "s2_id", "s3_id", "target_id", "source", "label", "target", "is_match"
    }

    if feature_cols is not None:
        X = features_df[feature_cols]
    else:
        # Select numeric feature columns only
        num_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns if c not in meta_cols]
        X = features_df[num_cols]

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if hasattr(probs, "ndim") and probs.ndim == 2 and probs.shape[1] == 2:
            probabilities = probs[:, 1]
        else:
            probabilities = probs
    elif hasattr(model, "predict"):
        probabilities = model.predict(X)
    else:
        raise AttributeError("Model does not support predict_proba or predict")

    result = pd.DataFrame({
        "source1_entity_id": features_df[s1_col] if s1_col in features_df.columns else features_df.index,
        "candidate_entity_id": features_df[cand_col] if cand_col in features_df.columns else features_df.index,
        "target_source": features_df[src_col] if src_col in features_df.columns else "s2",
        "probability": probabilities,
    })
    return result


__all__ = [
    "generate_hard_negatives",
    "train_catboost_model",
    "train_lightgbm_model",
    "compare_validation_results",
    "train_baseline_models",
    "predict_pair_probabilities",
]
