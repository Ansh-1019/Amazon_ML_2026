from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

import numpy as np
import pandas as pd


def _clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _safe_numeric(value):
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _string_similarity(left, right):
    left_text = _clean_text(left)
    right_text = _clean_text(right)
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def _token_overlap(left, right):
    left_tokens = set(_clean_text(left).split())
    right_tokens = set(_clean_text(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def _extract_pair_columns(columns: Iterable[str]):
    normalized = {}
    for column in columns:
        base = str(column).strip()
        lowered = base.lower()
        for prefix, replacement in [
            ("left_", ""),
            ("right_", ""),
            ("source_", ""),
            ("target_", ""),
            ("entity1_", ""),
            ("entity2_", ""),
            ("a_", ""),
            ("b_", ""),
            ("first_", ""),
            ("second_", ""),
            ("l_", ""),
            ("r_", ""),
        ]:
            if lowered.startswith(prefix):
                base = base[len(prefix):]
                lowered = lowered[len(prefix):]
                break
        for suffix, replacement in [
            ("_left", ""),
            ("_right", ""),
            ("_source", ""),
            ("_target", ""),
            ("_entity1", ""),
            ("_entity2", ""),
            ("_a", ""),
            ("_b", ""),
            ("_first", ""),
            ("_second", ""),
            ("_l", ""),
            ("_r", ""),
        ]:
            if lowered.endswith(suffix):
                base = base[: -len(suffix)]
                lowered = lowered[: -len(suffix)]
                break
        if not base:
            continue
        normalized.setdefault(base, []).append(column)
    return normalized


def _build_numeric_comparison(left_series, right_series):
    left = left_series.apply(_safe_numeric)
    right = right_series.apply(_safe_numeric)
    mask = left.notna() & right.notna()
    diff = np.full(len(left), np.nan, dtype=float)
    diff[mask] = np.abs(left[mask].to_numpy(dtype=float) - right[mask].to_numpy(dtype=float))
    ratio = np.full(len(left), np.nan, dtype=float)
    valid = mask & (left != 0)
    ratio[valid] = np.abs((left[valid].to_numpy(dtype=float) - right[valid].to_numpy(dtype=float)) / left[valid].to_numpy(dtype=float))
    same = np.full(len(left), 0.0, dtype=float)
    same[mask] = (left[mask].to_numpy(dtype=float) == right[mask].to_numpy(dtype=float)).astype(float)
    return {
        "numeric_diff": diff,
        "numeric_ratio": ratio,
        "numeric_same": same,
    }


def _build_text_comparison(left_series, right_series):
    similarity = left_series.combine(right_series, _string_similarity).to_numpy(dtype=float)
    overlap = left_series.combine(right_series, _token_overlap).to_numpy(dtype=float)
    exact = (left_series.astype(str).str.lower().fillna("") == right_series.astype(str).str.lower().fillna("")).astype(float).to_numpy()
    return {
        "text_similarity": similarity,
        "text_overlap": overlap,
        "text_exact": exact,
    }


def build_pair_features(candidate_pairs: pd.DataFrame, target_col: str | None = None):
    """Build a feature matrix for candidate pair matching.

    The function is intentionally simple: for each pair of like-named columns
    (e.g. left_name/right_name, source_city/target_city), it generates candidate
    similarity and equality features. This keeps the baseline model easy to
    reason about before adding more advanced matching logic.
    """
    if candidate_pairs is None or candidate_pairs.empty:
        raise ValueError("candidate_pairs must be a non-empty pandas DataFrame.")

    data = candidate_pairs.copy()
    if target_col is not None and target_col in data.columns:
        data = data.copy()
    feature_frame = pd.DataFrame(index=data.index)

    grouped = _extract_pair_columns(data.columns)
    explicit_pair_columns = []
    for base, columns in grouped.items():
        if len(columns) < 2:
            continue
        filtered = [col for col in columns if col != target_col]
        if len(filtered) < 2:
            continue
        left_candidates = [
            col
            for col in filtered
            if str(col).lower().startswith(("left_", "source_", "entity1_", "a_", "first_", "l_"))
        ]
        right_candidates = [
            col
            for col in filtered
            if str(col).lower().startswith(("right_", "target_", "entity2_", "b_", "second_", "r_"))
        ]
        if left_candidates and right_candidates:
            left_col = left_candidates[0]
            right_col = right_candidates[0]
            explicit_pair_columns.append((base, left_col, right_col))
        elif len(filtered) >= 2:
            left_col, right_col = filtered[:2]
            explicit_pair_columns.append((base, left_col, right_col))

    if not explicit_pair_columns:
        for column in data.columns:
            if target_col is not None and column == target_col:
                continue
            feature_frame[f"{column}_raw"] = data[column].apply(_safe_numeric)
        if target_col is not None and target_col in data.columns:
            feature_frame[target_col] = data[target_col].astype(int)
        return feature_frame

    for base, left_col, right_col in explicit_pair_columns:
        left_series = data[left_col]
        right_series = data[right_col]

        numeric = _build_numeric_comparison(left_series, right_series)
        text = _build_text_comparison(left_series, right_series)

        feature_frame[f"{base}_numeric_diff"] = numeric["numeric_diff"]
        feature_frame[f"{base}_numeric_ratio"] = numeric["numeric_ratio"]
        feature_frame[f"{base}_numeric_same"] = numeric["numeric_same"]
        feature_frame[f"{base}_text_similarity"] = text["text_similarity"]
        feature_frame[f"{base}_text_overlap"] = text["text_overlap"]
        feature_frame[f"{base}_text_exact"] = text["text_exact"]

    # Add a simple "all-equal" overall quality score for the case where an exact
    # match across all pair features is present.
    if feature_frame.shape[1] > 0:
        feature_frame["pair_quality_score"] = feature_frame.mean(axis=1, skipna=True)

    if target_col is not None and target_col in data.columns:
        feature_frame[target_col] = data[target_col].astype(int)

    return feature_frame.fillna(0.0)
