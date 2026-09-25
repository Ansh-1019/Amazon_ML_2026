from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _as_float_array(values):
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _f05_score(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = 0.5 ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = beta_sq * precision + recall
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def choose_threshold(
    labels: Iterable[int | float | bool] | None,
    probabilities: Iterable[float] | None,
    *,
    threshold_candidates: Iterable[float] | None = None,
    positive_label: int | float | bool = 1,
):
    """Choose a probability threshold optimized for the competition's F_0.5 metric."""
    if labels is None or probabilities is None:
        return 0.5

    label_list = list(labels)
    prob_list = list(probabilities)
    if len(label_list) != len(prob_list):
        raise ValueError("labels and probabilities must have the same length.")
    if not label_list:
        return 0.5

    truth = np.asarray(label_list, dtype=float)
    scores = np.asarray(prob_list, dtype=float)
    if threshold_candidates is None:
        candidates = np.linspace(0.0, 1.0, 101)
    else:
        candidates = np.asarray(list(threshold_candidates), dtype=float)
        candidates = np.clip(candidates, 0.0, 1.0)

    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        pred = (scores >= float(threshold)).astype(int)
        actual = (truth == float(positive_label)).astype(int)
        tp = float(np.sum((pred == 1) & (actual == 1)))
        fp = float(np.sum((pred == 1) & (actual == 0)))
        fn = float(np.sum((pred == 0) & (actual == 1)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        score = _f05_score(precision, recall)

        tie_break = abs(float(threshold) - 0.5)
        if score > best_score or (np.isclose(score, best_score) and tie_break < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(threshold)

    return float(best_threshold)


def keep_match(probability, threshold: float = 0.5, *, singleton: bool = False):
    """Return whether a candidate pair should be kept as a match."""
    if singleton:
        return True
    if probability is None:
        return False
    try:
        prob = float(probability)
    except (TypeError, ValueError):
        return False
    if np.isnan(prob):
        return False
    return prob >= float(threshold)


def aggregate_entity_matches(
    pair_predictions: pd.DataFrame | None,
    *,
    entity_col: str = "source1_entity_id",
    candidate_col: str = "candidate_entity_id",
    score_col: str = "probability",
    threshold: float = 0.5,
):
    """Aggregate pairwise predictions into the final S1 -> list of matched IDs format."""
    if pair_predictions is None or pair_predictions.empty:
        return {}

    result: dict[str, list[str]] = {}
    entity_ids = set()
    for _, row in pair_predictions.iterrows():
        entity_id = row.get(entity_col)
        candidate_id = row.get(candidate_col)
        probability = row.get(score_col, 0.0)
        if pd.notna(entity_id):
            entity_ids.add(str(entity_id))
        if pd.notna(entity_id) and pd.notna(candidate_id) and keep_match(probability, threshold=threshold, singleton=False):
            result.setdefault(str(entity_id), []).append(str(candidate_id))

    for entity_id in entity_ids:
        result.setdefault(entity_id, [])

    normalized = {}
    for entity_id, matches in sorted(result.items()):
        seen = set()
        ordered = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                ordered.append(match)
        normalized[entity_id] = ordered

    missing_entities = [
        str(entity_id)
        for entity_id in sorted(set(pair_predictions[entity_col].dropna().astype(str)))
        if str(entity_id) not in normalized
    ]
    for entity_id in missing_entities:
        normalized[entity_id] = []

    return normalized


def entity_decision(
    pair_scores: Iterable[float] | None,
    *,
    threshold: float = 0.5,
    singleton: bool = False,
):
    """Convert pair-level match probabilities into an entity-level decision."""
    if singleton:
        return True
    if pair_scores is None:
        return False

    scores = _as_float_array(pair_scores)
    if scores.size == 0:
        return False
    return float(scores.max()) >= float(threshold)


def decide_entity_match(pair_scores: Iterable[float] | None, *, threshold: float = 0.5, singleton: bool = False):
    """Backward-compatible alias for entity-level decisions."""
    return entity_decision(pair_scores, threshold=threshold, singleton=singleton)


def should_keep_match(probability, threshold: float = 0.5, *, singleton: bool = False):
    """Backward-compatible alias for pair-level decisions."""
    return keep_match(probability, threshold=threshold, singleton=singleton)


__all__ = [
    "aggregate_entity_matches",
    "choose_threshold",
    "keep_match",
    "entity_decision",
    "decide_entity_match",
    "should_keep_match",
]
