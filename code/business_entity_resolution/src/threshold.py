from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_float_array(values):
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def choose_threshold(
    labels: Iterable[int | float | bool] | None,
    probabilities: Iterable[float] | None,
    *,
    threshold_candidates: Iterable[float] | None = None,
    positive_label: int | float | bool = 1,
):
    """Choose a probability threshold for binary matching decisions."""
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
        if precision == 0.0 and recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        tie_break = abs(float(threshold) - 0.5)
        if f1 > best_score or (np.isclose(f1, best_score) and tie_break < abs(best_threshold - 0.5)):
            best_score = f1
            best_threshold = float(threshold)

    return float(best_threshold)


def keep_match(probability, threshold: float = 0.5, *, singleton: bool = False):
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


def entity_decision(pair_scores: Iterable[float] | None, *, threshold: float = 0.5, singleton: bool = False):
    if singleton:
        return True
    if pair_scores is None:
        return False

    scores = _as_float_array(pair_scores)
    if scores.size == 0:
        return False
    return float(scores.max()) >= float(threshold)


def decide_entity_match(pair_scores: Iterable[float] | None, *, threshold: float = 0.5, singleton: bool = False):
    return entity_decision(pair_scores, threshold=threshold, singleton=singleton)


def should_keep_match(probability, threshold: float = 0.5, *, singleton: bool = False):
    return keep_match(probability, threshold=threshold, singleton=singleton)


__all__ = [
    "choose_threshold",
    "keep_match",
    "entity_decision",
    "decide_entity_match",
    "should_keep_match",
]
