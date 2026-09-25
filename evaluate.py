from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Iterable, Mapping

import pandas as pd


def _as_record_set(values: Any) -> set[Any]:
    if values is None:
        return set()
    if isinstance(values, (str, bytes)):
        return {values}
    if isinstance(values, pd.Index):
        return set(values.tolist())
    if isinstance(values, pd.Series):
        return set(values.tolist())
    if isinstance(values, (set, frozenset, tuple, list)):
        return set(values)
    try:
        return set(values)
    except TypeError:
        return {values}


def f05_score(true_set: Iterable[Any], pred_set: Iterable[Any]) -> float:
    """Compute the precision-heavy F0.5 score for a single S1 entity.

    Precision and recall are evaluated against the records belonging to one true
    entity cluster. A false merge (predicted cluster contains records from other
    S1 entities) strongly reduces precision because F_0.5 weights precision more
    than recall.
    """
    true_members = set(true_set)
    pred_members = set(pred_set)
    if not true_members and not pred_members:
        return 1.0

    tp = len(true_members & pred_members)
    fp = len(pred_members - true_members)
    fn = len(true_members - pred_members)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_sq = 0.5 ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = beta_sq * precision + recall
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def _invert_record_to_group(record_to_group: Mapping[Any, Any]) -> dict[Any, set[Any]]:
    inverted: dict[Any, set[Any]] = defaultdict(set)
    for record_id, group_id in record_to_group.items():
        inverted[group_id].add(record_id)
    return {group_id: set(records) for group_id, records in inverted.items()}


def _coerce_group_mapping(groups: Any, *, record_col: str | None = None, entity_col: str | None = None, pred_col: str | None = None) -> dict[Any, set[Any]]:
    if groups is None:
        return {}

    if isinstance(groups, Mapping):
        values = list(groups.values())
        if values and all(not isinstance(v, (str, bytes, dict)) and hasattr(v, "__iter__") for v in values):
            return {group_id: set(_as_record_set(records)) for group_id, records in groups.items()}
        return _invert_record_to_group(groups)

    if isinstance(groups, pd.DataFrame):
        if record_col is None:
            candidate_cols = ["record_id", "id", "record", "entity_id", "s1_id", "key"]
            record_col = next((col for col in candidate_cols if col in groups.columns), None)
        if pred_col is None and entity_col is None:
            candidate_cols = ["cluster_id", "prediction_id", "group_id", "pred_cluster_id", "label", "entity"]
            pred_col = next((col for col in candidate_cols if col in groups.columns), None)
        if record_col is None:
            if groups.index.nlevels == 1:
                record_col = groups.index.name or "record_id"
            else:
                raise ValueError("Could not infer the record identifier column from the DataFrame input.")

        if pred_col is None:
            if entity_col is not None and entity_col in groups.columns:
                pred_col = entity_col
            else:
                raise ValueError("Could not infer the grouping column from the DataFrame input.")

        if entity_col is not None and entity_col in groups.columns and pred_col == entity_col:
            # This is a record-to-entity mapping already; invert it.
            return _invert_record_to_group(groups.set_index(record_col)[entity_col].to_dict())

        if record_col in groups.columns and pred_col in groups.columns:
            result: dict[Any, set[Any]] = defaultdict(set)
            for record_id, group_id in zip(groups[record_col].tolist(), groups[pred_col].tolist()):
                result[group_id].add(record_id)
            return {group_id: set(records) for group_id, records in result.items()}

        # Fallback: treat the DataFrame as record->group mapping with the index.
        return _invert_record_to_group(groups[groups.columns[0]].to_dict())

    if isinstance(groups, Iterable) and not isinstance(groups, (str, bytes)):
        groups_list = list(groups)
        if not groups_list:
            return {}
        first = groups_list[0]
        if isinstance(first, tuple) and len(first) == 2:
            return _invert_record_to_group(dict(groups_list))
        if isinstance(first, Mapping):
            return _invert_record_to_group({record_id: group_id for mapping in groups_list for record_id, group_id in mapping.items()})
        raise TypeError("Unsupported iterable input; expected a mapping or a list of (record_id, group_id) pairs.")

    return {groups: {groups}}


def _resolve_true_and_pred_groups(
    true_groups: Any,
    pred_groups: Any,
    *,
    record_col: str | None = None,
    entity_col: str | None = None,
    pred_col: str | None = None,
) -> tuple[dict[Any, set[Any]], dict[Any, set[Any]]]:
    true_mapping = _coerce_group_mapping(true_groups, record_col=record_col, entity_col=entity_col, pred_col=pred_col)
    pred_mapping = _coerce_group_mapping(pred_groups, record_col=record_col, entity_col=entity_col, pred_col=pred_col)
    return true_mapping, pred_mapping


def evaluate(
    true_groups: Any,
    pred_groups: Any,
    *,
    record_col: str | None = None,
    entity_col: str | None = None,
    pred_col: str | None = None,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Compute the challenge metric: macro-average of per-S1 F_0.5 scores.

    Each true S1 entity is treated as a ground-truth cluster of records. For each
    entity, the corresponding predicted cluster is the union of all prediction
    clusters that overlap with that entity. This makes false merges penalized
    naturally by the precision-heavy F_0.5 score.
    """
    true_mapping, pred_mapping = _resolve_true_and_pred_groups(
        true_groups,
        pred_groups,
        record_col=record_col,
        entity_col=entity_col,
        pred_col=pred_col,
    )

    if not true_mapping:
        return {"score": 0.0, "per_entity": {}} if return_details else 0.0

    per_entity_scores: dict[Any, float] = {}
    for entity_id, true_members in true_mapping.items():
        if not true_members:
            continue

        predicted_members = set()
        for predicted_cluster_id, cluster_members in pred_mapping.items():
            if true_members & cluster_members:
                predicted_members |= cluster_members

        per_entity_scores[entity_id] = f05_score(true_members, predicted_members)

    score = float(sum(per_entity_scores.values()) / len(per_entity_scores)) if per_entity_scores else 0.0
    if return_details:
        return {"score": score, "per_entity": per_entity_scores}
    return score


def macro_f05_s1(*args: Any, **kwargs: Any) -> float:
    """Backward-compatible alias for the official challenge metric."""
    return evaluate(*args, **kwargs)


def score(*args: Any, **kwargs: Any) -> float:
    return evaluate(*args, **kwargs)


def compute_metric(*args: Any, **kwargs: Any) -> float:
    return evaluate(*args, **kwargs)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate S1 entity resolution with the challenge F_0.5 metric.")
    parser.add_argument("true", help="Ground-truth mapping or file path containing true group assignments.")
    parser.add_argument("pred", help="Prediction mapping or file path containing predicted group assignments.")
    parser.add_argument("--record-col", default=None, help="Record ID column name for DataFrame inputs.")
    parser.add_argument("--entity-col", default=None, help="True entity column name for DataFrame inputs.")
    parser.add_argument("--pred-col", default=None, help="Predicted cluster column name for DataFrame inputs.")
    args = parser.parse_args()

    true_value = args.true
    pred_value = args.pred
    if true_value.endswith((".csv", ".parquet")):
        true_value = pd.read_csv(true_value)
    if pred_value.endswith((".csv", ".parquet")):
        pred_value = pd.read_csv(pred_value)

    result = evaluate(
        true_value,
        pred_value,
        record_col=args.record_col,
        entity_col=args.entity_col,
        pred_col=args.pred_col,
    )
    print(result)


if __name__ == "__main__":
    _cli()


__all__ = [
    "f05_score",
    "evaluate",
    "macro_f05_s1",
    "score",
    "compute_metric",
]
