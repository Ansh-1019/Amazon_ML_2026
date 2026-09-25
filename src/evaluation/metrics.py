"""
Entity-Level Macro Evaluation Module for Amazon ML Challenge 2026.

Implements the official competition evaluation metric:
  Entity-Level Macro-Averaged F_0.5.

Metric Definition:
  For every Source 1 entity i in the dataset:
    1. Ground-Truth match set: GT_i
    2. Predicted match set: Pred_i
    3. Set operations:
         TP_i = |Pred_i ∩ GT_i|
         FP_i = |Pred_i \\ GT_i|
         FN_i = |GT_i \\ Pred_i|
    4. Entity-level Precision & Recall:
         If GT_i == ∅ and Pred_i == ∅:
           Precision_i = 1.0, Recall_i = 1.0, F_beta_i = 1.0
         If GT_i == ∅ or Pred_i == ∅:
           Precision_i = 0.0, Recall_i = 0.0, F_beta_i = 0.0
         Else:
           Precision_i = TP_i / |Pred_i|
           Recall_i    = TP_i / |GT_i|
           F_beta_i    = (1 + beta^2) * Precision_i * Recall_i / ((beta^2 * Precision_i) + Recall_i)
    5. Macro-average over all N Source 1 entities:
         Macro F_beta = (1 / N) * ∑ F_beta_i
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

logger = logging.getLogger("pipeline.evaluation")


def parse_id_set(raw: Any) -> Set[str]:
    """
    Normalizes any input (str, list, tuple, set, None) into a deduplicated
    set of non-empty, stripped string IDs.
    """
    if raw is None:
        return set()
    if isinstance(raw, str):
        toks = raw.strip().strip("\"'").split(",")
        return set(t.strip().strip("\"'") for t in toks if t.strip().strip("\"'") and t.strip().lower() not in ["none", "nan"])
    if isinstance(raw, (list, tuple, set)):
        out = set()
        for item in raw:
            if item is not None:
                tok = str(item).strip().strip("\"'")
                if tok and tok.lower() not in ["none", "nan"]:
                    out.add(tok)
        return out
    return set()


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Calculates F_beta score given entity precision and recall.
    Default beta=0.5 prioritizes precision over recall (weighting precision 4x recall).
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    return numerator / denominator if denominator > 0 else 0.0


def compute_single_entity_metrics(
    gt_set: Set[str],
    pred_set: Set[str],
    beta: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes precision, recall, F_beta, and counts for a single Source 1 entity.
    
    Handles all boundary conditions:
      - GT = ∅, Pred = ∅  => F_beta = 1.0, Precision = 1.0, Recall = 1.0
      - GT ≠ ∅, Pred = ∅  => F_beta = 0.0, Precision = 0.0, Recall = 0.0
      - GT = ∅, Pred ≠ ∅  => F_beta = 0.0, Precision = 0.0, Recall = 0.0
      - Partial matches   => standard set precision & recall
    """
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    # Edge cases
    if len(gt_set) == 0 and len(pred_set) == 0:
        prec = 1.0
        rec = 1.0
        f_score = 1.0
        is_exact = True
    elif len(gt_set) == 0 or len(pred_set) == 0:
        prec = 0.0
        rec = 0.0
        f_score = 0.0
        is_exact = False
    else:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_score = compute_f_beta(prec, rec, beta=beta)
        is_exact = (pred_set == gt_set)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        f"f_{beta}": f_score,
        "is_exact_match": is_exact,
    }


class EntityEvaluator:
    """
    Evaluates Business Entity Resolution predictions against Ground Truth
    using the official entity-level macro-averaged F_0.5 metric.
    """

    def __init__(self, beta: float = 0.5):
        self.beta = beta

    def evaluate(
        self,
        y_true: Dict[str, Any],
        y_pred: Dict[str, Any],
        all_s1_ids: Optional[Union[List[str], Set[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Computes Entity-level Macro Precision, Macro Recall, Macro F_beta, and full diagnostics.

        Args:
            y_true: Dict mapping s1_id -> ground truth matched IDs (set/list/tuple/str).
            y_pred: Dict mapping s1_id -> predicted matched IDs (set/list/tuple/str).
            all_s1_ids: Optional list/set of all Source 1 entity IDs in evaluation split.
                        If provided, guarantees every entity is evaluated even if missing from dicts.

        Returns:
            Dict containing:
              - macro_f_beta / f_0.5
              - macro_precision
              - macro_recall
              - number_of_entities
              - exact_match_rate
              - empty_gt_count
              - correctly_empty_count
              - false_positive_empty_count
              - entities_with_false_positives
              - entities_with_false_negatives
              - per_entity_records (List of dicts for error analysis)
        """
        # Determine universal set of Source 1 entity IDs
        if all_s1_ids is not None:
            s1_keys = sorted(list(set(all_s1_ids)))
        else:
            s1_keys = sorted(list(set(y_true.keys()).union(set(y_pred.keys()))))

        num_entities = len(s1_keys)
        if num_entities == 0:
            logger.warning("No Source 1 entities to evaluate.")
            return {
                f"f_{self.beta}": 0.0,
                "macro_f_beta": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "number_of_entities": 0,
                "exact_match_rate": 0.0,
                "empty_gt_count": 0,
                "correctly_empty_count": 0,
                "false_positive_empty_count": 0,
                "entities_with_false_positives": 0,
                "entities_with_false_negatives": 0,
                "per_entity_records": [],
            }

        per_entity_records: List[Dict[str, Any]] = []
        f_scores: List[float] = []
        precisions: List[float] = []
        recalls: List[float] = []

        exact_matches = 0
        empty_gt_count = 0
        correctly_empty_count = 0
        false_positive_empty_count = 0
        entities_with_fp = 0
        entities_with_fn = 0

        for s1_id in s1_keys:
            gt_set = parse_id_set(y_true.get(s1_id))
            pred_set = parse_id_set(y_pred.get(s1_id))

            metrics = compute_single_entity_metrics(gt_set, pred_set, beta=self.beta)

            prec = metrics["precision"]
            rec = metrics["recall"]
            f_val = metrics[f"f_{self.beta}"]

            f_scores.append(f_val)
            precisions.append(prec)
            recalls.append(rec)

            if len(gt_set) == 0:
                empty_gt_count += 1
                if len(pred_set) == 0:
                    correctly_empty_count += 1
                else:
                    false_positive_empty_count += 1

            if metrics["is_exact_match"]:
                exact_matches += 1
            if metrics["fp"] > 0:
                entities_with_fp += 1
            if metrics["fn"] > 0:
                entities_with_fn += 1

            per_entity_records.append({
                "s1_id": s1_id,
                "ground_truth": sorted(list(gt_set)),
                "prediction": sorted(list(pred_set)),
                "precision": round(prec, 5),
                "recall": round(rec, 5),
                f"f_{self.beta}": round(f_val, 5),
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "is_exact_match": metrics["is_exact_match"],
            })

        macro_f_beta = float(np.mean(f_scores))
        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
        exact_match_rate = exact_matches / num_entities

        results = {
            f"f_{self.beta}": round(macro_f_beta, 5),
            "macro_f_beta": round(macro_f_beta, 5),
            "precision": round(macro_precision, 5),
            "macro_precision": round(macro_precision, 5),
            "recall": round(macro_recall, 5),
            "macro_recall": round(macro_recall, 5),
            "number_of_entities": num_entities,
            "exact_match_rate": round(exact_match_rate, 5),
            "empty_gt_count": empty_gt_count,
            "correctly_empty_count": correctly_empty_count,
            "false_positive_empty_count": false_positive_empty_count,
            "entities_with_false_positives": entities_with_fp,
            "entities_with_false_negatives": entities_with_fn,
            "per_entity_records": per_entity_records,
        }

        logger.info(
            f"Evaluation [N={num_entities}]: "
            f"Macro F_{self.beta}={macro_f_beta:.4f}, "
            f"Macro Precision={macro_precision:.4f}, "
            f"Macro Recall={macro_recall:.4f}, "
            f"Exact Match Rate={exact_match_rate:.4f}"
        )
        return results
