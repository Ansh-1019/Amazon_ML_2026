import logging
from typing import Dict, Set, Tuple, Union
import pandas as pd


logger = logging.getLogger("pipeline.evaluation")


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculates F_beta score with default beta=0.5 prioritizing precision."""
    if precision <= 0 or recall <= 0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    return numerator / denominator if denominator > 0 else 0.0


class EntityEvaluator:
    """
    Evaluates Entity Resolution at the entity level for Source 1 reference entities.
    Expects predictions and ground truth in tuple/dictionary formats:
    s1_entity_id -> (s2_entity_id, s3_entity_id) or set of matched entities.
    """

    def __init__(self, beta: float = 0.5):
        self.beta = beta

    def evaluate(self, y_true: Dict[str, Tuple[str, str]], y_pred: Dict[str, Tuple[str, str]]) -> Dict[str, float]:
        """
        Computes Entity-level Precision, Recall, and F_beta (default F_0.5).

        Args:
            y_true: Dict mapping s1_id to ground truth tuple (s2_id, s3_id)
            y_pred: Dict mapping s1_id to predicted tuple (s2_id, s3_id)

        Returns:
            Dict containing 'precision', 'recall', 'f_beta'
        """
        if not y_true:
            logger.warning("Ground truth dictionary is empty.")
            return {"precision": 0.0, "recall": 0.0, f"f_{self.beta}": 0.0}

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        all_s1_keys = set(y_true.keys()).union(set(y_pred.keys()))

        for s1_id in all_s1_keys:
            gt_match = y_true.get(s1_id)
            pred_match = y_pred.get(s1_id)

            if pred_match is not None:
                if gt_match == pred_match:
                    true_positives += 1
                else:
                    false_positives += 1
                    if gt_match is not None:
                        false_negatives += 1
            else:
                if gt_match is not None:
                    false_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f_beta_score = compute_f_beta(precision, recall, beta=self.beta)

        results = {
            "precision": round(precision, 5),
            "recall": round(recall, 5),
            f"f_{self.beta}": round(f_beta_score, 5),
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives
        }
        logger.info(f"Evaluation Results: Precision={precision:.4f}, Recall={recall:.4f}, F_{self.beta}={f_beta_score:.4f}")
        return results
