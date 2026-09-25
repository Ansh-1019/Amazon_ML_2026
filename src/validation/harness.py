"""
Validation Harness for Amazon ML Challenge 2026 — Business Entity Resolution.

Answers the core question:
  "Did this change actually improve our Amazon ML Challenge score?"

Architecture
------------
ValidationHarness wraps the existing pipeline, computes all competition-
relevant metrics, and produces a structured ValidationReport that can be:
  - Printed to the terminal (human-readable)
  - Saved as JSON (machine-readable, diff-able between experiments)
  - Logged to ExperimentTracker (persistent run history)

The harness is deliberately read-only with respect to the core matching
algorithm — it only orchestrates data loading, pipeline execution, and
post-run analysis.

Usage (programmatic)
---------------------
from src.validation.harness import ValidationHarness

harness = ValidationHarness(config_path="configs/default_config.yaml")
report  = harness.run(
    s1_df=..., s2_df=..., s3_df=...,
    ground_truth=...,           # Dict[str, Set[str]]
    threshold=0.5,
    experiment_id="exp42_tfidf_blocker",
)
report.print_summary()
report.save_json("experiments/exp42_tfidf_blocker/report.json")
"""

import json
import time
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.utils import setup_logger, load_config, ExperimentTracker
from src.pipeline import EntityResolutionPipeline, candidates_df_to_map, set_seed
from src.evaluation.metrics import EntityEvaluator, parse_id_set

logger = logging.getLogger("validation.harness")


# ─────────────────────────────────────────────────────────────────────────────
# ValidationReport  — structured, JSON-serialisable result of one harness run
# ─────────────────────────────────────────────────────────────────────────────

class ValidationReport:
    """
    Immutable snapshot of a single validation run.

    Sections:
      run_info        — experiment ID, timestamp, configuration snapshot
      data_stats      — entity counts, country distribution, source sizes
      blocking_stats  — recall, candidate counts (mean/median/p95)
      matching_stats  — zero/one/multi-match distribution
      metrics         — macro F0.5, precision, recall, exact-match rate
      error_analysis  — per-entity false-positive / false-negative lists
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def experiment_id(self) -> str:
        return self._data["run_info"]["experiment_id"]

    @property
    def macro_f05(self) -> float:
        return self._data["metrics"].get("macro_f0.5", 0.0)

    @property
    def blocking_recall(self) -> float:
        return self._data["blocking_stats"].get("blocking_recall", 0.0)

    @property
    def as_dict(self) -> Dict[str, Any]:
        return self._data

    # ── Serialisation ─────────────────────────────────────────────────────

    def save_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=indent, default=str)
        logger.info(f"Validation report saved -> {out}")
        return out

    # ── Human-readable summary ─────────────────────────────────────────────

    def print_summary(self) -> None:
        d = self._data
        ri = d["run_info"]
        ds = d["data_stats"]
        bs = d["blocking_stats"]
        ms = d["matching_stats"]
        mt = d["metrics"]
        ea = d["error_analysis"]

        w = 56
        sep = "-" * w
        print(f"\n{'=' * w}")
        print(f"  VALIDATION REPORT   [{ri['experiment_id']}]")
        print(f"{'=' * w}")
        print(f"  Timestamp : {ri['timestamp']}")
        print(f"  Threshold : {ri['threshold']}")
        print(f"  Run dir   : {ri.get('run_dir', 'N/A')}")

        print(f"\n  {'DATA':-<{w - 2}}")
        print(f"  Source 1 entities        : {ds['n_s1']:>8,}")
        print(f"  Source 2 entities        : {ds['n_s2']:>8,}")
        print(f"  Source 3 entities        : {ds['n_s3']:>8,}")
        print(f"  Countries in S1          : {', '.join(ds['countries_s1'])}")
        print(f"  GT entities w/ matches   : {ds['gt_entities_with_matches']:>8,}")
        print(f"  GT entities w/o matches  : {ds['gt_entities_without_matches']:>8,}")

        print(f"\n  {'BLOCKING':-<{w - 2}}")
        print(f"  Total candidates         : {bs['total_candidates']:>8,}")
        print(f"  Mean  cands / S1         : {bs['mean_candidates_per_s1']:>8.2f}")
        print(f"  Median cands / S1        : {bs['median_candidates_per_s1']:>8.2f}")
        print(f"  P95   cands / S1         : {bs['p95_candidates_per_s1']:>8.2f}")
        print(f"  Blocking recall          : {bs['blocking_recall']:>8.4f}")
        print(f"  Entities w/ zero cands   : {bs['entities_with_zero_candidates']:>8,}")
        print(f"  GT pairs missing in cands: {bs['gt_pairs_missing_in_candidates']:>8,}")

        print(f"\n  {'MATCHING':-<{w - 2}}")
        print(f"  Entities w/ zero matches : {ms['entities_zero_matches']:>8,}")
        print(f"  Entities w/ one match    : {ms['entities_one_match']:>8,}")
        print(f"  Entities w/ multi matches: {ms['entities_multi_matches']:>8,}")
        print(f"  Avg matches / entity     : {ms['avg_matches_per_entity']:>8.3f}")

        print(f"\n  {'METRICS':-<{w - 2}}")
        print(f"  Macro Precision          : {mt['macro_precision']:>8.4f}")
        print(f"  Macro Recall             : {mt['macro_recall']:>8.4f}")
        print(f"  Macro F0.5  (HEADLINE)   : {mt['macro_f0.5']:>8.4f}")
        print(f"  Exact entity match rate  : {mt['exact_match_rate']:>8.4f}")
        print(f"  Correctly-empty rate     : {mt['correctly_empty_rate']:>8.4f}")
        print(f"  False-positive-empty rate: {mt['false_positive_empty_rate']:>8.4f}")

        print(f"\n  {'ERROR ANALYSIS':-<{w - 2}}")
        print(f"  FP entities (pred not GT): {len(ea['false_positive_entities']):>8,}")
        print(f"  FN entities (GT not pred): {len(ea['false_negative_entities']):>8,}")
        print(f"  GT pairs missing in cands: {len(ea['missed_candidate_entities']):>8,}")
        print(f"  High-conf FP matches     : {len(ea['high_confidence_false_matches']):>8,}")

        if ea["false_positive_entities"]:
            sample = ea["false_positive_entities"][:5]
            print(f"\n  Sample FP entities: {sample}")
        if ea["false_negative_entities"]:
            sample = ea["false_negative_entities"][:5]
            print(f"  Sample FN entities: {sample}")

        print(f"{'=' * w}\n")

    # ── Comparison helper ──────────────────────────────────────────────────

    def compare(self, other: "ValidationReport") -> Dict[str, float]:
        """Returns delta (self - other) for key metrics. Positive = improvement."""
        mt_a = self._data["metrics"]
        mt_b = other._data["metrics"]
        return {
            "delta_macro_f0.5":     mt_a["macro_f0.5"]     - mt_b["macro_f0.5"],
            "delta_macro_precision": mt_a["macro_precision"] - mt_b["macro_precision"],
            "delta_macro_recall":   mt_a["macro_recall"]    - mt_b["macro_recall"],
            "delta_blocking_recall": self.blocking_recall    - other.blocking_recall,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ValidationHarness
# ─────────────────────────────────────────────────────────────────────────────

class ValidationHarness:
    """
    Orchestrates a full validation run against a ground-truth labelled split.

    The harness:
      1. Accepts data directly (DataFrames) OR loads from files via config.
      2. Runs the 6-stage pipeline internally.
      3. Computes all metric sections.
      4. Returns a ValidationReport.
      5. Logs to ExperimentTracker and saves report JSON to the run dir.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = "configs/default_config.yaml",
        experiments_dir: str = "experiments",
    ):
        if config is not None:
            self.config = config
        elif config_path and Path(config_path).exists():
            self.config = load_config(config_path)
        else:
            self.config = self._default_config()

        self.experiments_dir = experiments_dir

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        ground_truth: Dict[str, Any],          # s1_id → set/list/str of GT target IDs
        s1_df: Optional[pd.DataFrame] = None,
        s2_df: Optional[pd.DataFrame] = None,
        s3_df: Optional[pd.DataFrame] = None,
        experiment_id: Optional[str] = None,
        threshold: Optional[float] = None,
        output_dir: str = "output",
        save_report: bool = True,
    ) -> ValidationReport:
        """
        Executes the full pipeline on the supplied data + ground truth,
        then builds and returns a ValidationReport.

        Args:
            ground_truth : Dict mapping s1_id → GT match IDs (any iterable format).
            s1_df        : Source 1 DataFrame (or None to load from config paths).
            s2_df        : Source 2 DataFrame.
            s3_df        : Source 3 DataFrame.
            experiment_id: Human-readable run label (default: config project name).
            threshold    : Decision threshold override.
            output_dir   : Where to write TSV submission files.
            save_report  : If True, write report.json next to summary.json.

        Returns:
            ValidationReport with all metric sections populated.
        """
        exp_id = experiment_id or self.config.get("project", {}).get("experiment_id", "val_run")
        thr = threshold if threshold is not None else self.config.get("decision", {}).get("threshold", 0.5)

        # ── ExperimentTracker setup ──────────────────────────────────────
        tracker = ExperimentTracker(experiment_id=exp_id, experiments_dir=self.experiments_dir)
        run_logger = setup_logger(
            name="validation", log_file=tracker.get_run_dir() / "validation.log"
        )
        run_logger.info(f"=== Validation Run [{exp_id}] threshold={thr} ===")

        # ── Normalise ground truth ───────────────────────────────────────
        gt: Dict[str, Set[str]] = {
            k: parse_id_set(v) for k, v in ground_truth.items()
        }

        # ── Build pipeline config with threshold override ─────────────────
        run_config = {
            **self.config,
            "project": {**self.config.get("project", {}), "experiment_id": exp_id},
            "decision": {**self.config.get("decision", {}), "threshold": thr},
            "paths": {**self.config.get("paths", {}), "submissions_dir": output_dir},
        }
        seed = run_config.get("project", {}).get("seed", 42)
        set_seed(seed)

        # ── Run pipeline ─────────────────────────────────────────────────
        t0 = time.time()
        pipeline = EntityResolutionPipeline(config=run_config)
        pipeline_results = pipeline.run(
            s1_df=s1_df, s2_df=s2_df, s3_df=s3_df,
            train_matches=None,   # ground truth is handled by harness, not pipeline
            threshold=thr,
            output_dir=output_dir,
        )
        elapsed = time.time() - t0

        # ── Collect pipeline internals (re-run Stage 1+2 for stats) ──────
        s1_norm, s2_norm, s3_norm, _, all_s1_ids, valid_target_ids = (
            pipeline.load_and_normalize_data(s1_df=s1_df, s2_df=s2_df, s3_df=s3_df)
        )
        candidates_df, candidate_map = pipeline.generate_candidates(
            s1_norm, s2_norm, s3_norm, all_s1_ids, valid_target_ids
        )

        # ── Build prediction map from saved TSV ──────────────────────────
        match_map = self._load_match_map_from_tsv(
            pipeline_results["matching_results_path"], all_s1_ids
        )

        # ── Compute all sections ──────────────────────────────────────────
        data_section   = self._compute_data_stats(s1_norm, s2_norm, s3_norm, gt)
        blocking_sec   = self._compute_blocking_stats(candidate_map, gt)
        matching_sec   = self._compute_matching_stats(match_map, all_s1_ids)
        metrics_sec, per_entity = self._compute_metrics(gt, match_map, all_s1_ids)
        error_sec      = self._compute_error_analysis(
            gt, match_map, candidate_map, per_entity
        )

        # ── Assemble report data ──────────────────────────────────────────
        report_data: Dict[str, Any] = {
            "run_info": {
                "experiment_id": exp_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_seconds": round(elapsed, 3),
                "threshold": thr,
                "run_dir": str(tracker.get_run_dir()),
                "matching_results_path": pipeline_results["matching_results_path"],
                "candidate_pairs_path": pipeline_results["candidate_pairs_path"],
                "is_submission_valid": pipeline_results["is_submission_valid"],
            },
            "config_snapshot": {
                "blocking":  run_config.get("blocking", {}),
                "features":  run_config.get("features", {}),
                "modeling":  run_config.get("modeling", {}),
                "decision":  run_config.get("decision", {}),
                "seed":      seed,
            },
            "data_stats":    data_section,
            "blocking_stats": blocking_sec,
            "matching_stats": matching_sec,
            "metrics":       metrics_sec,
            "error_analysis": error_sec,
        }

        report = ValidationReport(report_data)

        # ── Persist ───────────────────────────────────────────────────────
        tracker.log_params(run_config)
        tracker.log_metrics({
            "macro_f0.5":     metrics_sec["macro_f0.5"],
            "macro_precision": metrics_sec["macro_precision"],
            "macro_recall":   metrics_sec["macro_recall"],
            "blocking_recall": blocking_sec["blocking_recall"],
            "threshold":       thr,
        })

        if save_report:
            report.save_json(tracker.get_run_dir() / "report.json")

        run_logger.info(
            f"Validation complete: macro_F0.5={metrics_sec['macro_f0.5']:.4f}, "
            f"blocking_recall={blocking_sec['blocking_recall']:.4f}, "
            f"elapsed={elapsed:.2f}s"
        )
        return report

    # ── Private computation helpers ───────────────────────────────────────

    def _load_match_map_from_tsv(
        self, tsv_path: str, all_s1_ids: List[str]
    ) -> Dict[str, List[str]]:
        """Reads matching_results.tsv and returns {s1_id: [matched_id, ...]}."""
        match_map: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ids}
        try:
            df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
            for _, row in df.iterrows():
                s1_id = str(row.iloc[0]).strip()
                raw_ids = str(row.iloc[1]).strip() if len(row) > 1 else ""
                ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
                if s1_id in match_map:
                    match_map[s1_id] = ids
        except Exception as exc:
            logger.warning(f"Could not load match map from {tsv_path}: {exc}")
        return match_map

    def _compute_data_stats(
        self,
        s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
        gt: Dict[str, Set[str]],
    ) -> Dict[str, Any]:
        id_col = self.config.get("data", {}).get("id_column_s1", "entity_id")
        country_col = "country"
        countries_s1 = (
            sorted(s1[country_col].dropna().unique().tolist())
            if country_col in s1.columns else []
        )
        gt_with    = sum(1 for v in gt.values() if v)
        gt_without = sum(1 for v in gt.values() if not v)
        return {
            "n_s1": len(s1),
            "n_s2": len(s2),
            "n_s3": len(s3),
            "countries_s1": countries_s1,
            "gt_entities_total": len(gt),
            "gt_entities_with_matches": gt_with,
            "gt_entities_without_matches": gt_without,
        }

    def _compute_blocking_stats(
        self,
        candidate_map: Dict[str, List[str]],
        gt: Dict[str, Set[str]],
    ) -> Dict[str, Any]:
        counts = [len(v) for v in candidate_map.values()]
        total_candidates = sum(counts)
        mean_c  = float(np.mean(counts))  if counts else 0.0
        median_c = float(np.median(counts)) if counts else 0.0
        p95_c   = float(np.percentile(counts, 95)) if counts else 0.0
        zero_c  = sum(1 for c in counts if c == 0)

        # Blocking recall
        total_gt_pairs = 0
        found = 0
        missed_entities: List[str] = []
        for s1_id, gt_set in gt.items():
            if not gt_set:
                continue
            cands = set(candidate_map.get(s1_id, []))
            hits = gt_set & cands
            total_gt_pairs += len(gt_set)
            found += len(hits)
            if len(hits) < len(gt_set):
                missed_entities.append(s1_id)

        recall = found / total_gt_pairs if total_gt_pairs > 0 else 1.0

        return {
            "total_candidates": total_candidates,
            "mean_candidates_per_s1":   round(mean_c, 3),
            "median_candidates_per_s1": round(median_c, 3),
            "p95_candidates_per_s1":    round(p95_c, 3),
            "entities_with_zero_candidates": zero_c,
            "total_gt_pairs": total_gt_pairs,
            "gt_pairs_found_in_candidates": found,
            "gt_pairs_missing_in_candidates": total_gt_pairs - found,
            "blocking_recall": round(recall, 5),
            "entities_missing_true_candidate": missed_entities,
        }

    def _compute_matching_stats(
        self, match_map: Dict[str, List[str]], all_s1_ids: List[str]
    ) -> Dict[str, Any]:
        counts = [len(match_map.get(s1, [])) for s1 in all_s1_ids]
        zero  = sum(1 for c in counts if c == 0)
        one   = sum(1 for c in counts if c == 1)
        multi = sum(1 for c in counts if c > 1)
        avg   = float(np.mean(counts)) if counts else 0.0
        return {
            "entities_zero_matches":  zero,
            "entities_one_match":     one,
            "entities_multi_matches": multi,
            "avg_matches_per_entity": round(avg, 4),
            "total_predicted_matches": sum(counts),
        }

    def _compute_metrics(
        self,
        gt: Dict[str, Set[str]],
        match_map: Dict[str, List[str]],
        all_s1_ids: List[str],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        evaluator = EntityEvaluator(beta=0.5)
        y_pred = {s1: set(ids) for s1, ids in match_map.items()}
        result = evaluator.evaluate(gt, y_pred, all_s1_ids=all_s1_ids)

        n = result["number_of_entities"]
        correctly_empty = result["correctly_empty_count"]
        empty_gt = result["empty_gt_count"]
        fp_empty = result["false_positive_empty_count"]

        metrics = {
            "macro_f0.5":              round(result.get("macro_f_beta", 0.0), 5),
            "macro_precision":         round(result.get("macro_precision", 0.0), 5),
            "macro_recall":            round(result.get("macro_recall", 0.0), 5),
            "exact_match_rate":        round(result.get("exact_match_rate", 0.0), 5),
            "correctly_empty_rate":    round(correctly_empty / n, 5) if n else 0.0,
            "false_positive_empty_rate": round(fp_empty / n, 5) if n else 0.0,
            "number_of_entities":      n,
        }
        return metrics, result.get("per_entity_records", [])

    def _compute_error_analysis(
        self,
        gt: Dict[str, Set[str]],
        match_map: Dict[str, List[str]],
        candidate_map: Dict[str, List[str]],
        per_entity: List[Dict[str, Any]],
        high_conf_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """Categorises error types per entity."""
        false_positive_entities: List[str] = []
        false_negative_entities: List[str] = []
        missed_candidate_entities: List[str] = []

        for rec in per_entity:
            s1_id = rec["s1_id"]
            if rec["fp"] > 0:
                false_positive_entities.append(s1_id)
            if rec["fn"] > 0:
                false_negative_entities.append(s1_id)
            # GT pair not even in candidates
            gt_set = gt.get(s1_id, set())
            cands = set(candidate_map.get(s1_id, []))
            if gt_set and not (gt_set & cands):
                missed_candidate_entities.append(s1_id)

        # High-confidence FPs: entities where model predicted confidently but GT disagrees
        # We approximate by looking at false-positive entities with non-zero predictions
        # (exact scores aren't stored here; we note entities for investigation)
        high_conf_fp: List[Dict[str, Any]] = [
            {
                "s1_id": rec["s1_id"],
                "predicted": rec["prediction"],
                "ground_truth": rec["ground_truth"],
                "precision": rec["precision"],
            }
            for rec in per_entity
            if rec["fp"] > 0 and rec["precision"] < 0.5
        ]

        # Low-confidence TMs: GT pairs predicted but with many FPs alongside
        low_conf_tm: List[Dict[str, Any]] = [
            {
                "s1_id": rec["s1_id"],
                "predicted": rec["prediction"],
                "ground_truth": rec["ground_truth"],
                "recall": rec["recall"],
            }
            for rec in per_entity
            if rec["tp"] > 0 and rec["recall"] < 0.5
        ]

        return {
            "false_positive_entities":      false_positive_entities,
            "false_negative_entities":      false_negative_entities,
            "missed_candidate_entities":    missed_candidate_entities,
            "high_confidence_false_matches": high_conf_fp,
            "low_confidence_true_matches":  low_conf_tm,
            "per_entity_detail":            per_entity,
        }

    # ── Config fallback ───────────────────────────────────────────────────

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "project": {"experiment_id": "val_run", "seed": 42},
            "data": {
                "id_column_s1": "entity_id",
                "id_column_s2": "entity_id",
                "id_column_s3": "entity_id",
            },
            "blocking": {
                "top_k": 20,
                "blocking_fields": ["business_name", "business_address", "country"],
            },
            "features": {"string_similarity_metrics": ["jaccard"]},
            "decision": {"threshold": 0.5, "threshold_s2": None, "threshold_s3": None, "top_margin": None},
            "evaluation": {"beta": 0.5},
            "submission": {
                "matching_filename": "matching_results.tsv",
                "candidates_filename": "candidate_pairs.tsv",
            },
            "paths": {"submissions_dir": "output", "experiments_dir": "experiments"},
        }
