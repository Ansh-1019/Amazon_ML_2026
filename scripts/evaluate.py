"""
scripts/evaluate.py — Competition Validation CLI for Amazon ML Challenge 2026.

Answers: "Did this change improve our score?"

Usage
-----
# Run against real data directory
python scripts/evaluate.py

# Override config and threshold
python scripts/evaluate.py --config configs/default_config.yaml --threshold 0.6

# Use a custom experiment ID for tracking
python scripts/evaluate.py --experiment-id "exp42_tfidf_blocker" --threshold 0.55

# Compare two saved reports
python scripts/evaluate.py --compare experiments/exp_A/report.json experiments/exp_B/report.json

# Dry-run using the built-in synthetic dataset (no real data needed)
python scripts/evaluate.py --synthetic

Full CLI help
-------------
python scripts/evaluate.py --help
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

# ── Ensure the project root is importable ───────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config, setup_logger
from src.validation.harness import ValidationHarness, ValidationReport
from src.data.loader import DataLoader

logger = setup_logger("evaluate")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth_from_dataframe(train_matches_df) -> Dict[str, Set[str]]:
    """
    Converts the loaded ground truth DataFrame into {s1_id: {target_ids}} dict.
    Supports both competition schema (source1_entity_id, matched_entity_ids)
    and legacy triplet schema (s1_id, s2_id, s3_id).
    """
    import pandas as pd
    from src.evaluation.metrics import parse_id_set

    gt: Dict[str, Set[str]] = {}

    if train_matches_df is None or train_matches_df.empty:
        return gt

    if "source1_entity_id" in train_matches_df.columns:
        for _, row in train_matches_df.iterrows():
            s1 = str(row["source1_entity_id"]).strip()
            if not s1 or s1 in ("nan", "None", ""):
                continue
            raw = str(row.get("matched_entity_ids", "")).strip()
            gt[s1] = parse_id_set(raw)
    elif "s1_id" in train_matches_df.columns:
        for _, row in train_matches_df.iterrows():
            s1 = str(row.get("s1_id", "")).strip()
            if not s1 or s1 in ("nan", "None", ""):
                continue
            targets: Set[str] = set()
            for col in ("s2_id", "s3_id"):
                if col in row and str(row[col]).strip() not in ("nan", "None", ""):
                    targets.add(str(row[col]).strip())
            gt[s1] = targets

    return gt


def compare_reports(path_a: str, path_b: str) -> None:
    """Loads two saved report JSONs and prints a comparison table."""
    with open(path_a) as f:
        data_a = json.load(f)
    with open(path_b) as f:
        data_b = json.load(f)

    rep_a = ValidationReport(data_a)
    rep_b = ValidationReport(data_b)

    w = 54
    print(f"\n{'=' * w}")
    print(f"  EXPERIMENT COMPARISON")
    print(f"{'=' * w}")
    print(f"  Experiment A : {rep_a.experiment_id}")
    print(f"  Experiment B : {rep_b.experiment_id}")
    print(f"{'-' * w}")

    metrics_a = data_a["metrics"]
    metrics_b = data_b["metrics"]
    blocking_a = data_a["blocking_stats"]
    blocking_b = data_b["blocking_stats"]

    def row(label, key_a, key_b, higher_is_better=True):
        va = key_a if isinstance(key_a, (int, float)) else 0.0
        vb = key_b if isinstance(key_b, (int, float)) else 0.0
        delta = va - vb
        arrow = ("+" if delta > 0 else "-" if delta < 0 else "=") if higher_is_better else \
                ("-" if delta > 0 else "+" if delta < 0 else "=")
        sign = "+" if delta > 0 else ""
        print(f"  {label:<26} {va:>8.4f}  {vb:>8.4f}  {arrow} {sign}{delta:.4f}")

    print(f"  {'Metric':<26} {'A':>8}  {'B':>8}  {'Delta(A-B)':>10}")
    print(f"  {'-' * 52}")
    row("Macro F0.5 (HEADLINE)",     metrics_a.get("macro_f0.5",0), metrics_b.get("macro_f0.5",0))
    row("Macro Precision",           metrics_a.get("macro_precision",0), metrics_b.get("macro_precision",0))
    row("Macro Recall",              metrics_a.get("macro_recall",0), metrics_b.get("macro_recall",0))
    row("Exact match rate",          metrics_a.get("exact_match_rate",0), metrics_b.get("exact_match_rate",0))
    row("Blocking recall",           blocking_a.get("blocking_recall",0), blocking_b.get("blocking_recall",0))
    row("Mean cands/entity",         blocking_a.get("mean_candidates_per_s1",0), blocking_b.get("mean_candidates_per_s1",0), higher_is_better=False)
    print(f"{'=' * w}\n")

    verdict = "A is BETTER" if rep_a.macro_f05 > rep_b.macro_f05 else \
              "B is BETTER" if rep_b.macro_f05 > rep_a.macro_f05 else "TIED"
    print(f"  Verdict (by macro F0.5): {verdict}\n")


def run_synthetic_validation(args) -> ValidationReport:
    """Runs a validation on the synthetic dataset from test_competition_er.py."""
    logger.info("Running validation on built-in synthetic dataset...")

    # Import synthetic dataset from existing test module
    sys.path.insert(0, str(_ROOT))
    from tests.test_competition_er import (
        make_source1, make_source2, make_source3, GROUND_TRUTH
    )

    harness = ValidationHarness(
        config={
            "project": {"experiment_id": args.experiment_id or "synthetic_val", "seed": 42},
            "data": {"id_column_s1": "entity_id", "id_column_s2": "entity_id", "id_column_s3": "entity_id"},
            "blocking": {"top_k": 20, "blocking_fields": ["business_name", "business_address", "country"]},
            "features": {"string_similarity_metrics": ["jaccard"]},
            "decision": {
                "threshold": args.threshold or 0.5,
                "threshold_s2": None, "threshold_s3": None, "top_margin": None,
            },
            "evaluation": {"beta": 0.5},
            "submission": {
                "matching_filename": "matching_results.tsv",
                "candidates_filename": "candidate_pairs.tsv",
            },
            "paths": {"submissions_dir": args.output_dir, "experiments_dir": args.experiments_dir},
        },
        experiments_dir=args.experiments_dir,
    )
    return harness.run(
        ground_truth={k: v for k, v in GROUND_TRUTH.items()},
        s1_df=make_source1(),
        s2_df=make_source2(),
        s3_df=make_source3(),
        experiment_id=args.experiment_id or "synthetic_val",
        threshold=args.threshold or 0.5,
        output_dir=args.output_dir,
        save_report=True,
    )


def run_real_validation(args, config: Dict[str, Any]) -> Optional[ValidationReport]:
    """Loads real data from disk and runs validation against ground truth."""
    data_cfg = config.get("data", {})
    raw_data_dir = config.get("paths", {}).get("raw_data_dir", "data/raw")

    loader = DataLoader(raw_data_dir=raw_data_dir, strict=False)
    logger.info(f"Loading data from: {raw_data_dir}")

    try:
        s1_df, s2_df, s3_df, train_matches = loader.load_sources(config)
    except Exception as exc:
        logger.error(f"Failed to load data: {exc}")
        logger.info("Tip: run --synthetic to use the built-in synthetic dataset instead.")
        return None

    if train_matches is None or train_matches.empty:
        logger.warning(
            "No ground truth found. Validation requires train_matches.tsv.\n"
            "  - Ensure data/raw/train_matches.tsv exists.\n"
            "  - Or use --synthetic for a self-contained demo."
        )
        return None

    gt = load_ground_truth_from_dataframe(train_matches)
    if not gt:
        logger.warning("Ground truth is empty — cannot compute metrics.")
        return None

    harness = ValidationHarness(config=config, experiments_dir=args.experiments_dir)
    return harness.run(
        ground_truth=gt,
        s1_df=s1_df, s2_df=s2_df, s3_df=s3_df,
        experiment_id=args.experiment_id or config.get("project", {}).get("experiment_id", "val_run"),
        threshold=args.threshold,
        output_dir=args.output_dir,
        save_report=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/evaluate.py",
        description="Amazon ML Challenge 2026 — Competition Validation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py --synthetic
  python scripts/evaluate.py --threshold 0.6 --experiment-id exp_tfidf
  python scripts/evaluate.py --compare experiments/expA/report.json experiments/expB/report.json
        """,
    )
    p.add_argument("--config", type=str, default="configs/default_config.yaml",
                   help="Path to YAML config file (default: configs/default_config.yaml)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold override (default: from config)")
    p.add_argument("--experiment-id", type=str, default=None,
                   help="Human-readable experiment label for tracking")
    p.add_argument("--output-dir", type=str, default="output",
                   help="Directory for TSV submission files (default: output/)")
    p.add_argument("--experiments-dir", type=str, default="experiments",
                   help="Root directory for experiment tracking (default: experiments/)")
    p.add_argument("--synthetic", action="store_true",
                   help="Run on the built-in synthetic dataset (no real data required)")
    p.add_argument("--compare", nargs=2, metavar=("REPORT_A", "REPORT_B"),
                   help="Compare two saved report.json files side by side")
    p.add_argument("--json-only", action="store_true",
                   help="Suppress terminal summary; print only the path to saved JSON report")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Comparison mode ──────────────────────────────────────────────────
    if args.compare:
        compare_reports(args.compare[0], args.compare[1])
        return 0

    # ── Config loading ───────────────────────────────────────────────────
    config: Dict[str, Any] = {}
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
        logger.info(f"Loaded config from: {config_path}")
    else:
        logger.warning(f"Config not found at {config_path}. Using defaults.")

    # ── Run validation ───────────────────────────────────────────────────
    t0 = time.time()

    if args.synthetic:
        report = run_synthetic_validation(args)
    else:
        report = run_real_validation(args, config)
        if report is None:
            logger.info("Falling back to synthetic dataset demo...")
            report = run_synthetic_validation(args)

    elapsed = time.time() - t0

    # ── Output ───────────────────────────────────────────────────────────
    if args.json_only:
        run_dir = report.as_dict["run_info"].get("run_dir", ".")
        json_path = Path(run_dir) / "report.json"
        print(str(json_path))
    else:
        report.print_summary()
        run_dir = report.as_dict["run_info"].get("run_dir", ".")
        print(f"  Saved -> {run_dir}/report.json")
        print(f"  Total wall time: {elapsed:.2f}s\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
