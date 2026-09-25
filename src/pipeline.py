"""
Integration Pipeline for Amazon ML Challenge 2026 - Business Entity Resolution.

Orchestrates the 6-stage end-to-end entity resolution workflow:
  1. Data Ingestion & Normalization
  2. Candidate Generation (Blocking)
  3. Pairwise Feature Extraction
  4. Model Inference & Scoring
  5. Entity-Level Decision Resolution
  6. Evaluation & Submission Generation

Guarantees clean interfaces, reproducible execution, configurable thresholds,
and full compliance with the official submission validator.
"""

import os
import sys
import random
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Core repository imports
from src.utils import setup_logger, load_config, ExperimentTracker
from src.data import DataLoader, DataNormalizer
from src.blocking import CandidateGenerator
from src.features import FeatureExtractor
from src.modeling import ModelTrainer, ModelPredictor
from src.evaluation import EntityEvaluator
from src.decision import EntityResolver
from src.submission import (
    generate_submission_files,
    run_official_validator,
    SubmissionGenerator,
    SubmissionValidationError,
)

logger = logging.getLogger("pipeline.orchestrator")


# ---------------------------------------------------------------------------
# Utility & Adapter Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def extract_target_ids_from_row(row: pd.Series, id_cols: List[str]) -> List[str]:
    """Extracts non-empty, non-null string IDs from specified columns in a row."""
    target_ids = []
    for col in id_cols:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip().strip("\"'")
            if val and val != "None" and val != "nan":
                if val not in target_ids:
                    target_ids.append(val)
    return target_ids


def candidates_df_to_map(
    candidates_df: pd.DataFrame,
    all_s1_ids: List[str],
    s1_id_col: str = "s1_id",
    valid_target_ids: Optional[Set[str]] = None,
) -> Dict[str, List[str]]:
    """
    Converts candidates DataFrame (triplet or pair format) into a dictionary mapping:
    s1_id -> list of unique candidate target IDs (Source 2 and Source 3 IDs).
    
    Guarantees:
      - Every s1_id in all_s1_ids is present as a key (empty list if no candidates).
      - No duplicate target IDs in list.
      - If valid_target_ids is provided, only IDs in valid_target_ids are retained.
    """
    candidate_map: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ids}
    if candidates_df.empty:
        return candidate_map

    # Determine s1 ID column
    actual_s1_col = s1_id_col if s1_id_col in candidates_df.columns else (
        "s1_id" if "s1_id" in candidates_df.columns else "entity_id"
    )

    # Determine target ID columns present in candidates_df
    possible_target_cols = [
        c for c in ["s2_id", "s3_id", "target_id", "candidate_id", "matched_id"]
        if c in candidates_df.columns
    ]

    for _, row in candidates_df.iterrows():
        s1_id = str(row.get(actual_s1_col, "")).strip().strip("\"'")
        if not s1_id:
            continue
        if s1_id not in candidate_map:
            candidate_map[s1_id] = []

        target_ids = extract_target_ids_from_row(row, possible_target_cols)
        for tid in target_ids:
            if valid_target_ids is not None and tid not in valid_target_ids:
                continue
            if tid not in candidate_map[s1_id]:
                candidate_map[s1_id].append(tid)

    return candidate_map


def resolved_df_to_map(
    resolved_df: pd.DataFrame,
    all_s1_ids: List[str],
    candidate_map: Optional[Dict[str, List[str]]] = None,
    s1_id_col: str = "s1_id",
    valid_target_ids: Optional[Set[str]] = None,
) -> Dict[str, List[str]]:
    """
    Converts resolved matches DataFrame into a dictionary mapping:
    s1_id -> list of unique matched target IDs.
    
    Guarantees:
      - Every s1_id in all_s1_ids is present as a key.
      - Matched IDs are a subset of candidate_map[s1_id] (if candidate_map is supplied).
      - Only valid_target_ids are emitted (if supplied).
    """
    match_map: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ids}
    if resolved_df.empty:
        return match_map

    actual_s1_col = s1_id_col if s1_id_col in resolved_df.columns else (
        "s1_id" if "s1_id" in resolved_df.columns else "entity_id"
    )

    possible_target_cols = [
        c for c in ["s2_id", "s3_id", "target_id", "matched_id", "candidate_id"]
        if c in resolved_df.columns
    ]

    for _, row in resolved_df.iterrows():
        s1_id = str(row.get(actual_s1_col, "")).strip().strip("\"'")
        if not s1_id:
            continue
        if s1_id not in match_map:
            match_map[s1_id] = []

        target_ids = extract_target_ids_from_row(row, possible_target_cols)
        cand_set = set(candidate_map.get(s1_id, [])) if candidate_map is not None else None

        for tid in target_ids:
            if valid_target_ids is not None and tid not in valid_target_ids:
                continue
            if cand_set is not None and tid not in cand_set:
                logger.warning(f"Excluding match '{tid}' for entity '{s1_id}' because it is not in candidates.")
                continue
            if tid not in match_map[s1_id]:
                match_map[s1_id].append(tid)

    return match_map


# ---------------------------------------------------------------------------
# Fallback / Baseline Adapters
# ---------------------------------------------------------------------------

class BaselineHeuristicModelAdapter:
    """
    Adapter for candidate scoring when a trained ML model artifact is not yet available.
    Uses blocking scores or feature averages as baseline match probabilities.
    """

    def __init__(self, default_score: float = 0.8):
        self.default_score = default_score

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if X.empty:
            return np.array([])
        # If similarity features exist, compute average similarity
        sim_cols = [c for c in X.columns if "jaccard" in c or "sim" in c or "score" in c]
        if sim_cols:
            scores = X[sim_cols].mean(axis=1).to_numpy()
            return np.clip(scores, 0.0, 1.0)
        return np.full(len(X), self.default_score)


# ---------------------------------------------------------------------------
# Main Integration Pipeline
# ---------------------------------------------------------------------------

class EntityResolutionPipeline:
    """
    End-to-End Modular Integration Pipeline for Business Entity Resolution.
    
    Stages:
      1. Data Loading: Loads and normalizes Source 1, 2, 3 tables.
      2. Candidate Generation: Generates candidate pairs/triplets via blocking.
      3. Feature Generation: Computes pairwise similarity features.
      4. Model Inference: Generates match probability scores.
      5. Entity-Level Decision: Selects matches per Source 1 entity above threshold.
      6. Evaluation & Submission: Validates and writes official TSV files.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = "configs/default_config.yaml",
        model: Optional[Any] = None,
    ):
        if config is not None:
            self.config = config
        elif config_path is not None and Path(config_path).exists():
            self.config = load_config(config_path)
        else:
            self.config = self._get_default_config()

        # Set random seed for reproducibility
        seed = self.config.get("project", {}).get("seed", 42)
        set_seed(seed)

        # Setup experiment tracking and logging
        exp_id = self.config.get("project", {}).get("experiment_id", "pipeline_run")
        exp_dir = self.config.get("paths", {}).get("experiments_dir", "experiments")
        self.tracker = ExperimentTracker(experiment_id=exp_id, experiments_dir=exp_dir)
        self.logger = setup_logger(name="pipeline", log_file=self.tracker.get_run_dir() / "pipeline.log")

        # Initialize modular components
        raw_data_dir = self.config.get("paths", {}).get("raw_data_dir", "data/raw")
        self.data_loader = DataLoader(raw_data_dir=raw_data_dir)
        self.normalizer = DataNormalizer()
        
        top_k = self.config.get("blocking", {}).get("top_k", 20)
        blocking_fields = self.config.get("blocking", {}).get(
            "blocking_fields", ["business_name", "business_address", "country"]
        )
        self.candidate_generator = CandidateGenerator(top_k=top_k, blocking_fields=blocking_fields)
        
        self.feature_extractor = FeatureExtractor()
        self.model = model
        
        dec_cfg = self.config.get("decision", {})
        self.resolver = EntityResolver(
            threshold=dec_cfg.get("threshold", 0.5),
            threshold_s2=dec_cfg.get("threshold_s2", None),
            threshold_s3=dec_cfg.get("threshold_s3", None),
            top_margin=dec_cfg.get("top_margin", None),
        )
        
        beta = self.config.get("evaluation", {}).get("beta", 0.5)
        self.evaluator = EntityEvaluator(beta=beta)
        
        submissions_dir = self.config.get("paths", {}).get("submissions_dir", "output")
        self.submission_generator = SubmissionGenerator(output_dir=submissions_dir)

    def _get_default_config(self) -> Dict[str, Any]:
        """Provides default fallback configuration if no file is found."""
        return {
            "project": {"name": "Amazon ML Challenge 2026", "experiment_id": "exp_default", "seed": 42},
            "paths": {
                "raw_data_dir": "data/raw",
                "processed_data_dir": "data/processed",
                "candidates_dir": "data/candidates",
                "models_dir": "models",
                "experiments_dir": "experiments",
                "submissions_dir": "output",
            },
            "data": {
                "source1_filename": "source1.tsv",
                "source2_filename": "source2.tsv",
                "source3_filename": "source3.tsv",
                "train_matches_filename": "train_matches.tsv",
                "id_column": "entity_id",
                "id_column_s1": "entity_id",
                "id_column_s2": "entity_id",
                "id_column_s3": "entity_id",
                "name_column": "business_name",
                "address_column": "business_address",
                "country_column": "country",
            },
            "blocking": {
                "top_k": 20,
                "blocking_fields": ["business_name", "business_address", "country"],
            },
            "features": {"string_similarity_metrics": ["levenshtein", "jaccard", "cosine_tfidf"]},
            "modeling": {"model_type": "catboost", "params": {"random_seed": 42}},
            "evaluation": {"beta": 0.5},
            "decision": {
                "threshold": 0.5,
                "threshold_s2": None,
                "threshold_s3": None,
                "top_margin": None,
            },
            "submission": {
                "matching_filename": "matching_results.tsv",
                "candidates_filename": "candidate_pairs.tsv",
            },
        }

    # -----------------------------------------------------------------------
    # Stage 1: Data Loading & Normalization
    # -----------------------------------------------------------------------
    def load_and_normalize_data(
        self,
        s1_df: Optional[pd.DataFrame] = None,
        s2_df: Optional[pd.DataFrame] = None,
        s3_df: Optional[pd.DataFrame] = None,
        train_matches: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], List[str], Set[str]]:
        """
        Stage 1: Ingests raw tables and applies text normalization.
        
        Returns:
            s1_norm, s2_norm, s3_norm, train_matches, all_s1_ids, valid_target_ids
        """
        self.logger.info("--- Stage 1: Data Ingestion & Normalization ---")
        start_t = time.time()

        if s1_df is None or s2_df is None or s3_df is None:
            s1_df, s2_df, s3_df, train_matches = self.data_loader.load_sources(self.config)

        data_cfg = self.config.get("data", {})
        
        # Dynamic ID column resolution (canonical 'entity_id' or configured / legacy aliases)
        id_s1 = data_cfg.get("id_column_s1", "entity_id")
        if id_s1 not in s1_df.columns and "entity_id" in s1_df.columns:
            id_s1 = "entity_id"
        elif id_s1 not in s1_df.columns and "s1_id" in s1_df.columns:
            id_s1 = "s1_id"

        id_s2 = data_cfg.get("id_column_s2", "entity_id")
        if id_s2 not in s2_df.columns and "entity_id" in s2_df.columns:
            id_s2 = "entity_id"
        elif id_s2 not in s2_df.columns and "s2_id" in s2_df.columns:
            id_s2 = "s2_id"

        id_s3 = data_cfg.get("id_column_s3", "entity_id")
        if id_s3 not in s3_df.columns and "entity_id" in s3_df.columns:
            id_s3 = "entity_id"
        elif id_s3 not in s3_df.columns and "s3_id" in s3_df.columns:
            id_s3 = "s3_id"

        all_s1_ids = s1_df[id_s1].astype(str).str.strip().tolist() if not s1_df.empty and id_s1 in s1_df.columns else []
        valid_s2_ids = set(s2_df[id_s2].astype(str).str.strip().dropna().tolist()) if not s2_df.empty and id_s2 in s2_df.columns else set()
        valid_s3_ids = set(s3_df[id_s3].astype(str).str.strip().dropna().tolist()) if not s3_df.empty and id_s3 in s3_df.columns else set()
        valid_target_ids = valid_s2_ids.union(valid_s3_ids)

        self.logger.info(
            f"Loaded records -> Source 1: {len(s1_df)}, Source 2: {len(s2_df)}, Source 3: {len(s3_df)}"
        )
        if train_matches is not None:
            self.logger.info(f"Loaded ground truth matches: {len(train_matches)} rows")

        # Normalization
        text_cols = self.config.get("blocking", {}).get(
            "blocking_fields", ["business_name", "business_address", "country", "name", "address"]
        )
        s1_norm = self.normalizer.normalize_dataframe(s1_df, text_columns=text_cols)
        s2_norm = self.normalizer.normalize_dataframe(s2_df, text_columns=text_cols)
        s3_norm = self.normalizer.normalize_dataframe(s3_df, text_columns=text_cols)

        self.logger.info(f"Stage 1 completed in {time.time() - start_t:.2f}s")
        return s1_norm, s2_norm, s3_norm, train_matches, all_s1_ids, valid_target_ids

    # -----------------------------------------------------------------------
    # Stage 2: Candidate Generation (Blocking)
    # -----------------------------------------------------------------------
    def generate_candidates(
        self,
        s1_norm: pd.DataFrame,
        s2_norm: pd.DataFrame,
        s3_norm: pd.DataFrame,
        all_s1_ids: List[str],
        valid_target_ids: Optional[Set[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Stage 2: Generates candidate pairs/triplets via blocking.
        Logs candidate counts per entity and total pairs.
        """
        self.logger.info("--- Stage 2: Candidate Generation (Blocking) ---")
        start_t = time.time()

        candidates_df = self.candidate_generator.generate_candidates(
            s1_norm, s2_norm, s3_norm, self.config
        )

        # Build candidate map
        candidate_map = candidates_df_to_map(
            candidates_df,
            all_s1_ids=all_s1_ids,
            s1_id_col=self.config.get("data", {}).get("id_column_s1", "s1_id"),
            valid_target_ids=valid_target_ids,
        )

        # Logging candidate metrics
        total_candidates = sum(len(cands) for cands in candidate_map.values())
        entities_with_cands = sum(1 for cands in candidate_map.values() if len(cands) > 0)
        avg_cands = total_candidates / max(len(all_s1_ids), 1)

        self.logger.info(
            f"Candidate Generation Metrics: "
            f"Total Entities={len(all_s1_ids)}, "
            f"Entities with candidates={entities_with_cands}/{len(all_s1_ids)}, "
            f"Total Candidate Pairs={total_candidates}, "
            f"Avg candidates/entity={avg_cands:.2f}"
        )
        self.logger.info(f"Stage 2 completed in {time.time() - start_t:.2f}s")
        return candidates_df, candidate_map

    # -----------------------------------------------------------------------
    # Stage 3: Feature Extraction
    # -----------------------------------------------------------------------
    def extract_features(
        self,
        candidates_df: pd.DataFrame,
        s1_norm: pd.DataFrame,
        s2_norm: pd.DataFrame,
        s3_norm: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Stage 3: Extracts pairwise similarity features for all candidate records.
        """
        self.logger.info("--- Stage 3: Pairwise Feature Generation ---")
        start_t = time.time()

        if candidates_df.empty:
            self.logger.warning("No candidate pairs to extract features from.")
            return pd.DataFrame()

        features_df = self.feature_extractor.extract_features(
            candidates_df, s1_norm, s2_norm, s3_norm, self.config
        )

        self.logger.info(f"Extracted {features_df.shape[1]} features for {len(features_df)} candidate rows.")
        self.logger.info(f"Stage 3 completed in {time.time() - start_t:.2f}s")
        return features_df

    # -----------------------------------------------------------------------
    # Stage 4: Model Inference
    # -----------------------------------------------------------------------
    def score_candidates(
        self,
        candidates_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Stage 4: Scores candidate pairs using ML model or fallback adapter.
        """
        self.logger.info("--- Stage 4: Model Inference & Candidate Scoring ---")
        start_t = time.time()

        scored_df = candidates_df.copy()

        if scored_df.empty:
            scored_df["match_score"] = []
            return scored_df

        if self.model is not None:
            predictor = ModelPredictor(model=self.model)
            probs = predictor.predict_proba(features_df)
            scored_df["match_score"] = probs
            self.logger.info(f"Scored {len(scored_df)} candidate pairs using active ModelPredictor.")
        else:
            self.logger.info(
                "No trained model artifact provided. Using BaselineHeuristicModelAdapter for candidate scoring."
            )
            adapter = BaselineHeuristicModelAdapter(default_score=0.8)
            probs = adapter.predict_proba(features_df)
            scored_df["match_score"] = probs

        self.logger.info(f"Stage 4 completed in {time.time() - start_t:.2f}s")
        return scored_df

    # -----------------------------------------------------------------------
    # Stage 5: Entity-Level Decision Resolution
    # -----------------------------------------------------------------------
    def resolve_decisions(
        self,
        scored_candidates_df: pd.DataFrame,
        all_s1_ids: List[str],
        candidate_map: Dict[str, List[str]],
        threshold: Optional[float] = None,
        valid_target_ids: Optional[Set[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Stage 5: Converts match scores into entity-level assignments above threshold.
        Enforces candidate-subset constraint.
        """
        self.logger.info("--- Stage 5: Entity-Level Decision Resolution ---")
        start_t = time.time()

        # Build a merged config that injects any threshold override into
        # the decision sub-dict (EntityResolver.resolve reads threshold from config).
        effective_config = self.config
        if threshold is not None:
            self.resolver.threshold = threshold  # keep instance in sync for logging
            effective_config = {
                **self.config,
                "decision": {**self.config.get("decision", {}), "threshold": threshold},
            }

        self.logger.info(f"Applying decision threshold: {self.resolver.threshold}")
        resolved_df = self.resolver.resolve(scored_candidates_df, effective_config)

        # Build match map
        match_map = resolved_df_to_map(
            resolved_df=resolved_df,
            all_s1_ids=all_s1_ids,
            candidate_map=candidate_map,
            s1_id_col=self.config.get("data", {}).get("id_column_s1", "s1_id"),
            valid_target_ids=valid_target_ids,
        )

        matched_entities_count = sum(1 for m in match_map.values() if len(m) > 0)
        total_matches_count = sum(len(m) for m in match_map.values())
        self.logger.info(
            f"Decision Metrics: "
            f"Resolved Matches for {matched_entities_count}/{len(all_s1_ids)} entities "
            f"(Total matched IDs: {total_matches_count})"
        )
        self.logger.info(f"Stage 5 completed in {time.time() - start_t:.2f}s")
        return resolved_df, match_map

    # -----------------------------------------------------------------------
    # Stage 6: Evaluation & Submission Generation
    # -----------------------------------------------------------------------
    def evaluate_and_generate_submission(
        self,
        candidate_map: Dict[str, List[str]],
        match_map: Dict[str, List[str]],
        all_s1_ids: List[str],
        train_matches: Optional[pd.DataFrame] = None,
        valid_target_ids: Optional[Set[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path, Dict[str, float], bool]:
        """
        Stage 6: Runs evaluation (if ground truth available), writes TSVs, and validates.
        
        Returns:
            matching_path, candidate_path, eval_metrics, is_valid
        """
        self.logger.info("--- Stage 6: Evaluation & Submission Generation ---")
        start_t = time.time()

        # 6a. Evaluation
        eval_metrics: Dict[str, float] = {}
        if train_matches is not None and not train_matches.empty:
            self.logger.info("Computing validation / training F_0.5 metrics...")
            data_cfg = self.config.get("data", {})
            # y_true: s1_id -> set of ground-truth matched IDs
            y_true: Dict[str, Set[str]] = {}

            # Support competition schema (source1_entity_id, matched_entity_ids)
            # and triplet schema (s1_id, s2_id, s3_id)
            if "source1_entity_id" in train_matches.columns:
                for _, row in train_matches.iterrows():
                    s1_v = str(row["source1_entity_id"]).strip()
                    if not s1_v or s1_v in ("nan", "None"):
                        continue
                    matched_raw = str(row.get("matched_entity_ids", "")).strip()
                    targets = set(
                        x.strip() for x in matched_raw.split(",")
                        if x.strip() and x.strip() not in ("nan", "None")
                    )
                    y_true[s1_v] = targets
            else:
                id_s1 = (
                    data_cfg.get("id_column_s1", "s1_id")
                    if data_cfg.get("id_column_s1") in train_matches.columns
                    else ("s1_id" if "s1_id" in train_matches.columns else "entity_id")
                )
                id_s2 = (
                    data_cfg.get("id_column_s2", "s2_id")
                    if data_cfg.get("id_column_s2") in train_matches.columns
                    else ("s2_id" if "s2_id" in train_matches.columns else "")
                )
                id_s3 = (
                    data_cfg.get("id_column_s3", "s3_id")
                    if data_cfg.get("id_column_s3") in train_matches.columns
                    else ("s3_id" if "s3_id" in train_matches.columns else "")
                )
                for _, row in train_matches.iterrows():
                    s1_v = str(row[id_s1]).strip() if id_s1 in row else ""
                    if not s1_v or s1_v in ("nan", "None"):
                        continue
                    gt_ids: Set[str] = set()
                    for id_col in [id_s2, id_s3]:
                        if id_col and id_col in row and pd.notna(row[id_col]):
                            val = str(row[id_col]).strip()
                            if val and val not in ("nan", "None"):
                                gt_ids.add(val)
                    y_true[s1_v] = gt_ids

            # y_pred: s1_id -> set of predicted matched IDs (from match_map)
            y_pred: Dict[str, Set[str]] = {
                s1_v: set(targets) for s1_v, targets in match_map.items()
            }

            eval_metrics = self.evaluator.evaluate(
                y_true, y_pred, all_s1_ids=all_s1_ids
            )
            self.tracker.log_metrics(eval_metrics)

        # 6b. Submission File Generation
        out_dir = Path(output_dir or self.config.get("paths", {}).get("submissions_dir", "output"))
        matching_fn = self.config.get("submission", {}).get("matching_filename", "matching_results.tsv")
        candidates_fn = self.config.get("submission", {}).get("candidates_filename", "candidate_pairs.tsv")

        matching_path, candidate_path = generate_submission_files(
            test_s1_ids=all_s1_ids,
            candidate_pairs_map=candidate_map,
            matched_results_map=match_map,
            output_dir=out_dir,
            valid_target_ids=valid_target_ids,
            matching_filename=matching_fn,
            candidates_filename=candidates_fn,
            strict=True,
        )

        # 6c. Official Validator Execution
        is_valid, val_log = run_official_validator(
            matching_filepath=matching_path,
            candidates_filepath=candidate_path,
        )
        if is_valid:
            self.logger.info("Official Validator PASS: All competition submission constraints satisfied.")
        else:
            self.logger.error(f"Official Validator FAIL: {val_log}")

        self.logger.info(f"Stage 6 completed in {time.time() - start_t:.2f}s")
        return matching_path, candidate_path, eval_metrics, is_valid

    # -----------------------------------------------------------------------
    # Orchestrated Full Pipeline Execution
    # -----------------------------------------------------------------------
    def run(
        self,
        s1_df: Optional[pd.DataFrame] = None,
        s2_df: Optional[pd.DataFrame] = None,
        s3_df: Optional[pd.DataFrame] = None,
        train_matches: Optional[pd.DataFrame] = None,
        threshold: Optional[float] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the entire 6-stage entity resolution pipeline.
        
        Returns:
            Dictionary containing execution metadata, paths, metrics, and validation status.
        """
        self.logger.info(f"=== Running Entity Resolution Pipeline [{self.tracker.experiment_id}] ===")
        total_start_t = time.time()
        self.tracker.log_params(self.config)

        # Stage 1: Load and normalize
        s1_norm, s2_norm, s3_norm, train_matches, all_s1_ids, valid_target_ids = self.load_and_normalize_data(
            s1_df=s1_df, s2_df=s2_df, s3_df=s3_df, train_matches=train_matches
        )

        # If data is empty (e.g. initial run before data files are placed)
        if not all_s1_ids:
            self.logger.warning("No Source 1 entities found in dataset. Pipeline will complete with empty outputs.")

        # Stage 2: Candidate generation
        candidates_df, candidate_map = self.generate_candidates(
            s1_norm=s1_norm,
            s2_norm=s2_norm,
            s3_norm=s3_norm,
            all_s1_ids=all_s1_ids,
            valid_target_ids=valid_target_ids,
        )

        # Stage 3: Feature extraction
        features_df = self.extract_features(
            candidates_df=candidates_df,
            s1_norm=s1_norm,
            s2_norm=s2_norm,
            s3_norm=s3_norm,
        )

        # Stage 4: Model inference
        scored_candidates_df = self.score_candidates(
            candidates_df=candidates_df,
            features_df=features_df,
        )

        # Stage 5: Entity decision
        resolved_df, match_map = self.resolve_decisions(
            scored_candidates_df=scored_candidates_df,
            all_s1_ids=all_s1_ids,
            candidate_map=candidate_map,
            threshold=threshold,
            valid_target_ids=valid_target_ids,
        )

        # Stage 6: Evaluation & Submission
        matching_path, candidate_path, eval_metrics, is_valid = self.evaluate_and_generate_submission(
            candidate_map=candidate_map,
            match_map=match_map,
            all_s1_ids=all_s1_ids,
            train_matches=train_matches,
            valid_target_ids=valid_target_ids,
            output_dir=output_dir,
        )

        elapsed = time.time() - total_start_t
        self.logger.info(f"=== Pipeline Completed Successfully in {elapsed:.2f}s ===")

        results = {
            "experiment_id": self.tracker.experiment_id,
            "matching_results_path": str(matching_path),
            "candidate_pairs_path": str(candidate_path),
            "num_s1_entities": len(all_s1_ids),
            "num_candidates_generated": len(candidates_df),
            "num_resolved_matches": len(resolved_df),
            "eval_metrics": eval_metrics,
            "is_submission_valid": is_valid,
            "elapsed_seconds": round(elapsed, 3),
        }
        self.tracker.log_metrics({"pipeline_summary": results})
        return results


def run_pipeline_cli():
    """Command-line entry point for pipeline execution."""
    import argparse
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - Entity Resolution Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to configuration YAML")
    parser.add_argument("--threshold", type=float, default=None, help="Optional decision threshold override")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory override")
    args = parser.parse_args()

    pipeline = EntityResolutionPipeline(config_path=args.config)
    results = pipeline.run(threshold=args.threshold, output_dir=args.output_dir)
    print("\n--- Pipeline Execution Summary ---")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_pipeline_cli()
