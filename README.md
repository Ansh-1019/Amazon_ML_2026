# Amazon ML Challenge 2026 — Business Entity Resolution

An end-to-end, modular, and competition-ready machine learning pipeline for **Multi-Source Business Entity Resolution**.

---

## 📌 Problem Overview

In this challenge, **Source 1** serves as the primary reference entity database. The objective is to identify and resolve matching business entities across **Source 2** and **Source 3** for every entity in Source 1.

The official evaluation metric is **Entity-Level Macro $F_{0.5}$** (which weights precision more heavily than recall):

$$F_{\beta} = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}} \quad (\text{where } \beta = 0.5)$$

$$F_{0.5} = 1.25 \times \frac{\text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Key Entity Matching Characteristics:
- **Zero Matches**: An entity in Source 1 may have no corresponding matches in Source 2 or Source 3.
- **Single Match**: An entity may match exactly one record in Source 2 or Source 3.
- **Multiple Matches**: An entity may match multiple records across Source 2 and Source 3.
- **Macro-Averaging**: Precision, Recall, and $F_{0.5}$ are computed *per Source 1 entity* and averaged uniformly across all Source 1 entities.

---

## 🏗️ End-to-End Pipeline Architecture

The system executes a strictly decoupled 6-stage pipeline:

```
[Raw TSV/CSV: Source 1, Source 2, Source 3, (train_matches)]
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Data Ingestion & Normalization                    │
│   • SchemaAdapter & DataLoader (src/data/loader.py)         │
│   • DataNormalizer (src/data/normalizer.py)                 │
└─────────────────────────┬───────────────────────────────────┘
                          │ Standardized internal DataFrames
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Candidate Generation (Blocking)                   │
│   • CandidateGenerator (src/blocking/candidate_gen.py)      │
│   • candidate_map adapter & candidate sparsity tracking     │
└─────────────────────────┬───────────────────────────────────┘
                          │ Candidate pairs / triplets
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Pairwise Feature Extraction                       │
│   • FeatureExtractor (src/features/pairwise.py)             │
│   • Multi-field similarity (Jaccard, Levenshtein, TF-IDF)   │
└─────────────────────────┬───────────────────────────────────┘
                          │ Feature matrix X
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Model Inference & Scoring                          │
│   • ModelPredictor (src/modeling/predict.py)                │
│   • BaselineHeuristicModelAdapter (fallback scorer)         │
└─────────────────────────┬───────────────────────────────────┘
                          │ Match probabilities
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Entity-Level Decision Resolution                  │
│   • EntityResolver (src/decision/entity_resolver.py)        │
│   • 0, 1, or Multi-Match resolution per S1 entity           │
│   • Source-specific thresholds & candidate-subset checks    │
└─────────────────────────┬───────────────────────────────────┘
                          │ Resolved match assignments
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: Evaluation & Submission Generation                 │
│   • EntityEvaluator (src/evaluation/metrics.py)             │
│   • SubmissionGenerator (src/submission.py)                 │
│   • Official Validator (utils/validate_submission.py)       │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
output/matching_results.tsv      output/candidate_pairs.tsv
```

---

## 🧪 Competition Validation Harness & Experiment Tracking

The repository includes a validation harness to answer: **"Did this change actually improve our Amazon ML Challenge score?"**

```
┌────────────────────────────────────────────────────────┐
│               Validation Harness Run                   │
│  python scripts/evaluate.py --synthetic --threshold 0.5│
└───────────────────────────┬────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
 terminal report & diffs              experiments/<exp_id>/
 • DATA summary                       • report.json (Full diagnostic)
 • BLOCKING efficiency                • summary.json (Params + metrics)
 • MATCHING cardinality               • validation.log
 • METRICS (Macro F0.5, P, R)
 • ERROR ANALYSIS (FP, FN, Missed)
```

### 1. Run Validation
```bash
# Run on built-in synthetic benchmark
python scripts/evaluate.py --synthetic --experiment-id baseline_exp

# Run with custom threshold override
python scripts/evaluate.py --synthetic --threshold 0.65 --experiment-id strict_exp

# Run on actual training split / dataset
python scripts/evaluate.py --config configs/default_config.yaml --threshold 0.5
```

### 2. Compare Two Experiments
Compare two validation runs side-by-side with metric deltas and an automated verdict:
```bash
python scripts/evaluate.py --compare experiments/baseline_exp/report.json experiments/strict_exp/report.json
```

**Example Comparison Table:**
```text
======================================================
  EXPERIMENT COMPARISON
======================================================
  Experiment A : baseline_exp
  Experiment B : strict_exp
------------------------------------------------------
  Metric                            A         B  Delta(A-B)
  ----------------------------------------------------
  Macro F0.5 (HEADLINE)        0.0828    0.1667  - -0.0839
  Macro Precision              0.0677    0.1667  - -0.0990
  Macro Recall                 0.8333    0.1667  + +0.6667
  Exact match rate             0.0000    0.1667  - -0.1667
  Blocking recall              1.0000    1.0000  = 0.0000
  Mean cands/entity           16.0000   16.0000  = 0.0000
======================================================

  Verdict (by macro F0.5): B is BETTER
```

---

## 📂 Repository Structure & Module Overview

```text
Amazon 2026/
├── configs/
│   └── default_config.yaml         # Central configuration (paths, blocking, features, models, thresholds)
├── data/
│   ├── raw/                        # Raw source TSV/CSV files (source1, source2, source3, train_matches)
│   ├── processed/                  # Cached intermediate normalized datasets
│   └── candidates/                 # Cached candidate pair artifacts
├── experiments/                    # Run directories containing report.json, summary.json, and validation.log
├── models/                         # Trained model artifacts and weights
├── output/                         # Submission TSVs (matching_results.tsv, candidate_pairs.tsv)
├── submissions/                    # Packaged submission archive storage
├── scripts/
│   └── evaluate.py                 # Validation CLI & experiment comparison tool
├── src/
│   ├── __init__.py                 # Core package exports
│   ├── pipeline.py                 # Master 6-stage orchestrator (EntityResolutionPipeline)
│   ├── predict.py                  # Standalone inference module for test sets
│   ├── submission.py               # Submission generation with strict pre-write checks & validation
│   ├── blocking/
│   │   ├── __init__.py
│   │   └── candidate_gen.py        # Candidate blocking (TF-IDF, nearest neighbors, Top-K)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # TSV/CSV ingestion with schema adapter
│   │   └── normalizer.py           # Text cleaning, abbreviation expansion, address standardisation
│   ├── decision/
│   │   ├── __init__.py
│   │   └── entity_resolver.py      # Multi-match entity resolver (0, 1, or Many matches per S1)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              # Macro F_0.5, Macro Precision, Macro Recall computation
│   ├── features/
│   │   ├── __init__.py
│   │   └── pairwise.py             # Pairwise similarity feature extraction
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── trainer.py              # GBDT training wrapper (CatBoost / LightGBM)
│   │   └── predict.py              # Match probability prediction wrapper
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py               # YAML config loader
│   │   ├── logger.py               # Formatted stdout & file logging
│   │   └── tracker.py              # ExperimentTracker for parameter & metric persistence
│   └── validation/
│       ├── __init__.py
│       └── harness.py              # ValidationHarness & ValidationReport engine
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py            # Pipeline integration & inference tests
│   ├── test_submission.py          # Submission constraints & formatting tests
│   ├── test_evaluation.py          # Entity-level metric & edge-case unit tests
│   ├── test_competition_er.py      # 55 deterministic noisy ER entity test cases
│   └── test_evaluate_harness.py    # 46 validation harness unit & CLI tests
├── utils/
│   ├── __init__.py
│   └── validate_submission.py      # Official competition validator script
├── run_pipeline.py                 # Top-level executable pipeline script
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

### Module Responsibilities

| Module / File | Responsibility & Use |
|---|---|
| [`src/pipeline.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/pipeline.py) | **Central Orchestrator**: Executes the 6 stages, manages adapters, tracks candidate sparsity, computes validation metrics, and produces validated TSV files. |
| [`scripts/evaluate.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/scripts/evaluate.py) | **Validation CLI**: Command-line tool to run validation, inspect error diagnostics, and compare experiments. |
| [`src/validation/harness.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/validation/harness.py) | **Validation Engine**: Computes comprehensive Data, Blocking, Matching, Metric, and Error Analysis sections and serialises JSON reports. |
| [`src/decision/entity_resolver.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/decision/entity_resolver.py) | **Multi-Match Resolver**: Resolves candidate probabilities into 0, 1, or Many match assignments per Source 1 entity with source-specific thresholds and margin filtering. |
| [`src/evaluation/metrics.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/evaluation/metrics.py) | **Competition Metrics**: Calculates exact per-entity Macro Precision, Macro Recall, and Macro $F_{0.5}$. |
| [`src/data/loader.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/data/loader.py) | **Data Loader**: Safely loads TSV/CSV files with schema translation (`entity_id, business_name, business_address, country`). |
| [`src/data/normalizer.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/data/normalizer.py) | **Text Normalizer**: Cleans string fields (lowercasing, punctuation stripping, whitespace normalization). |
| [`src/blocking/candidate_gen.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/blocking/candidate_gen.py) | **Candidate Generator**: Generates candidate pairs per Source 1 entity to reduce $O(N^2)$ search space. |
| [`src/features/pairwise.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/features/pairwise.py) | **Feature Extractor**: Computes multi-field similarity scores between Source 1 and candidate records. |
| [`src/modeling/trainer.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/modeling/trainer.py) | **Model Trainer**: Trains CatBoost / LightGBM / XGBoost binary classifiers on feature vectors with CV. |
| [`src/modeling/predict.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/modeling/predict.py) | **Model Predictor**: Generates probability scores for each candidate pair. |
| [`src/submission.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/submission.py) | **Submission Layer**: Formats and exports TSVs with pre-write constraint verification and validation wrapper. |
| [`src/utils/tracker.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/utils/tracker.py) | **Experiment Tracker**: Persists run configurations, timestamps, logs, and metric JSON files to `experiments/`. |
| [`utils/validate_submission.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/utils/validate_submission.py) | **Official Validator**: Authoritative competition validator enforcing all submission format constraints. |

---

## 📋 Competition Submission Rules

All generated outputs strictly satisfy the official requirements:

1. **Required Files**:
   - `output/matching_results.tsv` (`source1_entity_id \t matched_entity_ids`)
   - `output/candidate_pairs.tsv` (`source1_entity_id \t candidate_entity_ids`)
2. **Complete Coverage**: Exactly one row for every Source 1 test entity.
3. **Subset Constraint**: Matched IDs must be a strict subset of candidate IDs (`matched_entity_ids ⊆ candidate_entity_ids`).
4. **Valid Target IDs**: Only Source 2 and Source 3 entity IDs are permitted.
5. **No Duplicates**: No repeated IDs within any comma-separated ID list.
6. **Pure TSV Formatting**: Tab-separated without quotes or encoding corruption.

---

## ⚡ Quick Start & Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

Execute the master pipeline (with logging, candidate tracking, evaluation, and submission generation):

```bash
python run_pipeline.py --config configs/default_config.yaml
```

Optional CLI overrides:
```bash
# Override decision threshold
python run_pipeline.py --threshold 0.65 --output-dir output
```

### 3. Run Standalone Inference

To generate predictions directly on test datasets:

```bash
python -m src.predict --config configs/default_config.yaml --threshold 0.5 --output-dir output
```

### 4. Run Validation & Error Analysis

Run the competition validation harness:

```bash
# Evaluate on synthetic benchmark
python scripts/evaluate.py --synthetic --experiment-id baseline_run

# Compare two experiment runs
python scripts/evaluate.py --compare experiments/exp1/report.json experiments/exp2/report.json
```

### 5. Run Official Submission Validation

Directly validate submission files using the competition validator:

```bash
python utils/validate_submission.py --matching-results output/matching_results.tsv --candidate-pairs output/candidate_pairs.tsv
```

### 6. Run Automated Tests

Execute the complete test suite (185 tests covering submission formatting, edge cases, metric calculations, synthetic datasets, and validation harness):

```bash
pytest tests/ -v
```

---

## ⚙️ Configuration (`configs/default_config.yaml`)

```yaml
project:
  name: "Amazon ML Challenge 2026 - Entity Resolution"
  experiment_id: "exp01_baseline"
  seed: 42

paths:
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"
  candidates_dir: "data/candidates"
  models_dir: "models"
  experiments_dir: "experiments"
  submissions_dir: "output"

data:
  source1_filename: "source1.tsv"
  source2_filename: "source2.tsv"
  source3_filename: "source3.tsv"
  train_matches_filename: "train_matches.tsv"
  id_column_s1: "source1_entity_id"
  id_column_s2: "source2_entity_id"
  id_column_s3: "source3_entity_id"

blocking:
  top_k: 20
  min_similarity_threshold: 0.1
  blocking_fields: ["business_name", "business_address", "country"]

features:
  string_similarity_metrics: ["levenshtein", "jaccard", "cosine_tfidf"]

modeling:
  model_type: "catboost"
  params:
    iterations: 500
    learning_rate: 0.05
    depth: 6
    eval_metric: "Logloss"
    random_seed: 42
  cv_folds: 5

evaluation:
  beta: 0.5

decision:
  threshold: 0.5
  threshold_s2: 0.5
  threshold_s3: 0.5
  top_margin: null
  max_matches_per_entity: null

submission:
  matching_filename: "matching_results.tsv"
  candidates_filename: "candidate_pairs.tsv"
```