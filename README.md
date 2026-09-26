# Amazon ML Challenge 2026 — Business Entity Resolution

An end-to-end, modular, and competition-ready machine learning system for **Multi-Source Business Entity Resolution (ER)**.

---

## 📌 Problem Overview & Competition Objective

In this competition, **Source 1** serves as the primary reference database of business entities. The goal is to identify and resolve matching business entities across **Source 2** and **Source 3** for every entity in Source 1.

### Key Matching Characteristics
* **Zero Matches**: A Source 1 entity may have no corresponding match in Source 2 or Source 3.
* **Single Match**: A Source 1 entity may match exactly one record in Source 2 or Source 3.
* **Multiple Matches**: A Source 1 entity may match multiple records across Source 2 and Source 3.
* **Subset Constraint**: Every predicted match for a Source 1 entity MUST be present in its candidate set ($\text{matches} \subseteq \text{candidates}$).

### Official Evaluation Metric: Entity-Level Macro $F_{0.5}$
The evaluation metric is **Macro $F_{0.5}$**, which weights precision more heavily than recall ($\beta = 0.5$) to penalize false entity merges:

$$F_{\beta} = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}} \quad (\beta = 0.5)$$

$$F_{0.5} = 1.25 \times \frac{\text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Macro-averaging evaluates $F_{0.5}$ per Source 1 entity independently and averages across all entities:
$$\text{Macro } F_{0.5} = \frac{1}{|S_1|} \sum_{e \in S_1} F_{0.5}(e)$$

---

## 🏗️ Clean System Architecture & End-to-End Workflow

The architecture is strictly decoupled into 6 modular stages with well-defined interfaces and zero circular dependencies:

```text
                            Raw Data Ingestion
                 (Source 1, Source 2, Source 3, Ground Truth)
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 1: Ingestion & Text/Address Normalization                       │
│   • DataLoader: Robust TSV/CSV ingestion & schema translation         │
│   • DataNormalizer: Unicode NFKC case folding & legal suffix cleanup  │
│   • Address Processing: Address normalization & fingerprinting        │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │ Canonical DataFrames
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 2: Multi-Pass Candidate Blocking                                │
│   • Exact Name & Address Fingerprint Blocker                          │
│   • Token-Based Inverted Index Blocker                                │
│   • Character N-Gram TF-IDF Similarity Blocker                        │
│   • Rare Token Inverted Index Blocker                                 │
│   • Deterministic Candidate Union & Prefix-Safe ID Adapter            │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │ Canonical Candidate Pairs
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 3: Pairwise Feature Engineering                                 │
│   • Wide Pair Bridge (left_*, right_*)                                │
│   • 29 Numeric Similarity Features (Jaccard, Levenshtein, N-Grams,    │
│     Length Ratios, Country Matches, Number Overlaps)                  │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │ Feature Matrix X
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 4: ML Training, Inference & Hard Negatives                      │
│   • Hard Negative Mining (synthesizing near-miss false merges)        │
│   • CatBoost & LightGBM Binary GBDT Classifiers                       │
│   • Batch Probability Scoring & Model-to-Resolver Adapter             │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │ Match Probabilities
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 5: Decision Resolution & Threshold Optimization                 │
│   • Anmol choose_threshold(): Optimal Macro F0.5 threshold selection  │
│   • EntityResolver: 0, 1, or Multi-match resolution per S1 entity     │
│   • Source-specific thresholds & Top-Margin candidate pruning         │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │ Resolved Match Assignments
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 6: Unified Evaluation & Submission Generation                   │
│   • EntityEvaluator: Headline Macro F0.5, Precision, Recall, EMR      │
│   • SubmissionGenerator: Formats & writes official TSV files          │
│   • Official Validator: 100% strict compliance verification           │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
      output/matching_results.tsv      output/candidate_pairs.tsv
```

---

## 🔍 Detailed Component Breakdown

### 1. Data Normalization & Address Processing (`src/data/`)
* **`normalization.py`**: Unicode-aware NFKC normalization, casefolding, typographic quote/hyphen standardization, ampersand normalization (`&` $\to$ `and`), and legal business suffix expansion (`corp`, `ltd`, `inc`, `pvt`, `co`, `llc`).
* **`address.py`**: Specialized address normalization that cleans abbreviations (`street` $\to$ `st`, `avenue` $\to$ `ave`, `suite` $\to$ `ste`) and generates alphanumeric **address fingerprints** for deterministic indexing.
* **`loader.py`**: Flexible ingestion for Source 1, Source 2, Source 3, and Ground Truth files supporting both competition headers (`source1_entity_id`) and standard headers (`entity_id`, `s1_id`).

### 2. Multi-Pass Candidate Generation & Blocking (`src/blocking/`)
Reduces the $O(N_1 \times (N_2 + N_3))$ search space to high-precision candidate pools via a multi-pass union:
* **Exact Blocker (`exact_blocking.py`)**: Indexes exact normalized names, normalized addresses, and address fingerprints.
* **Token Blocker (`blocking.py`)**: Token inverted index with custom stopword filtering.
* **Character N-Gram Blocker (`char_ngram_blocking.py`)**: TF-IDF vectorization with cosine similarity matching for fuzzy name variations and OCR typos.
* **Rare Token Blocker (`rare_blocking.py`)**: Low-frequency discriminative token inverted index.
* **Candidate Union (`candidate_union.py`)**: Merges multiple blocker outputs deterministically without duplicate candidate IDs.
* **ID Adapter (`candidate_gen.py`)**: Reversible, prefix-safe ID wrapper that seamlessly supports arbitrary entity ID schemas.

### 3. Pairwise Feature Engineering (`src/features/`)
Extracts **29 fine-grained numerical features** across entity name, address, and country:
* **Name Similarities**: Exact match, lower match, token Jaccard similarity, character 2-gram / 3-gram / 4-gram / 5-gram Jaccard, Levenshtein ratio, token sort ratio, token set ratio, common prefix / suffix ratios.
* **Address Similarities**: Exact match, token Jaccard similarity, character n-gram similarities, Levenshtein distance, number/digit overlap indicator.
* **Geographic & Metadata Features**: Exact country match, country mismatch indicator, token count differences, string length ratios.

### 4. Machine Learning & Hard Negative Mining (`src/modeling/`)
* **Hard Negative Mining (`generate_hard_negatives`)**: Synthesizes challenging near-miss negative pairs (mutating business suffixes and street addresses) to train the model against high-penalty false merges.
* **GBDT Classifiers (`model.py`, `trainer.py`)**: High-performance CatBoost and LightGBM binary classification models with stratified train/validation splitting.
* **Prediction Adapter (`src/decision/adapter.py`)**: Harmonizes probability outputs with the downstream decision layer.

### 5. Threshold Optimization & Decision Resolver (`src/decision/`)
* **Threshold Optimizer (`threshold.py`)**: Evaluates a dense grid of decision thresholds on validation predictions to select the threshold $T^*$ that maximizes Macro $F_{0.5}$.
* **EntityResolver (`entity_resolver.py`)**: Converts continuous candidate match probabilities into final entity-level match sets:
  * Supports $0$, $1$, or multiple matches per Source 1 entity.
  * Enforces the hard competition constraint $\text{matches} \subseteq \text{candidates}$.
  * Supports source-specific thresholds (`threshold_s2`, `threshold_s3`) and top-margin pruning.

### 6. Validation Harness & Official Submission (`src/validation/`, `src/submission/`)
* **Validation Harness (`harness.py`)**: Generates comprehensive multi-level validation reports (Blocking Recall, Pairwise Counts, and Entity Macro $F_{0.5}$).
* **Submission Generator (`submission.py`)**: Emits `matching_results.tsv` and `candidate_pairs.tsv` with strict format validation.
* **Official Validator (`utils/validate_submission.py`)**: Authoritative competition validator verifying row completeness, non-empty candidate sets, subset rules, and pure TSV syntax.

---

## 📂 Repository Structure

```text
Amazon 2026/
├── configs/
│   └── default_config.yaml                  # Central pipeline configuration
├── data/
│   ├── raw/                                 # Raw input TSVs (source1, source2, source3)
│   ├── processed/                           # Processed / normalized intermediate tables
│   └── candidates/                          # Generated candidate pairs
├── experiments/
│   └── threshold_optimization/              # Threshold search metrics & JSON results
├── models/                                  # Serialized ML model artifacts
├── output/                                  # Output submission TSVs
├── submissions/                             # Final competition archive storage
├── scripts/
│   ├── evaluate.py                          # Full 6-stage validation & experiment comparison CLI
│   ├── regression_gate_blocking.py          # Strict candidate blocking regression gate
│   ├── run_integrated_training.py           # End-to-end ML model training with hard negatives
│   └── run_threshold_optimization.py        # Micro/macro F0.5 decision threshold tuner
├── src/
│   ├── __init__.py                          # Core exports
│   ├── pipeline.py                          # Master 6-stage pipeline orchestrator
│   ├── predict.py                           # Standalone batch inference runner
│   ├── submission.py                        # Submission file generation & pre-write checks
│   ├── blocking/                            # Multi-pass candidate blocking system
│   │   ├── candidate_gen.py                 # Unified blocker interface & ID adapter
│   │   ├── candidate_pipeline.py            # Blocker execution orchestrator
│   │   ├── candidate_union.py               # Deterministic candidate union
│   │   ├── candidate_diagnostics.py         # Candidate recall & sparsity diagnostics
│   │   ├── char_ngram_blocking.py           # Character n-gram TF-IDF similarity blocker
│   │   ├── exact_blocking.py                # Exact name & address fingerprint blocker
│   │   └── rare_blocking.py                 # Rare token inverted index blocker
│   ├── data/                                # Ingestion and text/address normalization
│   │   ├── loader.py                        # TSV/CSV data loader & schema adapter
│   │   ├── normalization.py                 # NFKC text and legal token normalization
│   │   ├── normalizer.py                    # Multi-column DataFrame normalizer
│   │   └── address.py                       # Address standardizer & fingerprint generator
│   ├── decision/                            # Entity resolution and decision thresholding
│   │   ├── entity_resolver.py               # Multi-match entity decision resolver
│   │   ├── threshold.py                     # Anmol's F0.5 threshold optimizer
│   │   └── adapter.py                       # Model-to-EntityResolver bridge
│   ├── evaluation/                          # Competition evaluation metrics
│   │   └── metrics.py                       # Exact per-entity Macro F0.5 calculation
│   ├── features/                            # Pairwise feature extraction
│   │   ├── features.py                      # 29 numeric similarity features
│   │   └── pairwise.py                      # Wide pair builder & feature bridge
│   ├── modeling/                            # Machine learning models
│   │   ├── model.py                         # CatBoost / LightGBM trainers & hard negatives
│   │   ├── trainer.py                       # Training wrapper
│   │   └── predict.py                       # Batch prediction wrapper
│   ├── utils/                               # Shared utilities
│   │   ├── config.py                        # YAML configuration loader
│   │   ├── logger.py                        # Formatted logger setup
│   │   └── tracker.py                       # Experiment tracking and metric logging
│   └── validation/                          # Validation harness
│       └── harness.py                       # Validation report generator & diff engine
├── tests/                                   # 26 comprehensive test suites (555 unit & integration tests)
├── utils/
│   └── validate_submission.py               # Official competition validator
├── run_pipeline.py                          # Top-level executable entry point
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## 🚀 Quick Start & Usage Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Execute the full 6-stage pipeline (loads data, normalizes text/address, generates candidates, extracts features, scores pairs, resolves entity decisions, and validates output):
```bash
python run_pipeline.py --config configs/default_config.yaml
```

Optional CLI overrides:
```bash
# Override decision threshold
python run_pipeline.py --threshold 0.55 --output-dir output
```

### 3. Supervised Model Training with Hard Negatives
Train CatBoost and LightGBM models on candidate pairs augmented with synthesized hard negatives:
```bash
python scripts/run_integrated_training.py
```

### 4. Decision Threshold Optimization
Optimize the probability threshold for maximum validation Macro $F_{0.5}$:
```bash
python scripts/run_threshold_optimization.py
```

### 5. Run Competition Validation Harness
Execute the validation harness on the built-in benchmark or real dataset:
```bash
# Run validation on synthetic benchmark
python scripts/evaluate.py --synthetic --experiment-id baseline_run

# Compare two experiment reports side-by-side
python scripts/evaluate.py --compare experiments/baseline_run/report.json experiments/new_run/report.json
```

### 6. Verify Submission Files with Official Validator
Validate the final output TSV files against all competition constraints:
```bash
python utils/validate_submission.py --matching-results output/matching_results.tsv --candidate-pairs output/candidate_pairs.tsv
```

### 7. Run Complete Test Suite
Execute the entire test suite of **555 unit and integration tests**:
```bash
python -m pytest tests/ -v
```

---

## ⚙️ Configuration Reference (`configs/default_config.yaml`)

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
  name_column: "business_name"
  address_column: "business_address"
  country_column: "country"

blocking:
  top_k: 20
  min_similarity_threshold: 0.1
  blocking_fields: ["business_name", "business_address", "country"]

features:
  string_similarity_metrics: ["levenshtein", "jaccard", "cosine_tfidf"]

modeling:
  model_type: "catboost"
  params:
    iterations: 300
    learning_rate: 0.05
    depth: 6
    loss_function: "Logloss"
    random_seed: 42

evaluation:
  beta: 0.5

decision:
  threshold: 0.5
  threshold_s2: null
  threshold_s3: null
  top_margin: null

submission:
  matching_filename: "matching_results.tsv"
  candidates_filename: "candidate_pairs.tsv"
```

---

## 📋 Competition Submission Compliance Summary

The pipeline automatically guarantees:
1. **Output Filenames**: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
2. **Header Specification**: `source1_entity_id \t matched_entity_ids` and `source1_entity_id \t candidate_entity_ids`.
3. **Exact Row Count**: Exactly one row for every Source 1 entity in the test set.
4. **Candidate-Subset Invariant**: For every entity $e$, $\text{matched}(e) \subseteq \text{candidates}(e)$.
5. **Deduplication**: No duplicate target IDs within any comma-separated list.
6. **Pure TSV Encoding**: Clean tab-separated output free of quotation marks and formatting corruption.