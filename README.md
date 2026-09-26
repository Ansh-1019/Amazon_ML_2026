# Amazon ML Challenge 2026 — Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-555%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/ML-CatBoost%20%7C%20LightGBM-orange.svg)]()
[![Validation](https://img.shields.io/badge/Official%20Validator-100%25%20Passing-success.svg)]()

An end-to-end, modular, and competition-ready machine learning framework for **Multi-Source Business Entity Resolution (ER)**, engineered for high precision, scalability, and strict compliance with the Amazon ML Challenge 2026 specifications.

---

## 📑 Table of Contents

* [1. About the Project](#1-about-the-project)
  * [Problem Statement](#problem-statement)
  * [Core Challenges](#core-challenges)
  * [Official Evaluation Metric: Macro $F_{0.5}$](#official-evaluation-metric-macro-f_05)
  * [Cardinality & Matching Characteristics](#cardinality--matching-characteristics)
* [2. Tech Stack & Dependencies](#2-tech-stack--dependencies)
* [3. Clean System Architecture](#3-clean-system-architecture)
  * [High-Level Data Flow](#high-level-data-flow)
  * [Subsystem Interaction Diagram](#subsystem-interaction-diagram)
* [4. Working of the Project (Stage-by-Stage)](#4-working-of-the-project-stage-by-stage)
  * [Stage 1: Ingestion & Text/Address Normalization](#stage-1-ingestion--textaddress-normalization)
  * [Stage 2: Multi-Pass Candidate Blocking](#stage-2-multi-pass-candidate-blocking)
  * [Stage 3: Pairwise Feature Engineering (29 Features)](#stage-3-pairwise-feature-engineering-29-features)
  * [Stage 4: Supervised ML Modeling & Hard Negative Mining](#stage-4-supervised-ml-modeling--hard-negative-mining)
  * [Stage 5: Decision Resolution & $F_{0.5}$ Threshold Optimization](#stage-5-decision-resolution--f_05-threshold-optimization)
  * [Stage 6: Unified Evaluation & Submission Generation](#stage-6-unified-evaluation--submission-generation)
* [5. Detailed Component & Module Directory](#5-detailed-component--module-directory)
  * [Directory Structure](#directory-structure)
  * [Module Responsibilities Reference](#module-responsibilities-reference)
* [6. Execution Guide & CLI Commands](#6-execution-guide--cli-commands)
  * [Environment Setup](#environment-setup)
  * [Running the End-to-End Pipeline](#running-the-end-to-end-pipeline)
  * [Model Training with Hard Negatives](#model-training-with-hard-negatives)
  * [Decision Threshold Optimization](#decision-threshold-optimization)
  * [Running the Blocking Regression Gate](#running-the-blocking-regression-gate)
  * [Evaluating Benchmark Experiments](#evaluating-benchmark-experiments)
  * [Validating Official Submission Files](#validating-official-submission-files)
  * [Executing Automated Test Suites](#executing-automated-test-suites)
* [7. Performance & Benchmark Metrics](#7-performance--benchmark-metrics)
* [8. Configuration Reference (`configs/default_config.yaml`)](#8-configuration-reference-configsdefault_configyaml)
* [9. Competition Submission Compliance](#9-competition-submission-compliance)

---

## 1. About the Project

### Problem Statement
In multi-source business entity resolution, entities from different systems, web crawls, government registries, or commercial directories often contain variations, OCR misspellings, varying abbreviation schemes, missing attributes, or differing address structures. 

In this competition:
* **Source 1 ($S_1$)** is the primary reference database containing ground-truth business entities.
* **Source 2 ($S_2$)** and **Source 3 ($S_3$)** represent external databases containing noisy, heterogeneous business records.
* **Goal**: For every business entity in Source 1, identify all matching entity records in Source 2 and Source 3 while strictly bounding candidate search space and maximizing precision-focused resolution.

### Core Challenges
1. **Combinatorial Explosion**: A naive pairwise comparison across $N_1 \times (N_2 + N_3)$ requires billions of comparisons. Intelligent multi-pass blocking is required to achieve $>99\%$ recall with $<20$ candidates per entity.
2. **Heavy Precision Penalty**: The competition metric is **Macro $F_{0.5}$**, penalizing false entity merges twice as heavily as false dismissals. Models must be trained against challenging near-miss negatives.
3. **Many-to-Many Matching**: Unlike standard 1-to-1 record linkage, Source 1 entities can match 0, 1, or multiple records across both target sources simultaneously.
4. **Strict Submission Constraints**: Predictions must be formatted into TSV files where predicted matches are a strict subset of candidate pairs ($\text{matches} \subseteq \text{candidates}$) and all test entities have exactly one row.

### Official Evaluation Metric: Macro $F_{0.5}$
The evaluation metric is **Macro-averaged $F_{0.5}$**, setting $\beta = 0.5$:

$$F_{\beta} = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}$$

$$F_{0.5} = 1.25 \times \frac{\text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Macro-averaging calculates precision, recall, and $F_{0.5}$ for each Source 1 entity independently, handling edge cases:
* $\text{Ground Truth} = \emptyset \land \text{Prediction} = \emptyset \implies F_{0.5} = 1.0$ (Correctly identified empty)
* $\text{Ground Truth} = \emptyset \land \text{Prediction} \neq \emptyset \implies F_{0.5} = 0.0$ (False merge)
* $\text{Ground Truth} \neq \emptyset \land \text{Prediction} = \emptyset \implies F_{0.5} = 0.0$ (Missed match)

$$\text{Macro } F_{0.5} = \frac{1}{|S_1|} \sum_{e \in S_1} F_{0.5}(e)$$

### Cardinality & Matching Characteristics
* **Zero Matches ($|M| = 0$)**: An entity has no counterpart in Source 2 or Source 3.
* **Single Match ($|M| = 1$)**: An entity matches exactly one record in Source 2 or Source 3.
* **Multiple Matches ($|M| > 1$)**: An entity matches multiple branches, legal entities, or cross-source records.
* **Subset Rule**: $\text{matched\_entity\_ids}(e) \subseteq \text{candidate\_entity\_ids}(e)$ for all $e \in S_1$.

---

## 2. Tech Stack & Dependencies

| Category | Technologies / Libraries |
|---|---|
| **Core Language** | Python 3.10 / 3.11 / 3.12 / 3.14 (Fully Type-Annotated) |
| **Machine Learning** | [CatBoost](https://catboost.ai/) (Default GBDT), [LightGBM](https://lightgbm.readthedocs.io/), [Scikit-learn](https://scikit-learn.org/) |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **String & Text Similarity** | Levenshtein Distance, Jaccard Token Set, Char N-Gram Cosine, TF-IDF Vectorization |
| **Configuration & Logging** | [PyYAML](https://pyyaml.org/), Python Standard `logging`, Structured JSON Experiment Tracker |
| **Testing & CI** | [Pytest](https://pytest.org/) (26 Test Suites, 555 Unit & Integration Tests) |

---

## 3. Clean System Architecture

### High-Level Data Flow

```text
[ Raw Data: Source 1, Source 2, Source 3, Train Matches ]
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Data Ingestion & Normalization                    │
│   • Schema mapping to canonical entity representation       │
│   • NFKC Unicode casefold & legal suffix expansion          │
│   • Address abbreviation standardizer & fingerprinting      │
└──────────────────────────┬──────────────────────────────────┘
                           │ Canonical Normalized DataFrames
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Multi-Pass Candidate Blocking                     │
│   • Exact name & address fingerprint blocker                │
│   • Inverted token index blocker                            │
│   • Char n-gram TF-IDF cosine similarity blocker            │
│   • Rare token inverted index blocker                       │
│   • Deterministic Candidate Union & Prefix-Safe ID Adapter  │
└──────────────────────────┬──────────────────────────────────┘
                           │ Candidate Pairs DataFrame
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Pairwise Feature Engineering                      │
│   • Wide pair constructor (left_*, right_*)                 │
│   • 29 numeric similarity features extracted                │
└──────────────────────────┬──────────────────────────────────┘
                           │ Feature Matrix X
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: ML Modeling & Hard Negative Mining                 │
│   • Near-miss negative synthesis (suffix & address mutation)│
│   • CatBoost / LightGBM binary classifier training          │
│   • Probability prediction & ModelPredictor bridge          │
└──────────────────────────┬──────────────────────────────────┘
                           │ Match Probabilities
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: Entity-Level Decision & Threshold Optimization     │
│   • Anmol choose_threshold(): optimal Macro F0.5 selection  │
│   • EntityResolver: 0, 1, or multi-match decision resolution│
│   • Candidate subset enforcement & top margin pruning       │
└──────────────────────────┬──────────────────────────────────┘
                           │ Resolved Match Sets {s1 -> [targets]}
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: Unified Evaluation & Submission Generation         │
│   • EntityEvaluator: Macro F0.5, Precision, Recall, EMR     │
│   • SubmissionGenerator: Formats candidate & matching TSVs  │
│   • Official Validator: 100% strict competition validation  │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
 output/matching_results.tsv      output/candidate_pairs.tsv
```

### Subsystem Interaction Diagram

```text
                   ┌────────────────────────────────────────┐
                   │               DataLoader               │
                   └───────────────────┬────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
         ┌─────────────────────┐               ┌─────────────────────┐
         │   DataNormalizer    │               │  AddressNormalizer  │
         │   (normalize_name)  │               │ (normalize_address) │
         └──────────┬──────────┘               └──────────┬──────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │     CandidatePipeline     │
                         │ ┌───────────────────────┐ │
                         │ │ Exact Match Blocker   │ │
                         │ │ Token Inverted Index  │ │
                         │ │ Char N-Gram TF-IDF    │ │
                         │ │ Rare Token Index      │ │
                         │ └───────────────────────┘ │
                         └─────────────┬─────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │   Candidate ID Adapter    │
                         └─────────────┬─────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │    build_pair_features    │
                         │       (29 Features)       │
                         └─────────────┬─────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
        ┌──────────────────────┐                ┌──────────────────────┐
        │   Hard Negatives     │                │   GBDT Classifiers   │
        │      Generator       │                │(CatBoost / LightGBM) │
        └──────────┬───────────┘                └──────────┬───────────┘
                   │                                       │
                   └───────────────────┬───────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │     choose_threshold      │
                         │ (Optimal Macro F0.5 cut)  │
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │      EntityResolver       │
                         │ (0, 1, or Multi Matches)  │
                         └─────────────┬─────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
         ┌─────────────────────┐               ┌─────────────────────┐
         │   EntityEvaluator   │               │ SubmissionGenerator │
         │   (Macro F0.5)      │               │ + Official Validator│
         └─────────────────────┘               └─────────────────────┘
```

---

## 4. Working of the Project (Stage-by-Stage)

### Stage 1: Ingestion & Text/Address Normalization
* **File Ingestion**: [`src/data/loader.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/data/loader.py) ingests tab-separated and comma-separated raw files. Handles dynamic schema translation into the canonical schema:
  `entity_id | business_name | business_address | country | source`
* **Name Normalization**: [`src/data/normalization.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/data/normalization.py) performs Unicode NFKC normalization, case folding, strips accents/punctuation, standardizes symbols (`&` $\to$ `and`), and expands trailing legal forms (`corp` $\to$ `corporation`, `ltd` $\to$ `limited`, `inc` $\to$ `incorporated`, `pvt` $\to$ `private`, `co` $\to$ `company`).
* **Address Normalization**: [`src/data/address.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/data/address.py) expands street suffixes, preserves numbers, removes non-alphanumeric noise, and creates deterministic **alphanumeric address fingerprints** for high-precision hashing.

### Stage 2: Multi-Pass Candidate Blocking
To scale efficiently without losing true matches, [`src/blocking/candidate_pipeline.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/blocking/candidate_pipeline.py) runs 4 complementary blockers:
1. **Exact Blocker**: Matches identical clean business names, clean addresses, and address fingerprints.
2. **Token Inverted Index Blocker**: Builds an inverted index over entity tokens, retrieving candidates sharing significant tokens while filtering high-frequency stopwords.
3. **Character N-Gram TF-IDF Blocker**: Converts business names into character 3-gram and 4-gram TF-IDF vectors; performs sparse matrix multiplication to retrieve top-$K$ candidates above cosine similarity threshold $\theta_{\text{sim}} \ge 0.25$.
4. **Rare Token Blocker**: Indexes words with low corpus frequency (e.g., unique brand names or unusual keywords) to guarantee recall on distinctive entities.
5. **Union & Adapter**: Merges candidates across all passes deterministically and wraps them using [`IdAdapter`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/blocking/candidate_gen.py) to preserve original IDs.

### Stage 3: Pairwise Feature Engineering (29 Features)
[`src/features/features.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/features/features.py) transforms candidate pairs into 29 dense numerical features:

| Feature Name | Description |
|---|---|
| `name_exact_match` | Binary flag indicating exact normalized name equality. |
| `name_lower_match` | Binary flag for case-insensitive match. |
| `name_jaccard_tokens` | Jaccard token overlap between name token sets $\frac{\|A \cap B\|}{\|A \cup B\|}$. |
| `name_jaccard_2gram` – `5gram` | Character $N$-gram Jaccard similarities ($N \in \{2, 3, 4, 5\}$). |
| `name_levenshtein_ratio` | Normalized Levenshtein edit distance between names. |
| `name_token_sort_ratio` | Levenshtein similarity computed after sorting name tokens alphabetically. |
| `name_token_set_ratio` | Levenshtein similarity on intersection vs unique token remainder. |
| `name_len_diff` & `name_len_ratio` | Absolute character length difference and length ratio. |
| `name_token_count_diff` | Absolute difference in token counts. |
| `name_prefix_match_ratio` | Common prefix character length normalized by shortest string. |
| `name_suffix_match_ratio` | Common suffix character length normalized by shortest string. |
| `address_exact_match` | Binary flag indicating exact address string equality. |
| `address_jaccard_tokens` | Jaccard token overlap between street address tokens. |
| `address_jaccard_2gram` – `4gram` | Character $N$-gram Jaccard similarities on street addresses. |
| `address_levenshtein_ratio` | Normalized Levenshtein ratio on street addresses. |
| `address_number_overlap` | Binary indicator if street numbers / building digits match. |
| `country_exact_match` | 1 if country strings match, 0 otherwise. |
| `country_mismatch` | 1 if both countries are non-empty and differ, 0 otherwise. |

### Stage 4: Supervised ML Modeling & Hard Negative Mining
* **Hard Negative Generation**: Standard random negative sampling is too easy for ER. [`generate_hard_negatives()`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/modeling/model.py) generates adversarial near-miss negatives by swapping legal suffixes (`Inc` $\to$ `Group`, `Corp` $\to$ `LLC`) and mutating street names/numbers (`123 Main St` $\to$ `123 Main Ave`).
* **GBDT Classifiers**: Trains **CatBoost** (primary) and **LightGBM** (secondary) models using stratified train/validation splits.
* **Prediction Bridge**: [`src/decision/adapter.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/decision/adapter.py) normalizes predictions into standard schema `[s1_id, target_id, match_score, source]`.

### Stage 5: Decision Resolution & $F_{0.5}$ Threshold Optimization
* **Threshold Search**: [`src/decision/threshold.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/decision/threshold.py) performs grid search over threshold values $[0.00, 1.00]$ to select the optimal cut $T^*$ that maximizes validation Macro $F_{0.5}$.
* **Multi-Match Resolution**: [`src/decision/entity_resolver.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/decision/entity_resolver.py) resolves match sets:
  * Accepts all candidates with $\text{score} \ge T^*$.
  * Supports source-specific thresholds (`threshold_s2`, `threshold_s3`).
  * Applies **Top-Margin Pruning**: If an entity has a match with score $0.98$, candidates below $0.98 - \text{margin}$ can be filtered to eliminate lower-confidence duplicates.
  * Guarantees that every resolved match is an element of the candidate set.

### Stage 6: Unified Evaluation & Submission Generation
* **Unified Metrics**: [`src/evaluation/metrics.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/evaluation/metrics.py) evaluates Macro $F_{0.5}$, Macro Precision, Macro Recall, Exact Match Rate, and Error Cardinalities (0-match, 1-match, multi-match).
* **Submission Emission**: [`src/submission.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/submission.py) writes clean TSVs with pre-write schema validation.
* **Official Validator**: Automatically invokes [`utils/validate_submission.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/utils/validate_submission.py) to guarantee full competition compliance.

---

## 5. Detailed Component & Module Directory

### Directory Structure

```text
Amazon 2026/
├── configs/
│   └── default_config.yaml                  # Unified pipeline hyperparameter & path config
├── data/
│   ├── raw/                                 # Raw input source TSVs (source1, source2, source3)
│   ├── processed/                           # Normalized entity tables
│   └── candidates/                          # Generated candidate pairs
├── experiments/
│   └── threshold_optimization/              # Threshold search metrics & JSON results
├── models/                                  # Trained ML model weights
├── output/                                  # Final competition TSVs
├── submissions/                             # Packaged submission archive storage
├── scripts/
│   ├── evaluate.py                          # Full 6-stage validation & experiment comparison CLI
│   ├── regression_gate_blocking.py          # Strict candidate blocking regression gate
│   ├── run_integrated_training.py           # End-to-end ML model training with hard negatives
│   └── run_threshold_optimization.py        # Micro/macro F0.5 decision threshold tuner
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── pipeline.py                          # Master 6-stage orchestrator (EntityResolutionPipeline)
│   ├── predict.py                           # Standalone batch inference runner
│   ├── submission.py                        # Submission file generation & pre-write verification
│   ├── blocking/                            # Multi-pass candidate blocking subsystem
│   │   ├── __init__.py
│   │   ├── candidate_gen.py                 # Unified blocker interface & ID adapter
│   │   ├── candidate_pipeline.py            # Multi-pass blocker orchestrator
│   │   ├── candidate_union.py               # Deterministic candidate union & deduplication
│   │   ├── candidate_diagnostics.py         # Candidate coverage & recall metrics
│   │   ├── char_ngram_blocking.py           # Character n-gram TF-IDF similarity blocker
│   │   ├── exact_blocking.py                # Exact name & address fingerprint blocker
│   │   └── rare_blocking.py                 # Rare token inverted index blocker
│   ├── data/                                # Data ingestion & text/address normalization
│   │   ├── __init__.py
│   │   ├── loader.py                        # TSV/CSV data loader & schema translator
│   │   ├── normalization.py                 # NFKC text cleaning & legal token normalization
│   │   ├── normalizer.py                    # Multi-column DataFrame normalizer
│   │   └── address.py                       # Address standardizer & fingerprint generator
│   ├── decision/                            # Entity resolution and decision thresholding
│   │   ├── __init__.py
│   │   ├── entity_resolver.py               # Many-to-many entity decision resolver
│   │   ├── threshold.py                     # Anmol's F0.5 threshold optimizer
│   │   └── adapter.py                       # Model-to-EntityResolver bridge
│   ├── evaluation/                          # Unified evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py                       # Exact per-entity Macro F0.5 calculation
│   ├── features/                            # Pairwise feature engineering
│   │   ├── __init__.py
│   │   ├── features.py                      # 29 numeric similarity features
│   │   └── pairwise.py                      # Wide pair builder & feature extractor bridge
│   ├── modeling/                            # Machine learning modeling
│   │   ├── __init__.py
│   │   ├── model.py                         # CatBoost / LightGBM trainers & hard negatives
│   │   ├── trainer.py                       # Training wrapper
│   │   └── predict.py                       # Batch prediction wrapper
│   ├── utils/                               # Shared utilities
│   │   ├── __init__.py
│   │   ├── config.py                        # YAML configuration loader
│   │   ├── logger.py                        # Formatted logger setup
│   │   └── tracker.py                       # Experiment tracking and metric persistence
│   └── validation/                          # Validation harness
│       ├── __init__.py
│       └── harness.py                       # Validation report generator & diff engine
├── tests/                                   # 26 comprehensive test suites (555 unit & integration tests)
├── utils/
│   ├── __init__.py
│   └── validate_submission.py               # Official competition validator
├── run_pipeline.py                          # Top-level executable pipeline script
├── requirements.txt                         # Dependency definitions
└── README.md                                # Project documentation
```

### Module Responsibilities Reference

| Module / Script | Key Class / Functions | Primary Responsibility |
|---|---|---|
| [`src/pipeline.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/pipeline.py) | `EntityResolutionPipeline` | End-to-end execution of all 6 stages; coordinates data, features, models, decisions, and submissions. |
| [`src/data/normalizer.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/data/normalizer.py) | `DataNormalizer` | Normalizes text columns across DataFrames using `normalize_name()` and `normalize_address()`. |
| [`src/data/address.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/data/address.py) | `normalize_address`, `address_fingerprint` | Address abbreviation cleaning and alphanumeric fingerprint generation. |
| [`src/blocking/candidate_pipeline.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/blocking/candidate_pipeline.py) | `CandidatePipeline` | Orchestrates multi-pass candidate generation across exact, token, char n-gram, and rare blockers. |
| [`src/features/features.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/features/features.py) | `build_pair_features` | Computes 29 numerical similarity features on wide candidate pairs. |
| [`src/modeling/model.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/modeling/model.py) | `train_catboost_model`, `generate_hard_negatives` | Mines adversarial hard negatives and trains CatBoost / LightGBM models. |
| [`src/decision/threshold.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/decision/threshold.py) | `choose_threshold` | Optimizes decision threshold on validation probabilities to maximize Macro $F_{0.5}$. |
| [`src/decision/entity_resolver.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/decision/entity_resolver.py) | `EntityResolver` | Resolves probabilities into 0, 1, or multi-match sets with top-margin and source filtering. |
| [`src/evaluation/metrics.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/evaluation/metrics.py) | `EntityEvaluator` | Computes exact competition Macro $F_{0.5}$, Macro Precision, Macro Recall, and Exact Match Rate. |
| [`src/validation/harness.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/validation/harness.py) | `ValidationHarness`, `ValidationReport` | Generates structured validation reports and facilitates experiment comparisons. |
| [`src/submission.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/src/submission.py) | `generate_submission_files` | Exports `matching_results.tsv` and `candidate_pairs.tsv` with pre-write integrity checks. |
| [`utils/validate_submission.py`](file:///c:/Users/Ansh%20jaiswal/OneDrive/Desktop/Amazon%202026/utils/validate_submission.py) | `validate_submission` | Official validator checking header syntax, completeness, subset invariant, and TSV encoding. |

---

## 6. Execution Guide & CLI Commands

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/Ansh-1019/Amazon_ML_2026.git
cd Amazon_ML_2026

# Install dependencies
pip install -r requirements.txt
```

### Running the End-to-End Pipeline
```bash
# Run full pipeline with default config
python run_pipeline.py --config configs/default_config.yaml

# Run with threshold override
python run_pipeline.py --threshold 0.50 --output-dir output
```

### Model Training with Hard Negatives
```bash
python scripts/run_integrated_training.py
```
* **Output**: Generates candidate pairs, mines 200+ hard negatives, trains CatBoost & LightGBM classifiers, and prints validation ROC-AUC / Average Precision.

### Decision Threshold Optimization
```bash
python scripts/run_threshold_optimization.py
```
* **Output**: Searches $[0.00, 1.00]$ for the optimal Macro $F_{0.5}$ cut and saves reports to `experiments/threshold_optimization/threshold_results.json`.

### Running the Blocking Regression Gate
```bash
python scripts/regression_gate_blocking.py
```
* **Output**: Verifies that 100% of ground-truth matches are recovered during blocking with zero true match loss.

### Evaluating Benchmark Experiments
```bash
# Run validation on built-in synthetic benchmark
python scripts/evaluate.py --synthetic --experiment-id baseline_exp

# Compare two experiment reports
python scripts/evaluate.py --compare experiments/exp_A/report.json experiments/exp_B/report.json
```

### Validating Official Submission Files
```bash
python utils/validate_submission.py \
    --matching-results output/matching_results.tsv \
    --candidate-pairs output/candidate_pairs.tsv
```

### Executing Automated Test Suites
Run all **555 unit and integration tests** across 26 test modules:
```bash
python -m pytest tests/ -v
```

---

## 7. Performance & Benchmark Metrics

### A. Candidate Blocking Efficiency
* **Blocking Recall**: `1.0000` (100% true matches recovered)
* **Source 2 Recall**: `1.0000`
* **Source 3 Recall**: `1.0000`
* **Average Candidates / S1 Entity**: `10.45`
* **Zero Candidate Rate**: `0.00%`

### B. Supervised GBDT Classifiers
* **CatBoost Validation ROC-AUC**: `0.9999`
* **CatBoost Average Precision**: `0.9987`
* **LightGBM Validation ROC-AUC**: `0.9997`
* **LightGBM Average Precision**: `0.9959`

### C. Decision Threshold Optimization
* **Optimal Threshold $T^*$**: `0.2000`
* **Validation Macro $F_{0.5}$**: `0.9859`
* **Validation Precision**: `1.0000` (0 False Merges)
* **Validation Recall**: `0.9333`

### D. End-to-End Synthetic Benchmark
* **Headline Macro $F_{0.5}$**: `0.8208`
* **Macro Precision**: `0.8125`
* **Macro Recall**: `0.9167`
* **Exact Match Rate**: `75.00%`
* **Official Validator**: `PASS` (`is_submission_valid: True`)

---

## 8. Configuration Reference (`configs/default_config.yaml`)

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
  min_similarity_threshold: 0.10
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
  threshold: 0.50
  threshold_s2: null
  threshold_s3: null
  top_margin: null

submission:
  matching_filename: "matching_results.tsv"
  candidates_filename: "candidate_pairs.tsv"
```

---

## 9. Competition Submission Compliance

The output generator automatically enforces and verifies all official competition invariants before writing files:

1. **File Names**: Exactly `matching_results.tsv` and `candidate_pairs.tsv`.
2. **Column Names**:
   * Matching file: `source1_entity_id \t matched_entity_ids`
   * Candidates file: `source1_entity_id \t candidate_entity_ids`
3. **Row Count Consistency**: Exactly one row per test Source 1 entity; matching entity IDs must be present in the candidate pairs file for that entity.
4. **Subset Constraint**: Every resolved match is guaranteed to be a candidate:
   $$\text{matched\_entity\_ids}(e) \subseteq \text{candidate\_entity\_ids}(e) \quad \forall e \in S_1$$
5. **Deduplication**: Target IDs are separated by single commas without whitespace or duplicates.
6. **Pure TSV Formatting**: Clean tab-separated output, UTF-8 encoded without quotes.
7. **Empty String Handling**: Entities with zero matches output an empty string for the `matched_entity_ids` field.