# Anmol Assigned Work

## Role
Anmol is responsible for the model-side entity resolution pipeline for the business matching challenge.

## Scope of Work
Anmol’s tasks focus on the core matching logic, not the final submission orchestration or team-wide integration flow.

### 1. Feature Engineering
Implement and validate pairwise similarity features for candidate matching.

Main area:
- `business_entity_resolution/src/features.py`

Tasks:
- compare names, addresses, cities, countries, and numeric fields
- generate text similarity, token overlap, n-gram similarity, and exact-match signals
- normalize paired columns such as left/right or source/target fields
- ensure the feature matrix is numeric and model-ready

### 2. Baseline Model Training
Train and compare baseline classifiers for pairwise match scoring.

Main area:
- `business_entity_resolution/src/model.py`

Tasks:
- train CatBoost and LightGBM models
- compare validation results using AUC, average precision, accuracy, and log loss
- validate that the model can learn from pairwise comparison features

### 3. Hard Negative Generation
Create realistic near-miss negatives to reduce false merges.

Main area:
- `business_entity_resolution/src/model.py`

Tasks:
- mutate business names and addresses to create ambiguous non-matching pairs
- simulate the common false-merge failure mode in entity resolution
- improve model robustness against highly similar but distinct entities

### 4. Threshold Optimization
Choose a probability cutoff tuned to precision-heavy matching.

Main area:
- `business_entity_resolution/src/threshold.py`

Tasks:
- optimize threshold using F0.5 instead of generic F1
- keep false merges penalized more strongly than missed matches
- select a threshold appropriate for the competition metric

### 5. Entity-Level Aggregation
Turn pairwise predictions into final entity-to-entity matches.

Main area:
- `business_entity_resolution/src/threshold.py`

Tasks:
- aggregate individual pair scores into final entity lists
- deduplicate matches
- keep only confident pair predictions above threshold

### 6. Validation and Regression Testing
Protect model logic with automated checks.

Main areas:
- `tests/test_baseline_model.py`
- `tests/test_evaluate_metric.py`
- `tests/test_threshold.py`

Tasks:
- verify feature quality
- verify baseline model behavior
- verify hard-negative generation
- verify F0.5 threshold logic
- verify entity aggregation logic

### 7. Integration Hygiene
Ensure the implementation works cleanly as a package and import path.

Main areas:
- `business_entity_resolution/__init__.py`
- `business_entity_resolution/src/__init__.py`
- `pytest.ini`
- `tests/conftest.py`

Tasks:
- keep project imports consistent
- avoid duplicate or conflicting package paths
- ensure tests run from the repository root

---

## Completed Work
The following tasks are already implemented and validated:

- pairwise feature engineering
- hard-negative generation
- CatBoost and LightGBM training comparison
- F0.5-aware threshold logic
- entity aggregation logic
- regression tests for matching model functionality
- import-path and package consistency fixes

---

## Remaining Work
The remaining work for Anmol is mainly execution and handoff:

1. validate the real dataset available in `dataset/`
2. run the model pipeline on actual training/testing records
3. tune threshold on real validation data
4. generate final entity-level matches
5. measure output using the competition metric in `evaluate.py`
6. hand off final model output to the rest of the team for full workflow integration

---

## Final Note
Anmol’s primary responsibility is the model and matching layer of the pipeline. Once this is working on real data, the output can be handed over to the remaining project stages for downstream integration and final submission preparation.
