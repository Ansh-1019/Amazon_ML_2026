"""Blocking and candidate generation subpackage for entity resolution."""
from src.blocking.blocking import (
    DEFAULT_STOPWORDS,
    TokenIndex,
    build_token_index,
    generate_token_candidates,
)
from src.blocking.exact_blocking import (
    ExactIndex,
    build_exact_index,
    generate_exact_candidates,
)
from src.blocking.char_ngram_blocking import (
    CharNgramIndex,
    build_char_ngram_index,
    generate_char_ngram_candidates,
)
from src.blocking.rare_blocking import (
    RareTokenIndex,
    build_rare_token_index,
    generate_rare_token_candidates,
)
from src.blocking.candidate_union import union_candidates
from src.blocking.candidate_pipeline import (
    CandidateBlock,
    CandidateResult,
    CandidatePipeline,
)
from src.blocking.candidate_diagnostics import (
    CandidateDiagnostics,
    diagnose_candidates,
    aggregate_diagnostics,
    split_truth_by_source,
    diagnose_by_source,
    compare_candidate_blocks,
)
from src.blocking.evaluation import (
    MissedPair,
    EvaluationResult,
    evaluate_pipeline,
)
from src.blocking.candidate_gen import (
    CandidateGenerator,
    IdAdapter,
)

__all__ = [
    "DEFAULT_STOPWORDS",
    "TokenIndex",
    "build_token_index",
    "generate_token_candidates",
    "ExactIndex",
    "build_exact_index",
    "generate_exact_candidates",
    "CharNgramIndex",
    "build_char_ngram_index",
    "generate_char_ngram_candidates",
    "RareTokenIndex",
    "build_rare_token_index",
    "generate_rare_token_candidates",
    "union_candidates",
    "CandidateBlock",
    "CandidateResult",
    "CandidatePipeline",
    "CandidateDiagnostics",
    "diagnose_candidates",
    "aggregate_diagnostics",
    "split_truth_by_source",
    "diagnose_by_source",
    "compare_candidate_blocks",
    "MissedPair",
    "EvaluationResult",
    "evaluate_pipeline",
    "CandidateGenerator",
    "IdAdapter",
]
