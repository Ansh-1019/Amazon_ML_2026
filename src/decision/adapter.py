"""Adapter connecting Model prediction output to EntityResolver input format."""
import pandas as pd


def adapt_model_predictions_for_resolver(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Adapts Model pairwise prediction DataFrame to the format expected by EntityResolver.

    Canonical Mapping:
      source1_entity_id / s1_id -> s1_id
      candidate_entity_id / target_id -> target_id
      probability / match_score -> match_score
      target_source / source -> source

    Guarantees:
      - Clean s1_id, target_id, match_score, source columns.
      - Safe handling of empty DataFrames.
      - Preservation of raw numeric score values and target identities.
    """
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame(columns=["s1_id", "target_id", "match_score", "source"])

    df = predictions_df.copy()

    # Resolve S1 ID column
    s1_col = None
    for col in ("s1_id", "source1_entity_id", "entity_id_s1", "left_id"):
        if col in df.columns:
            s1_col = col
            break

    # Resolve Candidate / Target ID column
    cand_col = None
    for col in ("target_id", "candidate_entity_id", "candidate_id", "matched_id", "s2_id", "s3_id", "right_id"):
        if col in df.columns:
            cand_col = col
            break

    # Resolve Score / Probability column
    score_col = None
    for col in ("match_score", "probability", "score", "prob"):
        if col in df.columns:
            score_col = col
            break

    # Resolve Source column
    source_col = None
    for col in ("source", "target_source", "source_name"):
        if col in df.columns:
            source_col = col
            break

    s1_vals = df[s1_col] if s1_col else df.index.astype(str)
    target_vals = df[cand_col] if cand_col else df.index.astype(str)
    score_vals = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0) if score_col else 0.0

    adapted = pd.DataFrame({
        "s1_id": s1_vals,
        "target_id": target_vals,
        "match_score": score_vals,
    })

    if source_col:
        adapted["source"] = df[source_col].astype(str)

    return adapted
