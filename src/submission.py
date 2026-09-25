"""
Submission / Output Layer for Amazon ML Challenge 2026 — Business Entity Resolution

Generates:
  output/matching_results.tsv   (source1_entity_id <TAB> matched_entity_ids)
  output/candidate_pairs.tsv    (source1_entity_id <TAB> candidate_entity_ids)

Guarantees enforced *before* writing:
  - Tab-separated, no quoting, no BOM.
  - Exactly one row per Source 1 test entity.
  - No duplicate target IDs within any row's ID list.
  - Final matches are a strict subset of that entity's candidate IDs.
  - Only IDs from valid_target_ids (when supplied) are emitted.
  - No stray quoting or corrupt comma-separated ID lists.
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Set, Union, Optional, Tuple

logger = logging.getLogger("pipeline.submission")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_id_list(ids: Union[List[str], Set[str], Tuple[str, ...], None]) -> str:
    """
    Turn an iterable of target entity IDs into a clean, unquoted,
    order-preserving, deduplicated, comma-separated string.

    Handles None values, empty strings, and accidentally-quoted IDs.
    """
    if not ids:
        return ""

    seen: set = set()
    clean: list = []
    for raw in ids:
        if raw is None:
            continue
        tok = str(raw).strip().strip("\"'")
        if tok and tok not in seen:
            seen.add(tok)
            clean.append(tok)
    return ",".join(clean)


def _sanitise_s1_ids(raw_ids: List[str]) -> List[str]:
    """Deduplicate and strip Source-1 IDs while preserving order."""
    seen: set = set()
    out: list = []
    for s1 in raw_ids:
        s = str(s1).strip().strip("\"'")
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Pre-write validation
# ---------------------------------------------------------------------------

class SubmissionValidationError(Exception):
    """Raised when pre-write validation catches a rule violation."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(
            "Submission pre-write validation failed:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )


def _pre_validate(
    s1_ids: List[str],
    candidates: Dict[str, List[str]],
    matches: Dict[str, List[str]],
    valid_target_ids: Optional[Set[str]] = None,
) -> List[str]:
    """
    Return a list of error strings (empty ⇒ all OK).

    Checks:
      1. matched IDs ⊆ candidate IDs  (per entity)
      2. no duplicate IDs inside any list
      3. all target IDs ∈ valid_target_ids (when provided)
    """
    errors: List[str] = []
    for s1 in s1_ids:
        cand_raw = candidates.get(s1, [])
        match_raw = matches.get(s1, [])

        cand_set: set = set()
        for c in cand_raw:
            tok = str(c).strip().strip("\"'") if c is not None else ""
            if not tok:
                continue
            if tok in cand_set:
                errors.append(
                    f"Duplicate candidate ID '{tok}' for entity '{s1}'"
                )
            cand_set.add(tok)

        match_set: set = set()
        for m in match_raw:
            tok = str(m).strip().strip("\"'") if m is not None else ""
            if not tok:
                continue
            if tok in match_set:
                errors.append(
                    f"Duplicate matched ID '{tok}' for entity '{s1}'"
                )
            match_set.add(tok)

        not_in_cand = match_set - cand_set
        if not_in_cand:
            errors.append(
                f"Entity '{s1}': matched IDs not in candidate list: "
                f"{sorted(not_in_cand)}"
            )

        if valid_target_ids is not None:
            bad_cands = cand_set - valid_target_ids
            if bad_cands:
                errors.append(
                    f"Entity '{s1}': invalid candidate IDs: {sorted(bad_cands)}"
                )
            bad_matches = match_set - valid_target_ids
            if bad_matches:
                errors.append(
                    f"Entity '{s1}': invalid matched IDs: {sorted(bad_matches)}"
                )

    return errors


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

def generate_submission_files(
    test_s1_ids: List[str],
    candidate_pairs_map: Dict[str, List[str]],
    matched_results_map: Dict[str, List[str]],
    output_dir: Union[str, Path] = "output",
    valid_target_ids: Optional[Set[str]] = None,
    matching_filename: str = "matching_results.tsv",
    candidates_filename: str = "candidate_pairs.tsv",
    strict: bool = True,
) -> Tuple[Path, Path]:
    """
    Generate both TSV submission files from mapping dictionaries.

    Args:
        test_s1_ids:          Ordered list of every Source 1 test entity ID.
        candidate_pairs_map:  s1_id → list of candidate entity IDs.
        matched_results_map:  s1_id → list of matched entity IDs.
        output_dir:           Directory for the TSV files.
        valid_target_ids:     Optional whitelist of legal S2/S3 IDs.
        matching_filename:    Output filename (default: matching_results.tsv).
        candidates_filename:  Output filename (default: candidate_pairs.tsv).
        strict:               If True (default), raise SubmissionValidationError
                              on any rule violation *before* writing.

    Returns:
        (matching_results_path, candidate_pairs_path)

    Raises:
        SubmissionValidationError if strict=True and a rule is violated.
    """
    s1_ids = _sanitise_s1_ids(test_s1_ids)

    # --- pre-write checks ---------------------------------------------------
    if strict:
        errors = _pre_validate(
            s1_ids, candidate_pairs_map, matched_results_map, valid_target_ids
        )
        if errors:
            raise SubmissionValidationError(errors)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    matching_file = out / matching_filename
    candidates_file = out / candidates_filename

    # Write matching_results.tsv
    with open(matching_file, "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_ids:
            fh.write(f"{s1}\t{format_id_list(matched_results_map.get(s1, []))}\n")

    # Write candidate_pairs.tsv
    with open(candidates_file, "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_ids:
            fh.write(f"{s1}\t{format_id_list(candidate_pairs_map.get(s1, []))}\n")

    logger.info(
        "Generated submission files in '%s': %s, %s",
        out, matching_filename, candidates_filename,
    )
    return matching_file, candidates_file



# ---------------------------------------------------------------------------
# Class wrapper for pipeline integration
# ---------------------------------------------------------------------------

class SubmissionGenerator:
    """Class interface for generating and validating competition submissions."""

    def __init__(self, output_dir: Union[str, Path] = "output"):
        self.output_dir = Path(output_dir)

    def generate(
        self,
        test_s1_ids: List[str],
        candidate_pairs_map: Dict[str, List[str]],
        matched_results_map: Dict[str, List[str]],
        valid_target_ids: Optional[Set[str]] = None,
        matching_filename: str = "matching_results.tsv",
        candidates_filename: str = "candidate_pairs.tsv",
        strict: bool = True,
    ) -> Tuple[Path, Path]:
        """Generates matching_results.tsv and candidate_pairs.tsv."""
        return generate_submission_files(
            test_s1_ids=test_s1_ids,
            candidate_pairs_map=candidate_pairs_map,
            matched_results_map=matched_results_map,
            output_dir=self.output_dir,
            valid_target_ids=valid_target_ids,
            matching_filename=matching_filename,
            candidates_filename=candidates_filename,
            strict=strict,
        )

    def validate(
        self,
        matching_filepath: Union[str, Path] = "output/matching_results.tsv",
        candidates_filepath: Union[str, Path] = "output/candidate_pairs.tsv",
        s1_ids_file: Optional[Union[str, Path]] = None,
        target_ids_file: Optional[Union[str, Path]] = None,
        validator_script: Union[str, Path] = "utils/validate_submission.py",
    ) -> Tuple[bool, str]:
        """Runs the official validator."""
        return run_official_validator(
            matching_filepath=matching_filepath,
            candidates_filepath=candidates_filepath,
            s1_ids_file=s1_ids_file,
            target_ids_file=target_ids_file,
            validator_script=validator_script,
        )

# ---------------------------------------------------------------------------
# Official-validator wrapper
# ---------------------------------------------------------------------------

def run_official_validator(
    matching_filepath: Union[str, Path] = "output/matching_results.tsv",
    candidates_filepath: Union[str, Path] = "output/candidate_pairs.tsv",
    s1_ids_file: Optional[Union[str, Path]] = None,
    target_ids_file: Optional[Union[str, Path]] = None,
    validator_script: Union[str, Path] = "utils/validate_submission.py",
) -> Tuple[bool, str]:
    """
    Shell out to the **official** validator (utils/validate_submission.py).

    Returns:
        (passed: bool, combined_stdout_stderr: str)
    """
    validator_p = Path(validator_script)
    if not validator_p.exists():
        return False, f"Official validator not found at '{validator_p}'"

    cmd = [
        sys.executable,
        str(validator_p),
        "--matching-results", str(matching_filepath),
        "--candidate-pairs", str(candidates_filepath),
    ]
    if s1_ids_file:
        cmd += ["--s1-ids-file", str(s1_ids_file)]
    if target_ids_file:
        cmd += ["--target-ids-file", str(target_ids_file)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    log = result.stdout.strip()
    if result.stderr.strip():
        log += f"\nStderr:\n{result.stderr.strip()}"
    return result.returncode == 0, log


# ---------------------------------------------------------------------------
# CLI entry-point: synthetic demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Generating Synthetic Submission Files ===")

    synthetic_s1 = ["s1_001", "s1_002", "s1_003", "s1_004", "s1_005"]
    synthetic_cands = {
        "s1_001": ["s2_101", "s3_201"],
        "s1_002": ["s2_102", "s3_202", "s3_203"],
        "s1_003": ["s2_103"],
        "s1_004": ["s2_104", "s3_204"],
        "s1_005": [],  # entity with zero candidates
    }
    synthetic_matches = {
        "s1_001": ["s2_101", "s3_201"],  # multiple matches
        "s1_002": ["s2_102"],            # one match out of 3 candidates
        "s1_003": ["s2_103"],            # singleton
        "s1_004": [],                    # no match (candidates exist)
        "s1_005": [],                    # no match, no candidates
    }

    mp, cp = generate_submission_files(
        test_s1_ids=synthetic_s1,
        candidate_pairs_map=synthetic_cands,
        matched_results_map=synthetic_matches,
        output_dir="output",
        strict=True,
    )
    print(f"  {mp}")
    print(f"  {cp}")

    print("\n=== Official Validator ===")
    passed, log = run_official_validator(mp, cp)
    print(log)
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
