#!/usr/bin/env python3
"""
Official Submission Validator for Amazon ML Challenge 2026 - Entity Resolution

Validates:
1. File existence and headers (matching_results.tsv and candidate_pairs.tsv).
2. Tab-separation without unintended quotes or corrupt formatting.
3. Every Source 1 test entity is present exactly once (no missing/duplicate S1 rows).
4. Candidate and matched ID lists contain only valid Source 2 and Source 3 IDs (no Source 1 IDs).
5. No duplicate IDs within any entity's match or candidate list.
6. Final matches are a strict subset of candidate pairs for every Source 1 entity.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


def load_tsv(filepath: Path, expected_header: Tuple[str, str]) -> Dict[str, List[str]]:
    """
    Parses a TSV file with format: source1_entity_id <TAB> comma_separated_ids
    Returns dictionary mapping source1_entity_id -> list of target entity IDs.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Submission file does not exist: {filepath}")

    results: Dict[str, List[str]] = {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Submission file is empty: {filepath}")

    header_line = lines[0].rstrip("\r\n")
    expected_header_str = f"{expected_header[0]}\t{expected_header[1]}"
    if header_line != expected_header_str:
        raise ValueError(
            f"Invalid header in {filepath.name}. Expected '{expected_header_str}', got '{header_line}'"
        )

    for line_num, line in enumerate(lines[1:], start=2):
        line_clean = line.rstrip("\r\n")
        if not line_clean:
            continue

        parts = line_clean.split("\t")
        if len(parts) > 2:
            raise ValueError(f"Line {line_num} in {filepath.name} has more than 2 tab-separated columns: {line_clean}")
        
        s1_id = parts[0].strip()
        
        # Check for unintended surrounding quotes around s1_id
        if (s1_id.startswith('"') and s1_id.endswith('"')) or (s1_id.startswith("'") and s1_id.endswith("'")):
            raise ValueError(f"Line {line_num} in {filepath.name} has quoted s1_id: {s1_id}")

        if s1_id in results:
            raise ValueError(f"Duplicate Source 1 entity ID '{s1_id}' on line {line_num} in {filepath.name}")

        raw_ids = parts[1].strip() if len(parts) > 1 else ""
        
        # Check for unintended quotes around ID list
        if (raw_ids.startswith('"') and raw_ids.endswith('"')) or (raw_ids.startswith("'") and raw_ids.endswith("'")):
            raise ValueError(f"Line {line_num} in {filepath.name} has quoted ID list: {raw_ids}")

        if not raw_ids:
            target_ids = []
        else:
            target_ids = [idx.strip() for idx in raw_ids.split(",") if idx.strip()]
            # Check individual ID quotes
            for idx in target_ids:
                if (idx.startswith('"') and idx.endswith('"')) or (idx.startswith("'") and idx.endswith("'")):
                    raise ValueError(f"Line {line_num} in {filepath.name} contains quoted target ID: {idx}")

        results[s1_id] = target_ids

    return results


def validate_submission_files(
    matching_filepath: Path,
    candidates_filepath: Path,
    valid_s1_ids: Optional[Set[str]] = None,
    valid_target_ids: Optional[Set[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validates matching_results.tsv and candidate_pairs.tsv against all competition rules.
    Returns (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    # 1. Parse files
    try:
        matches = load_tsv(matching_filepath, ("source1_entity_id", "matched_entity_ids"))
    except Exception as e:
        errors.append(f"Failed parsing matching_results.tsv: {str(e)}")
        matches = {}

    try:
        candidates = load_tsv(candidates_filepath, ("source1_entity_id", "candidate_entity_ids"))
    except Exception as e:
        errors.append(f"Failed parsing candidate_pairs.tsv: {str(e)}")
        candidates = {}

    if errors:
        return False, errors

    # 2. Source 1 Test Entity Coverage Check
    if valid_s1_ids is not None:
        missing_match_s1 = valid_s1_ids - set(matches.keys())
        if missing_match_s1:
            errors.append(f"matching_results.tsv is missing {len(missing_match_s1)} Source 1 test entities. Example: {list(missing_match_s1)[:3]}")

        missing_cand_s1 = valid_s1_ids - set(candidates.keys())
        if missing_cand_s1:
            errors.append(f"candidate_pairs.tsv is missing {len(missing_cand_s1)} Source 1 test entities. Example: {list(missing_cand_s1)[:3]}")

        extra_match_s1 = set(matches.keys()) - valid_s1_ids
        if extra_match_s1:
            errors.append(f"matching_results.tsv contains {len(extra_match_s1)} invalid/unknown Source 1 IDs. Example: {list(extra_match_s1)[:3]}")

        extra_cand_s1 = set(candidates.keys()) - valid_s1_ids
        if extra_cand_s1:
            errors.append(f"candidate_pairs.tsv contains {len(extra_cand_s1)} invalid/unknown Source 1 IDs. Example: {list(extra_cand_s1)[:3]}")

    # Ensure keys match between candidates and matches
    if set(matches.keys()) != set(candidates.keys()):
        errors.append("Mismatch in Source 1 entity IDs present between matching_results.tsv and candidate_pairs.tsv")

    # 3. Row-by-Row Checks
    all_s1_keys = set(matches.keys())

    for s1_id in sorted(all_s1_keys):
        match_ids = matches.get(s1_id, [])
        cand_ids = candidates.get(s1_id, [])

        # Check duplicate IDs within match_ids
        if len(match_ids) != len(set(match_ids)):
            seen = set()
            dups = [x for x in match_ids if x in seen or seen.add(x)]
            errors.append(f"Duplicate matched IDs for entity '{s1_id}': {dups}")

        # Check duplicate IDs within cand_ids
        if len(cand_ids) != len(set(cand_ids)):
            seen = set()
            dups = [x for x in cand_ids if x in seen or seen.add(x)]
            errors.append(f"Duplicate candidate IDs for entity '{s1_id}': {dups}")

        # Check valid target IDs (Only S2 and S3 IDs allowed; S1 IDs forbidden)
        if valid_target_ids is not None:
            invalid_matches = [m for m in match_ids if m not in valid_target_ids]
            if invalid_matches:
                errors.append(f"Invalid target ID(s) in matched_entity_ids for entity '{s1_id}': {invalid_matches}")

            invalid_cands = [c for c in cand_ids if c not in valid_target_ids]
            if invalid_cands:
                errors.append(f"Invalid target ID(s) in candidate_entity_ids for entity '{s1_id}': {invalid_cands}")

        # 4. Subset Check: Final matches must be a subset of candidates
        not_in_candidates = set(match_ids) - set(cand_ids)
        if not_in_candidates:
            errors.append(
                f"Entity '{s1_id}' has matched IDs not present in candidate list: {sorted(list(not_in_candidates))}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    parser = argparse.ArgumentParser(description="Official Validator for Amazon ML Challenge Submission Files")
    parser.add_argument("--matching-results", type=str, default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate-pairs", type=str, default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--s1-ids-file", type=str, default=None, help="Optional text/JSON file containing valid s1 IDs")
    parser.add_argument("--target-ids-file", type=str, default=None, help="Optional text/JSON file containing valid S2/S3 IDs")
    args = parser.parse_args()

    matching_path = Path(args.matching_results)
    candidates_path = Path(args.candidate_pairs)

    valid_s1_ids = None
    if args.s1_ids_file and Path(args.s1_ids_file).exists():
        with open(args.s1_ids_file, "r", encoding="utf-8") as f:
            valid_s1_ids = set(line.strip() for line in f if line.strip())

    valid_target_ids = None
    if args.target_ids_file and Path(args.target_ids_file).exists():
        with open(args.target_ids_file, "r", encoding="utf-8") as f:
            valid_target_ids = set(line.strip() for line in f if line.strip())

    is_valid, errors = validate_submission_files(matching_path, candidates_path, valid_s1_ids, valid_target_ids)

    if is_valid:
        print("PASS: Submission validation successful! All competition rules satisfied.")
        sys.exit(0)
    else:
        print("FAIL: Submission validation errors encountered:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
