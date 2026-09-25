"""
Test suite for the submission / output layer.

Covers every competition rule enforced by
  src/submission.py  (pre-write validation + file generation)
  utils/validate_submission.py  (official post-write validator)

Test matrix:
  ✓ singleton entity with single candidate & match
  ✓ one match selected from multiple candidates
  ✓ multiple matches (S2 + S3 per entity, multiple S1 entities)
  ✓ duplicate IDs (auto-dedup by generator, detection by validator)
  ✓ invalid / forbidden target IDs
  ✓ match not present in candidate list
  ✓ missing Source 1 entity
  ✓ empty match list (row present, ID list blank)
  ✓ empty candidate list
  ✓ raw TSV byte-level integrity (tabs, no quoting)
  ✓ official validator wrapper round-trip
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.submission import (
    format_id_list,
    generate_submission_files,
    run_official_validator,
    SubmissionValidationError,
)
from utils.validate_submission import validate_submission_files


VALIDATOR = Path("utils/validate_submission.py")


class TestFormatIdList(unittest.TestCase):
    """Unit tests for the format_id_list helper."""

    def test_empty_list(self):
        self.assertEqual(format_id_list([]), "")

    def test_none_input(self):
        self.assertEqual(format_id_list(None), "")

    def test_basic(self):
        self.assertEqual(format_id_list(["a", "b", "c"]), "a,b,c")

    def test_strips_whitespace_and_quotes(self):
        self.assertEqual(
            format_id_list([" s2_1 ", "'s3_2'", '"s2_3"']),
            "s2_1,s3_2,s2_3",
        )

    def test_deduplicates_preserving_order(self):
        self.assertEqual(
            format_id_list(["x", "y", "x", "z", "y"]),
            "x,y,z",
        )

    def test_skips_none_and_blank(self):
        self.assertEqual(
            format_id_list(["a", None, "", "  ", "b"]),
            "a,b",
        )


class TestSubmissionGeneration(unittest.TestCase):
    """Integration tests: generation + official validator."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = Path(self.tmp) / "output"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    # ----- helper ----------------------------------------------------------
    def _gen(self, s1, cands, matches, **kw):
        return generate_submission_files(
            test_s1_ids=s1,
            candidate_pairs_map=cands,
            matched_results_map=matches,
            output_dir=self.out,
            **kw,
        )

    def _validate(self, mp, cp, s1_ids=None, target_ids=None):
        return validate_submission_files(
            mp, cp,
            valid_s1_ids=s1_ids,
            valid_target_ids=target_ids,
        )

    # ----- positive cases --------------------------------------------------
    def test_singleton(self):
        """Single S1, single candidate, single match."""
        mp, cp = self._gen(["s1_1"], {"s1_1": ["s2_1"]}, {"s1_1": ["s2_1"]})
        ok, errs = self._validate(mp, cp, {"s1_1"}, {"s2_1"})
        self.assertTrue(ok, errs)

    def test_one_match_from_many_candidates(self):
        """Pick one match out of several candidates."""
        mp, cp = self._gen(
            ["s1_1"],
            {"s1_1": ["s2_1", "s2_2", "s3_1"]},
            {"s1_1": ["s2_1"]},
        )
        ok, errs = self._validate(mp, cp, {"s1_1"}, {"s2_1", "s2_2", "s3_1"})
        self.assertTrue(ok, errs)

    def test_multiple_matches_multiple_entities(self):
        """Two S1 entities each matching a mix of S2/S3."""
        mp, cp = self._gen(
            ["s1_a", "s1_b"],
            {"s1_a": ["s2_1", "s3_1", "s3_2"], "s1_b": ["s2_5", "s3_5"]},
            {"s1_a": ["s2_1", "s3_1"], "s1_b": ["s2_5", "s3_5"]},
        )
        targets = {"s2_1", "s3_1", "s3_2", "s2_5", "s3_5"}
        ok, errs = self._validate(mp, cp, {"s1_a", "s1_b"}, targets)
        self.assertTrue(ok, errs)

    def test_empty_match_list(self):
        """Entity with candidates but no matches (empty match_ids)."""
        mp, cp = self._gen(
            ["s1_1"],
            {"s1_1": ["s2_1", "s3_1"]},
            {"s1_1": []},
        )
        ok, errs = self._validate(mp, cp, {"s1_1"}, {"s2_1", "s3_1"})
        self.assertTrue(ok, errs)

    def test_empty_candidate_and_match_lists(self):
        """Entity with no candidates and no matches."""
        mp, cp = self._gen(["s1_1"], {"s1_1": []}, {"s1_1": []})
        ok, errs = self._validate(mp, cp, {"s1_1"})
        self.assertTrue(ok, errs)

    def test_official_validator_wrapper_pass(self):
        """Round-trip through the subprocess wrapper."""
        mp, cp = self._gen(
            ["s1_1", "s1_2"],
            {"s1_1": ["s2_1"], "s1_2": ["s3_2"]},
            {"s1_1": ["s2_1"], "s1_2": []},
        )
        passed, log = run_official_validator(mp, cp, validator_script=VALIDATOR)
        self.assertTrue(passed, log)

    # ----- negative cases --------------------------------------------------
    def test_duplicate_ids_auto_deduplicated(self):
        """Generator deduplicates; validator sees clean output."""
        mp, cp = self._gen(
            ["s1_1"],
            {"s1_1": ["s2_1", "s2_1", "s3_1"]},
            {"s1_1": ["s2_1", "s2_1"]},
            strict=False,  # skip pre-write so we test the dedup in format_id_list
        )
        ok, errs = self._validate(mp, cp, {"s1_1"}, {"s2_1", "s3_1"})
        self.assertTrue(ok, errs)

    def test_duplicate_ids_detected_by_validator(self):
        """Manually craft a TSV with dupes — validator must reject."""
        # First create a valid candidate file
        _, cp = self._gen(
            ["s1_1"],
            {"s1_1": ["s2_1", "s3_1"]},
            {"s1_1": ["s2_1"]},
        )
        # Write a bad matching file with duplicate IDs
        dup = self.out / "dup_matching.tsv"
        dup.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "s1_1\ts2_1,s2_1\n",
            encoding="utf-8",
        )
        ok, errs = self._validate(dup, cp, {"s1_1"})
        self.assertFalse(ok)
        self.assertTrue(any("Duplicate" in e for e in errs))

    def test_duplicate_ids_strict_prewrite_raises(self):
        """strict=True catches dupes before anything is written."""
        with self.assertRaises(SubmissionValidationError) as ctx:
            self._gen(
                ["s1_1"],
                {"s1_1": ["s2_1", "s2_1"]},
                {"s1_1": ["s2_1"]},
                strict=True,
            )
        self.assertTrue(any("Duplicate" in e for e in ctx.exception.errors))

    def test_invalid_target_ids(self):
        """IDs not in valid_target_ids are rejected."""
        mp, cp = self._gen(
            ["s1_1"],
            {"s1_1": ["s2_1", "s1_BAD"]},
            {"s1_1": ["s2_1"]},
            strict=False,
        )
        ok, errs = self._validate(mp, cp, {"s1_1"}, {"s2_1"})
        self.assertFalse(ok)
        self.assertTrue(any("Invalid target ID" in e for e in errs))

    def test_invalid_target_ids_strict_prewrite_raises(self):
        """strict=True rejects invalid targets before writing."""
        with self.assertRaises(SubmissionValidationError) as ctx:
            self._gen(
                ["s1_1"],
                {"s1_1": ["s2_1", "UNKNOWN"]},
                {"s1_1": ["s2_1"]},
                valid_target_ids={"s2_1"},
                strict=True,
            )
        self.assertTrue(any("invalid candidate" in e for e in ctx.exception.errors))

    def test_match_not_in_candidates(self):
        """Matched ID that doesn't appear in candidates → fail."""
        mp, cp = self._gen(
            ["s1_1"],
            {"s1_1": ["s2_1"]},
            {"s1_1": ["s2_MISSING"]},
            strict=False,
        )
        ok, errs = self._validate(mp, cp, {"s1_1"})
        self.assertFalse(ok)
        self.assertTrue(any("not present in candidate list" in e for e in errs))

    def test_match_not_in_candidates_strict_prewrite_raises(self):
        """strict=True catches subset violation before writing."""
        with self.assertRaises(SubmissionValidationError) as ctx:
            self._gen(
                ["s1_1"],
                {"s1_1": ["s2_1"]},
                {"s1_1": ["s2_MISSING"]},
                strict=True,
            )
        self.assertTrue(
            any("not in candidate list" in e for e in ctx.exception.errors)
        )

    def test_missing_source1_entity(self):
        """Fewer rows than required S1 set → validator rejects."""
        mp, cp = self._gen(
            ["s1_1", "s1_2"],        # only 2 written
            {"s1_1": ["s2_1"], "s1_2": ["s2_2"]},
            {"s1_1": ["s2_1"], "s1_2": []},
        )
        # validate against 3 required
        ok, errs = self._validate(mp, cp, {"s1_1", "s1_2", "s1_3"})
        self.assertFalse(ok)
        self.assertTrue(any("missing" in e.lower() for e in errs))

    # ----- raw TSV integrity ------------------------------------------------
    def test_tsv_bytes_no_quoting(self):
        """Files must be pure tab-separated, no CSV quoting artefacts."""
        mp, cp = self._gen(
            ["s1_1"],
            {"s1_1": ["s2_1", "s3_1"]},
            {"s1_1": ["s2_1"]},
        )
        for path in (mp, cp):
            raw = path.read_bytes()
            self.assertNotIn(b'"', raw, f"Unexpected double-quote in {path.name}")
            self.assertNotIn(b"'", raw, f"Unexpected single-quote in {path.name}")
            # header + 1 data row + trailing newline
            lines = raw.rstrip(b"\n").split(b"\n")
            self.assertEqual(len(lines), 2)
            for line in lines:
                cols = line.split(b"\t")
                self.assertEqual(len(cols), 2, f"Expected 2 tab-columns in {path.name}")


if __name__ == "__main__":
    unittest.main()
