"""
Integration tests for Maithili's normalization, imported from src.data.normalization.

These tests are ported verbatim from project-branch2/tests/test_normalization.py.
The ONLY change is the import path:
    was:  from src.normalization import ...      (branch2 package)
    now:  from src.data.normalization import ...  (main project package)

No test assertions were modified. If a test fails here but passed in branch2,
the integration introduced a bug — fix the import/wiring, never the algorithm.
"""

import pytest
from src.data.normalization import normalize_basic, normalize_name, tokenize_name
from src.data import DataNormalizer


# ---------------------------------------------------------------------------
# normalize_basic
# ---------------------------------------------------------------------------

def test_normalize_basic_lowercase():
    assert normalize_basic("RAJ INVESTMENTS LLP") == "raj investments llp"


def test_normalize_basic_whitespace():
    assert normalize_basic("  Raj   Investments LLP  ") == "raj investments llp"


def test_normalize_basic_empty_string():
    assert normalize_basic("") == ""


def test_normalize_basic_none():
    assert normalize_basic(None) == ""


def test_normalize_basic_preserves_existing_behavior():
    assert normalize_basic(123) == "123"
    assert normalize_basic("  STRAẞE & ＡＢＣ Pvt.  ") == "straße & ａｂｃ pvt."


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------

def test_normalize_name_case():
    assert normalize_name("STRAẞE Trading") == "strasse trading"


def test_normalize_name_whitespace():
    assert normalize_name(" \tRaj\n Investments\u00a0 LLP  ") == "raj investments llp"


def test_normalize_name_punctuation():
    assert normalize_name("O'Neil\u2013Smith \u201cTrading\u201d") == 'o\'neil-smith "trading"'
    assert normalize_name("A/B, C+D #1 (East)") == "a/b, c+d #1 (east)"
    assert normalize_name("Acme-Tools") != normalize_name("Acme Tools")


def test_normalize_name_ampersand():
    assert normalize_name("A & B") == normalize_name("A and B") == "a and b"
    assert normalize_name("R&D AT&T") == "r&d at&t"
    assert normalize_name("A & & B") == "a and and b"


def test_normalize_name_abbreviations():
    pairs = [
        ("ABC Pvt. Ltd.", "ABC Private Limited"),
        ("ABC pvt ltd", "ABC Private Limited"),
        ("ABC Corp.", "ABC Corporation"),
        ("ABC corp", "ABC Corporation"),
        ("ABC Inc.", "ABC Incorporated"),
        ("ABC inc", "ABC Incorporated"),
        ("ABC Co.", "ABC Company"),
        ("ABC Co Ltd", "ABC Company Limited"),
    ]
    for abbreviated, expanded in pairs:
        assert normalize_name(abbreviated) == normalize_name(expanded), (
            f"Mismatch: {abbreviated!r} → {normalize_name(abbreviated)!r} "
            f"vs {expanded!r} → {normalize_name(expanded)!r}"
        )


def test_normalize_name_avoids_ambiguous_expansion():
    assert normalize_name("Co Op Trading") == "co op trading"
    assert normalize_name("Acme Co") == "acme co"
    assert normalize_name("Co") == "co"
    assert normalize_name("Corp Design Studio") == "corp design studio"
    assert normalize_name("Acme-Co.") == "acme-co."
    assert normalize_name("Acme Ltd") != normalize_name("Other Ltd")
    assert normalize_name("Acme Ltd") != normalize_name("Acme")


def test_normalize_name_unicode():
    assert normalize_name("ＡＢＣ Ｐｖｔ． Ｌｔｄ．") == "abc private limited"
    assert normalize_name("Cafe\u0301") == normalize_name("Café") == "café"
    assert normalize_name("Café") != normalize_name("Cafe")


def test_normalize_name_non_latin():
    # Illustrative multilingual fixtures, not asserted dataset records.
    assert normalize_name("  श्री गणेश ट्रेडर्स  ") == "श्री गणेश ट्रेडर्स"
    assert normalize_name("北京商贸有限公司") == "北京商贸有限公司"
    assert normalize_name("شركة النور") == "شركة النور"


def test_normalize_name_mixed_script():
    assert normalize_name("श्री Ganesh Pvt. Ltd.") == "श्री ganesh private limited"
    assert normalize_name("東京 Trading Co.") == "東京 trading company"


def test_normalize_name_missing_and_conversion():
    assert normalize_name(None) == ""
    assert normalize_name("") == ""
    assert normalize_name(" \t\n ") == ""
    assert normalize_name(123) == "123"


# ---------------------------------------------------------------------------
# tokenize_name
# ---------------------------------------------------------------------------

def test_tokenize_name():
    assert tokenize_name("श्री Ganesh Pvt. Ltd.") == [
        "श्री", "ganesh", "private", "limited",
    ]
    assert tokenize_name("北京商贸有限公司") == ["北京商贸有限公司"]
    assert tokenize_name("B A B") == ["b", "a", "b"]
    assert tokenize_name("R&D O'Neil-Smith +") == [
        "r", "&", "d", "o", "'", "neil", "-", "smith", "+",
    ]
    assert tokenize_name("A & B") == ["a", "and", "b"]
    assert tokenize_name(None) == []
    assert tokenize_name("") == []
    assert tokenize_name(" \t ") == []


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------

def test_normalize_name_idempotence():
    examples = [
        None, "", 123, "ABC Pvt. Ltd.", "ABC Co Ltd", "Acme Co",
        "Co Op Trading", "ＡＢＣ Ｐｖｔ． Ｌｔｄ．", "Cafe\u0301",
        "STRAẞE", "श्री Ganesh Pvt. Ltd.", "北京商贸有限公司",
        "شركة النور", "O'Neil\u2013Smith", "R&D", "A & & B", "  A\tB  ",
    ]
    for example in examples:
        normalized = normalize_name(example)
        assert normalize_name(normalized) == normalized, (
            f"Non-idempotent: {example!r} → {normalized!r} → "
            f"{normalize_name(normalized)!r}"
        )
        assert tokenize_name(normalized) == tokenize_name(example), (
            f"Token mismatch after normalization for {example!r}"
        )


# ---------------------------------------------------------------------------
# DataNormalizer integration — verify the adapter works
# ---------------------------------------------------------------------------

class TestDataNormalizerAdapter:
    """Verify DataNormalizer.clean_text() is now powered by normalize_name()."""

    def setup_method(self):
        self.dn = DataNormalizer()

    def test_clean_text_none_returns_empty(self):
        assert self.dn.clean_text(None) == ""

    def test_clean_text_empty_returns_empty(self):
        assert self.dn.clean_text("") == ""

    def test_clean_text_whitespace_only_returns_empty(self):
        assert self.dn.clean_text("   \t\n  ") == ""

    def test_clean_text_abbreviation_expansion(self):
        # normalize_name() expands trailing legal abbreviations
        assert self.dn.clean_text("Acme Corp.") == "acme corporation"
        assert self.dn.clean_text("Raj Pvt Ltd") == "raj private limited"

    def test_clean_text_ampersand(self):
        assert self.dn.clean_text("A & B") == "a and b"
        # Embedded & is preserved
        assert self.dn.clean_text("R&D") == "r&d"

    def test_clean_text_unicode_casefold(self):
        assert self.dn.clean_text("STRAẞE Trading") == "strasse trading"

    def test_clean_text_non_string_input(self):
        # DataNormalizer previously only handled str; now accepts any type
        assert self.dn.clean_text(123) == "123"

    def test_normalize_dataframe_adds_clean_columns(self):
        import pandas as pd
        df = pd.DataFrame([
            {"business_name": "Acme Corp.", "business_address": "123 Main St", "country": "US"},
            {"business_name": "Raj Pvt Ltd", "business_address": "456 Park Ave", "country": "IN"},
        ])
        result = self.dn.normalize_dataframe(df)
        # Clean columns exist
        assert "business_name_clean" in result.columns
        assert "business_address_clean" in result.columns
        assert "text_clean_combined" in result.columns
        # Values are normalized via normalize_name()
        assert result.loc[0, "business_name_clean"] == "acme corporation"
        assert result.loc[1, "business_name_clean"] == "raj private limited"

    def test_normalize_dataframe_empty_returns_copy(self):
        import pandas as pd
        empty = pd.DataFrame()
        result = self.dn.normalize_dataframe(empty)
        assert result.empty

    def test_normalize_dataframe_missing_values_handled(self):
        import pandas as pd
        df = pd.DataFrame([
            {"business_name": None},
            {"business_name": ""},
            {"business_name": "Valid Name"},
        ])
        result = self.dn.normalize_dataframe(df, text_columns=["business_name"])
        assert result.loc[0, "business_name_clean"] == ""
        assert result.loc[1, "business_name_clean"] == ""
        assert result.loc[2, "business_name_clean"] == "valid name"
