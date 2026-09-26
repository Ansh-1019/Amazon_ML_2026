"""
Integration tests for Maithili's address normalization, imported from src.data.address.

These tests are ported verbatim from project-branch2/tests/test_address.py.
The ONLY change is the import path:
    was:  from src.address import ...      (branch2 package)
    now:  from src.data.address import ...  (main project package)

No test assertions were modified. If a test fails here but passed in branch2,
the integration introduced a bug — fix the import/wiring, never the algorithm.
"""
from decimal import Decimal

import pytest
import pandas as pd

from src.data.address import address_fingerprint, normalize_address
from src.data import DataNormalizer


# ---------------------------------------------------------------------------
# Verbatim ports of Maithili's test_address.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("address", [None, "", " \t\r\n ", float("nan"),
                                      Decimal("NaN"), Decimal("sNaN")])
def test_missing_and_empty(address):
    assert normalize_address(address) == ""
    assert address_fingerprint(address) == ""


@pytest.mark.parametrize("address", [
    "12A Main St Apt 4B",
    "  12A   Main\tSt\nApt 4B  ",
    "12a MAIN st APT 4b",
    "12A, Main St., Apt. 4B",
    "12A/Main/St/Apt/4B",
    "12A-Main\u2013St\u2014Apt\u2011 4B",
    "12A\\Main;St: Apt_(4B)",
    "12A | Main | St | Apt #4B",
    "\uff11\uff12\uff21\u3000\uff2d\uff21\uff29\uff2e \uff33\uff34 \uff21\uff30\uff34 \uff14\uff22",
    "12A\u00a0Main\u200bSt Apt 4B",
])
def test_formatting_variants_share_canonical_address(address):
    canonical = "12a main st apt 4b"
    assert normalize_address(address) == normalize_address(canonical) == canonical
    assert address_fingerprint(address) == canonical


@pytest.mark.parametrize("address,expected", [
    ("7 Cafe\u0301 Stra\u00dfe", "7 café strasse"),
    ("7 Café STRASSE", "7 café strasse"),
    ("\u0967\u0968 \u0936\u094d\u0930\u0940 \u0917\u0923\u0947\u0936 \u092e\u093e\u0930\u094d\u0917",
     "\u0967\u0968 \u0936\u094d\u0930\u0940 \u0917\u0923\u0947\u0936 \u092e\u093e\u0930\u094d\u0917"),
    ("\u5317\u4eac\u8def\uff11\uff12\u53f7", "\u5317\u4eac\u8def12\u53f7"),
    ("12 \u0634\u0627\u0631\u0639 \u0627\u0644\u0646\u0648\u0631", "12 \u0634\u0627\u0631\u0639 \u0627\u0644\u0646\u0648\u0631"),
    ("Unit 007B, Block A2, 00120", "unit 007b block a2 00120"),
    ("\ufeff12 Ma\u00adin\u200e Road", "12 main road"),
    ("12\x00Main\x01Road", "12 main road"),
    ("12 A+B & C", "12 a+b c"),
    ("12 \u0645\u06cc\u200c\u0631\u0648\u062f", "12 \u0645\u06cc\u200c\u0631\u0648\u062f"),
    ("12 \u0915\u094d\u200d\u0937", "12 \u0915\u094d\u200d\u0937"),
    ("... , / -", ""),
    (0, "0"),
    (123, "123"),
    (12.5, "12 5"),
    ("NA", "na"),
    ("None", "none"),
    ("null", "null"),
    ("NaN", "nan"),
])
def test_preserved_content_and_scalar_conversion(address, expected):
    assert normalize_address(address) == expected
    assert address_fingerprint(address) == expected
    assert normalize_address(expected) == expected
    for _ in range(3):
        assert address_fingerprint(address) == expected


@pytest.mark.parametrize("left,right", [
    ("12 Main St", "13 Main St"),
    ("12 Main St Apt 4", "12 Main St Apt 5"),
    ("12A Main St", "12B Main St"),
    ("12A Main St", "12 A Main St"),
    ("12 Main St", "12 Oak St"),
    ("12 North Main", "12 South Main"),
    ("12 Main 00120", "12 Main 120"),
    ("12 Café Rd", "12 Cafe Rd"),
    ("12 Main St", "12 St Main"),
    ("12 Main St", "12 Main Street"),
    ("12 Main Main", "12 Main"),
    ("12/3 Main", "123 Main"),
])
def test_distinct_address_content_is_not_collapsed(left, right):
    assert normalize_address(left) != normalize_address(right)
    assert address_fingerprint(left) != address_fingerprint(right)


def test_custom_string_conversion():
    class Address:
        def __str__(self):
            return " 12 Main ST. "

    assert normalize_address(Address()) == "12 main st"


def test_failed_string_conversion_is_not_silently_missing():
    class InvalidAddress:
        def __str__(self):
            raise ValueError("invalid address")

    with pytest.raises(ValueError, match="invalid address"):
        normalize_address(InvalidAddress())


# ---------------------------------------------------------------------------
# DataNormalizer adapter — verify address integration wiring
# ---------------------------------------------------------------------------

class TestDataNormalizerAddressAdapter:
    """Verify DataNormalizer produces business_address_normalized correctly."""

    def setup_method(self):
        self.dn = DataNormalizer()

    def test_clean_address_none_returns_empty(self):
        assert self.dn.clean_address(None) == ""

    def test_clean_address_empty_returns_empty(self):
        assert self.dn.clean_address("") == ""

    def test_clean_address_float_nan_returns_empty(self):
        assert self.dn.clean_address(float("nan")) == ""

    def test_clean_address_decimal_nan_returns_empty(self):
        assert self.dn.clean_address(Decimal("NaN")) == ""

    def test_clean_address_basic(self):
        assert self.dn.clean_address("12A Main St., Apt. 4B") == "12a main st apt 4b"

    def test_clean_address_unicode(self):
        assert self.dn.clean_address("7 Café STRASSE") == "7 café strasse"

    def test_clean_address_does_not_expand_abbreviations(self):
        # normalize_address() intentionally does NOT expand legal abbreviations
        # (unlike normalize_name). "St" stays "st", not "street".
        result = self.dn.clean_address("303 Cedar St, Dallas, TX")
        assert "st" in result
        assert "street" not in result

    def test_normalize_dataframe_creates_normalized_column(self):
        df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Acme Corp",
             "business_address": "12A Main St., Apt. 4B", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Raj Pvt Ltd",
             "business_address": "7 Café STRASSE", "country": "DE"},
        ])
        result = self.dn.normalize_dataframe(df)

        # Normalized address column exists
        assert "business_address_normalized" in result.columns
        assert result.loc[0, "business_address_normalized"] == "12a main st apt 4b"
        assert result.loc[1, "business_address_normalized"] == "7 café strasse"

    def test_normalize_dataframe_preserves_original_address(self):
        df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Acme",
             "business_address": "12A Main St., Apt. 4B", "country": "US"},
        ])
        result = self.dn.normalize_dataframe(df)
        # Original column is UNTOUCHED
        assert result.loc[0, "business_address"] == "12A Main St., Apt. 4B"

    def test_normalize_dataframe_clean_column_still_exists(self):
        df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Acme",
             "business_address": "12 Main St", "country": "US"},
        ])
        result = self.dn.normalize_dataframe(df)
        # The existing {col}_clean column is still produced (backward compatibility)
        assert "business_address_clean" in result.columns
        assert "business_address_normalized" in result.columns

    def test_normalize_dataframe_null_address_handled(self):
        df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Acme",
             "business_address": None, "country": "US"},
            {"entity_id": "S1-2", "business_name": "Beta",
             "business_address": "", "country": "US"},
        ])
        result = self.dn.normalize_dataframe(df)
        assert result.loc[0, "business_address_normalized"] == ""
        assert result.loc[1, "business_address_normalized"] == ""

    def test_normalize_dataframe_no_address_column(self):
        """DataFrames without a business_address column should work fine."""
        df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Acme Corp", "country": "US"},
        ])
        result = self.dn.normalize_dataframe(df)
        assert "business_name_clean" in result.columns
        assert "business_address_normalized" not in result.columns

    def test_normalize_dataframe_all_sources_supported(self):
        """S1, S2, S3 rows all go through the same normalizer — no source-specific logic."""
        rows = [
            {"entity_id": "S1-1", "business_name": "Acme Corp",
             "business_address": "100 Main St", "country": "US", "source": "source1"},
            {"entity_id": "S2-1", "business_name": "Acme Corp.",
             "business_address": "100 Main St.", "country": "US", "source": "source2"},
            {"entity_id": "S3-1", "business_name": "ACME CORP",
             "business_address": "100 MAIN ST", "country": "US", "source": "source3"},
        ]
        df = pd.DataFrame(rows)
        result = self.dn.normalize_dataframe(df)
        # All three sources produce the same normalized address
        assert result.loc[0, "business_address_normalized"] == "100 main st"
        assert result.loc[1, "business_address_normalized"] == "100 main st"
        assert result.loc[2, "business_address_normalized"] == "100 main st"

    def test_normalize_dataframe_unicode_countries_not_hardcoded(self):
        """Country column is not hardcoded — any country passes through."""
        df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "A",
             "business_address": "7 Rue de la Paix", "country": "France"},
            {"entity_id": "S1-2", "business_name": "B",
             "business_address": "\u5317\u4eac\u8def12\u53f7", "country": "\u4e2d\u56fd"},
            {"entity_id": "S1-3", "business_name": "C",
             "business_address": "12 \u0634\u0627\u0631\u0639 \u0627\u0644\u0646\u0648\u0631",
             "country": "\u0645\u0635\u0631"},
        ])
        result = self.dn.normalize_dataframe(df)
        assert result.loc[0, "business_address_normalized"] == "7 rue de la paix"
        # Non-Latin scripts preserved
        assert "\u5317\u4eac\u8def" in result.loc[1, "business_address_normalized"]
        assert "\u0634\u0627\u0631\u0639" in result.loc[2, "business_address_normalized"]

    def test_text_clean_combined_does_not_include_normalized_column(self):
        """text_clean_combined is built from _clean columns only, not _normalized."""
        df = pd.DataFrame([
            {"entity_id": "S1-1", "business_name": "Acme Corp",
             "business_address": "12 Main St", "country": "US"},
        ])
        result = self.dn.normalize_dataframe(df)
        combined = result.loc[0, "text_clean_combined"]
        # Should contain name and address clean versions, not duplicated normalized form
        assert combined  # non-empty
        # normalized address value ("12 main st") should appear exactly once at most
        assert combined.count("12 main st") <= 1
