from decimal import Decimal

import pytest

from src.address import address_fingerprint, normalize_address


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
    "12A-Main–St—Apt‑4B",
    "12A\\Main;St: Apt_(4B)",
    "12A | Main | St | Apt #4B",
    "１２Ａ　Ｍａｉｎ Ｓｔ Ａｐｔ ４Ｂ",
    "12A\u00a0Main\u200bSt Apt 4B",
])
def test_formatting_variants_share_canonical_address(address):
    canonical = "12a main st apt 4b"
    assert normalize_address(address) == normalize_address(canonical) == canonical
    assert address_fingerprint(address) == canonical


@pytest.mark.parametrize("address,expected", [
    ("7 Cafe\u0301 Straße", "7 café strasse"),
    ("7 Café STRASSE", "7 café strasse"),
    ("१२ श्री गणेश मार्ग", "१२ श्री गणेश मार्ग"),
    ("北京路１２号", "北京路12号"),
    ("12 شارع النور", "12 شارع النور"),
    ("Unit 007B, Block A2, 00120", "unit 007b block a2 00120"),
    ("\ufeff12 Ma\u00adin\u200e Road", "12 main road"),
    ("12\x00Main\x01Road", "12 main road"),
    ("12 A+B & C", "12 a+b c"),
    ("12 می\u200cرود", "12 می\u200cرود"),
    ("12 क्\u200dष", "12 क्\u200dष"),
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
