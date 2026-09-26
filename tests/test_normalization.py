from src.normalization import normalize_basic


def test_normalize_basic_lowercase():
    assert normalize_basic("RAJ INVESTMENTS LLP") == "raj investments llp"


def test_normalize_basic_whitespace():
    assert normalize_basic("  Raj   Investments LLP  ") == "raj investments llp"


def test_normalize_basic_empty_string():
    assert normalize_basic("") == ""

def test_normalize_basic_none():
    assert normalize_basic(None) == ""
