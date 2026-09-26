from src.normalization import normalize_basic, normalize_name, tokenize_name


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


def test_normalize_name_case():
    assert normalize_name("STRAẞE Trading") == "strasse trading"


def test_normalize_name_whitespace():
    assert normalize_name(" \tRaj\n Investments\u00a0 LLP  ") == "raj investments llp"


def test_normalize_name_punctuation():
    assert normalize_name("O’Neil–Smith “Trading”") == 'o\'neil-smith "trading"'
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
        assert normalize_name(abbreviated) == normalize_name(expanded)


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


def test_tokenize_name():
    assert tokenize_name("श्री Ganesh Pvt. Ltd.") == [
        "श्री", "ganesh", "private", "limited",
    ]
    assert tokenize_name("北京商贸有限公司") == ["北京商贸有限公司"]
    assert tokenize_name("B A B") == ["b", "a", "b"]
    assert tokenize_name("R&D O’Neil-Smith +") == [
        "r", "&", "d", "o", "'", "neil", "-", "smith", "+",
    ]
    assert tokenize_name("A & B") == ["a", "and", "b"]
    assert tokenize_name(None) == []
    assert tokenize_name("") == []
    assert tokenize_name(" \t ") == []


def test_normalize_name_idempotence():
    examples = [
        None, "", 123, "ABC Pvt. Ltd.", "ABC Co Ltd", "Acme Co",
        "Co Op Trading", "ＡＢＣ Ｐｖｔ． Ｌｔｄ．", "Cafe\u0301",
        "STRAẞE", "श्री Ganesh Pvt. Ltd.", "北京商贸有限公司",
        "شركة النور", "O’Neil–Smith", "R&D", "A & & B", "  A\tB  ",
    ]
    for example in examples:
        normalized = normalize_name(example)
        assert normalize_name(normalized) == normalized
        assert tokenize_name(normalized) == tokenize_name(example)
