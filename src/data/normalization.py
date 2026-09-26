# Authoritative business-name normalization — Maithili's implementation.
# Copied verbatim from project-branch2/src/normalization.py.
# DO NOT rewrite or simplify. All Unicode handling is intentional.
import re
import unicodedata


_ABBREVIATIONS = {
    "corp": "corporation",
    "pvt": "private",
    "ltd": "limited",
    "inc": "incorporated",
    "co": "company",
}
_LEGAL_WORDS = frozenset(_ABBREVIATIONS.values()) | {"llc", "llp"}
_PUNCTUATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-",
})


def normalize_basic(text: object) -> str:
    """Handle None, stringify, lowercase, collapse whitespace, and trim."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(text: object) -> str:
    """Return a conservative Unicode business-name representation.

    Only whitespace-delimited '&' becomes 'and'; embedded forms like R&D
    remain intact. Abbreviations expand only in a trailing legal-name run
    with a preceding name. Bare terminal 'co' remains ambiguous and is kept;
    'co.' or 'co' followed by another legal word can expand. Other punctuation
    is retained, apart from typographic quote/hyphen variants and the optional
    period on an expanded abbreviation. This is not an entity identity key.
    """
    if text is None:
        return ""

    # Normalize again after casefold, which can introduce combining sequences.
    value = unicodedata.normalize("NFKC", str(text))
    value = unicodedata.normalize("NFKC", value.casefold())
    value = value.translate(_PUNCTUATION)
    words = value.split()
    words = ["and" if word == "&" else word for word in words]

    for index in range(len(words) - 1, 0, -1):
        word = words[index]
        abbreviation = word[:-1] if word.endswith(".") else word
        if abbreviation in _ABBREVIATIONS:
            if word == "co" and index == len(words) - 1:
                break
            words[index] = _ABBREVIATIONS[abbreviation]
        elif word not in _LEGAL_WORDS:
            break

    return " ".join(words)


def tokenize_name(text: object) -> list[str]:
    """Tokenize the normalized name in order without losing punctuation.

    Unicode letters, numbers, and combining marks form word tokens. Each
    punctuation/symbol character is a separate token; whitespace is omitted.
    Keeping marks with letters preserves scripts such as Devanagari. No
    language-specific word segmentation is attempted.
    """
    tokens: list[str] = []
    word: list[str] = []
    for char in normalize_name(text):
        if unicodedata.category(char)[0] in "LNM":
            word.append(char)
        else:
            if word:
                tokens.append("".join(word))
                word = []
            if not char.isspace():
                tokens.append(char)
    if word:
        tokens.append("".join(word))
    return tokens
