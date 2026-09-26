# Authoritative address normalization — Maithili's implementation.
# Copied verbatim from project-branch2/src/address.py.
# DO NOT rewrite, simplify, or remove Unicode handling.
"""Deterministic address representations for later exact blocking."""
from decimal import Decimal
import math
import unicodedata


def normalize_address(address: object) -> str:
    """Normalize a scalar address with NFKC and case folding, as for names.

    None and numeric NaN are missing; other inputs use str(), with conversion
    errors propagated. Literal strings such as 'NA' and 'null' remain content.
    Punctuation, whitespace, controls and vertical bars become spaces. Invisible
    formatting marks are removed, except zero-width spaces (word boundaries)
    and script joiners (preserved). Letters, accents, numbers, symbols and word
    order remain; no abbreviation expansion, translation or correction occurs.

    Separator distinctions are intentionally lost (e.g. 12/3 and 12-3). Equality
    is a blocking heuristic, not proof of address identity.
    """
    if address is None:
        return ""
    if isinstance(address, float) and math.isnan(address):
        return ""
    if isinstance(address, Decimal) and address.is_nan():
        return ""

    value = unicodedata.normalize("NFKC", str(address))
    value = unicodedata.normalize("NFKC", value.casefold())
    characters: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if char == "\u200b" or char == "|" or char.isspace() or category[0] == "P":
            characters.append(" ")
        elif category == "Cf":
            if char in "\u200c\u200d":
                characters.append(char)
        elif category[0] == "C":
            characters.append(" ")
        else:
            characters.append(char)
    return " ".join("".join(characters).split())


def address_fingerprint(address: object) -> str:
    """Return the normalized address itself as an order-preserving exact key.

    No hashing, token sorting or similarity is applied. Empty keys represent
    missing/empty addresses and should be excluded from downstream blocking.
    """
    return normalize_address(address)
