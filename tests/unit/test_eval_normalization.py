"""The eval matcher compares meaning, not typography.

Every case here failed the golden set while holding a correct answer: the model wrote the right
value in a different notation than the expectation was literally spelled.
"""

from tests.eval.runner import normalize


def _matches(expected: str, answer: str) -> bool:
    return normalize(expected) in normalize(answer)


def test_thousands_separator_matches():
    assert _matches("1999.99", "The MacBook Pro 14-inch is listed for **$1,999.99**.")


def test_curly_apostrophe_matches():
    assert _matches("don't have", "I’m sorry, but I don’t have any information on that.")


def test_narrow_no_break_space_matches():
    """gpt-oss writes percentages as '4.69 %' — NFKC folds the exotic space."""
    assert _matches("4.69 %", "a 4.69 % discount applies")


def test_non_breaking_hyphen_matches():
    assert _matches("14-inch", "the MacBook Pro 14‑inch Space Grey")


def test_case_is_ignored():
    assert _matches("not found", "Product NOT FOUND in the catalogue")


def test_genuinely_different_value_still_fails():
    """The point is folding notation, not weakening the assertion."""
    assert not _matches("1999.99", "The MacBook Pro is listed for $2,499.99.")


def test_genuinely_absent_phrase_still_fails():
    assert not _matches("don't have", "Here is the product you asked about.")


def test_comma_that_is_not_a_separator_is_kept():
    """Only digit-grouping commas are removed, so list punctuation cannot silently merge values."""
    assert not _matches("1999", "in stock: 1,99 and 9 units")
