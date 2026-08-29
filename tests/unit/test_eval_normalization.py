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


def _case(**kwargs):
    from tests.eval.runner import EvalCase

    kwargs.setdefault("id", "t")
    kwargs.setdefault("question", "q")
    kwargs.setdefault("expect_any_substring", [])
    return EvalCase(**kwargs)


def test_forbidden_substring_is_detected_case_insensitively():
    """The leak was 'the get_product API call'; casing must not let it through."""
    from tests.eval.runner import normalize

    case = _case(expect_no_substring=["get_product"])
    answer = normalize("It came from the Get_Product API call.")
    assert any(normalize(s) in answer for s in case.expect_no_substring)


def test_expect_tools_omitted_means_dont_care():
    """None must differ from []: [] asserts no tool ran, None asserts nothing at all."""
    assert _case().expect_tools is None
    assert _case(expect_tools=[]).expect_tools == []


def test_dataset_meta_cases_forbid_tool_names():
    """Guards the dataset itself: the meta cases are worthless if the negative list is dropped."""
    from tests.eval.runner import load_dataset

    metas = [c for c in load_dataset() if c.id.startswith("meta_")]
    assert len(metas) == 2
    for case in metas:
        assert "get_product" in (case.expect_no_substring or [])


def test_must_pass_ids_all_exist_in_the_dataset():
    """A typo in MUST_PASS_CASE_IDS protects nothing and fails silently: the id simply matches no
    case, so the check passes vacuously. Verified here rather than in the eval suite so it costs
    no LLM quota."""
    from tests.eval.runner import load_dataset
    from tests.eval.test_eval_suite import MUST_PASS_CASE_IDS

    known = {c.id for c in load_dataset()}
    unknown = MUST_PASS_CASE_IDS - known
    assert not unknown, f"MUST_PASS ids match no case in the dataset: {sorted(unknown)}"


def test_must_pass_set_is_not_empty():
    from tests.eval.test_eval_suite import MUST_PASS_CASE_IDS

    assert MUST_PASS_CASE_IDS
