"""Tests for mapping a user's phrasing onto a real catalogue category slug.

Users write "men watches"; the catalogue uses "mens-watches". Passing the raw phrasing through
returns zero results, which reads to the model as "nothing found" and sends it retrying until it
exhausts the step limit — the failure a user actually reported.
"""

import asyncio

import pytest

from mcp_servers.dummyjson import dummyjson_client as dj

CATEGORIES = [
    "beauty",
    "fragrances",
    "furniture",
    "laptops",
    "mens-shirts",
    "mens-shoes",
    "mens-watches",
    "skin-care",
    "smartphones",
    "womens-watches",
]


@pytest.fixture(autouse=True)
def stub_categories(monkeypatch):
    async def fake_list_categories():
        return CATEGORIES

    monkeypatch.setattr(dj, "list_categories", fake_list_categories)


def _resolve(query: str) -> str | None:
    return asyncio.run(dj.resolve_category(query))


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("mens-watches", "mens-watches"),  # exact
        ("men watches", "mens-watches"),  # plural + separator
        ("mens watches", "mens-watches"),
        ("watches for men", "mens-watches"),  # stopword
        ("skincare", "skin-care"),  # joined spelling
        ("skin care", "skin-care"),
        ("laptops", "laptops"),
        ("laptop", "laptops"),  # singular
        ("smartphone", "smartphones"),
    ],
)
def test_phrasings_resolve_to_the_right_slug(query, expected):
    assert _resolve(query) == expected


def test_prefers_the_more_specific_category():
    """ "men watches" must not resolve to womens-watches just because "watches" matches."""
    assert _resolve("men watches") == "mens-watches"
    assert _resolve("women watches") == "womens-watches"


def test_unrelated_query_resolves_to_nothing():
    """Returning None lets the caller fall back to keyword search rather than browsing a wrong
    category and confidently reporting the wrong products."""
    assert _resolve("nonsense zzz") is None
    assert _resolve("macbook pro") is None


def test_empty_query_resolves_to_nothing():
    assert _resolve("") is None
    assert _resolve("   ") is None


def test_partial_match_is_rejected():
    """Every meaningful word must match; "red shoes" is not the mens-shoes category."""
    assert _resolve("red laptops bicycle") is None
