"""Query relaxation in the DummyJSON search client.

Upstream requires every word of the query to match, so an over-specified query returns nothing for
a product the catalogue plainly has: "gucci bloom perfume" finds 0 results while "gucci bloom"
finds it. Models append the user's category word routinely, so this is not an edge case.
"""

import asyncio

import pytest

from mcp_servers.dummyjson import dummyjson_client as dj


class _Response:
    def __init__(self, products):
        self._products = products

    def raise_for_status(self):
        return None

    def json(self):
        return {"products": self._products}


class _FakeClient:
    """Returns products only for queries listed in `hits`; records what was asked."""

    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    async def get(self, _path, params=None):
        query = (params or {}).get("q")
        self.queries.append(query)
        return _Response(self.hits.get(query, []))


@pytest.fixture
def fake_client(monkeypatch):
    def _install(hits):
        client = _FakeClient(hits)
        monkeypatch.setattr(dj, "_get_client", lambda: client)
        return client

    return _install


def test_relaxed_queries_drops_trailing_words():
    assert dj._relaxed_queries("gucci bloom perfume") == [
        "gucci bloom perfume",
        "gucci bloom",
        "gucci",
    ]


def test_relaxed_queries_never_empties_the_query():
    assert dj._relaxed_queries("macbook") == ["macbook"]


def test_relaxed_queries_is_bounded():
    """A long phrase must not fan out into one upstream request per word."""
    candidates = dj._relaxed_queries("a b c d e f g")
    assert len(candidates) == dj._MAX_QUERY_RELAXATIONS + 1


def test_exact_query_short_circuits(fake_client):
    """A query that matches must not trigger extra upstream requests."""
    client = fake_client({"gucci bloom": [{"title": "Gucci Bloom", "price": 79.99}]})
    results = asyncio.run(dj.search("gucci bloom"))
    assert [r["title"] for r in results] == ["Gucci Bloom"]
    assert client.queries == ["gucci bloom"]


def test_over_specified_query_falls_back(fake_client):
    client = fake_client({"gucci bloom": [{"title": "Gucci Bloom", "price": 79.99}]})
    results = asyncio.run(dj.search("gucci bloom perfume"))
    assert [r["title"] for r in results] == ["Gucci Bloom"]
    assert client.queries == ["gucci bloom perfume", "gucci bloom"]


def test_no_match_at_any_width_returns_empty(fake_client):
    client = fake_client({})
    assert asyncio.run(dj.search("nonexistent gadget xyz123")) == []
    assert len(client.queries) == dj._MAX_QUERY_RELAXATIONS + 1


def test_price_ceiling_is_not_relaxed(fake_client):
    """A matched query filtered out by max_price stays empty: the price excluded it, not the
    wording, so widening the search would answer a different question."""
    client = fake_client({"gucci bloom perfume": [{"title": "Gucci Bloom", "price": 79.99}]})
    assert asyncio.run(dj.search("gucci bloom perfume", max_price=10.0)) == []
    assert client.queries == ["gucci bloom perfume"]
