"""MCP server exposing the product catalogue.

Each tool reads through a Postgres cache and falls back to the DummyJSON API on a miss or a stale
row. Cached rows are keyed by canonical catalogue title; see `app.db.repository.find_product`.

The complete upstream payload is stored, and `get_product` returns everything answerable from it,
so questions about stock, warranty, shipping, returns, brand or dimensions are answerable without
a schema change. Tool *schemas* stay small — they are re-sent on every model call, unlike results,
which cost tokens only when a tool actually runs.
"""

import json

import httpx
from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.db.base import SessionLocal
from app.db.repository import find_product, is_stale, upsert_product, upsert_review
from mcp_servers.dummyjson import dummyjson_client as dj

mcp = FastMCP("dummyjson")


def _store(raw: dict) -> None:
    """Cache one upstream product record.

    Keyed on the canonical title, never a caller's query, so one product cannot end up with a row
    per phrasing. The whole payload is kept alongside the denormalised lookup columns.
    """
    title = raw["title"]
    reviews = raw.get("reviews", [])
    with SessionLocal() as session:
        upsert_product(
            session,
            title,
            price=raw["price"],
            rating=raw["rating"],
            description=raw["description"],
            data=json.dumps(raw),
        )
        upsert_review(
            session,
            title,
            reviews_count=len(reviews),
            rating=raw["rating"],
            comments=json.dumps(
                [{"rating": r["rating"], "comment": r["comment"]} for r in reviews]
            ),
        )
        session.commit()


async def _fetch_and_cache(name: str) -> dict | None:
    """Fetch the best upstream match for `name`, cache it, and return the raw record.

    Returns None when there is no match or the request fails, leaving callers to fall back to
    cached data.
    """
    try:
        results = await dj.search(name)
    except httpx.HTTPError:
        return None
    if not results:
        return None

    raw = results[0]
    _store(raw)
    return raw


def _cached_product(name: str) -> tuple[dict | None, bool]:
    """Return the cached upstream record for `name` and whether it is still fresh.

    A row without `data` predates that column, so it cannot answer most questions; treating it as
    stale refills it on first use rather than serving a degraded record until its TTL expires.
    """
    with SessionLocal() as session:
        product = find_product(session, name)
        if product is None or not product.data:
            return None, False
        raw = json.loads(product.data)
        return raw, not is_stale(product, settings.product_cache_ttl_seconds)


@mcp.tool()
async def get_product(name: str) -> str:
    """Get everything known about one product: price, discount, rating, stock, availability,
    brand, category, warranty, shipping, return policy, dimensions, weight, tags, description,
    and what reviewers said."""
    cached, fresh = _cached_product(name)
    if fresh:
        return str(dj.describe_product(cached))

    raw = await _fetch_and_cache(name)
    if raw is not None:
        return str(dj.describe_product(raw))
    # Upstream unreachable: stale data beats no data.
    if cached is not None:
        return str(dj.describe_product(cached))
    return f"Product not found: {name}"


@mcp.tool()
async def search_products(query: str | None = None, max_price: float | str | None = None) -> str:
    """Browse the catalogue. `query` is a category or keyword; omit it to list all categories.
    Returns a summary per match — call get_product by name for full detail on one of them."""
    # Accepts str as well as float because models routinely emit numeric arguments as strings,
    # which the provider rejects against a float-only schema. Retrying does not help: at
    # temperature 0 the model reproduces the identical invalid call every attempt.
    if isinstance(max_price, str):
        try:
            max_price = float(max_price)
        except ValueError:
            max_price = None

    try:
        if not query:
            return str({"categories": await dj.list_categories()})

        # Try the term as a category first, then as a free-text keyword. Doing this here rather
        # than exposing both as parameters removes a choice models were getting wrong. The slug is
        # resolved from the user's phrasing, since "men watches" and "mens-watches" are the same
        # request and the raw form returns nothing.
        slug = await dj.resolve_category(query)
        results = await dj.browse_category(slug, max_price=max_price) if slug else []
        if not results:
            results = await dj.search(query, max_price=max_price)
    except httpx.HTTPError:
        return "Product search is temporarily unavailable."

    if not results:
        return "No matching products found."

    for raw in results:
        _store(raw)

    return str([dj.summarize_product(raw) for raw in results])


if __name__ == "__main__":
    mcp.run()
