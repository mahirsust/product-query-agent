"""HTTP client for the DummyJSON product API, plus the shaping applied to its responses.

No credentials: the API is fully public.
"""

import re
from urllib.parse import quote

import httpx

BASE_URL = "https://dummyjson.com"
_TIMEOUT = 10.0

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Lazily-created, reused client for connection pooling across calls (the MCP server process
    is long-lived, so a fresh client per request would just discard pooled connections)."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=BASE_URL, timeout=_TIMEOUT)
    return _client


def _filter_max_price(results: list[dict], max_price: float | None) -> list[dict]:
    """Apply the price ceiling client-side; the API has no price-range parameter."""
    if max_price is None:
        return results
    return [r for r in results if r["price"] <= max_price]


async def list_categories() -> list[str]:
    """Category slugs only (`/products/category-list`), not the richer `/products/categories`
    objects — the agent just needs browsable names to offer the user, and slugs are exactly what
    `browse_category()` takes back as input."""
    resp = await _get_client().get("/products/category-list")
    resp.raise_for_status()
    return resp.json()


# Filler words that carry no category meaning; without this "watches for men" fails to match
# "mens-watches", because every word of the query is required to match something.
_STOPWORDS = frozenset(
    {"a", "an", "the", "for", "of", "in", "on", "with", "and", "to", "me", "my", "some", "any"}
)


def _tokens(text: str, drop_stopwords: bool = False) -> list[str]:
    """Split text into comparable words, ignoring punctuation and separators."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS] or tokens
    return tokens


def _matches(query_token: str, category_token: str) -> bool:
    """Whether two words refer to the same thing, tolerating plurals and joined spellings."""
    shorter, longer = sorted((query_token, category_token), key=len)
    return len(shorter) >= 3 and longer.startswith(shorter)


async def resolve_category(query: str) -> str | None:
    """Map a user's phrasing onto a real category slug, or None if nothing matches well.

    Users write "men watches" or "skincare"; the catalogue uses "mens-watches" and "skin-care".
    Passing the raw phrasing through returns zero results, which reads to the model as "nothing
    found" and sends it retrying. Matching is prefix-based per word so plural and hyphenation
    differences resolve, and requires every word of the query to match something.
    """
    query_tokens = _tokens(query, drop_stopwords=True)
    if not query_tokens:
        return None

    best, best_score = None, 0.0
    for category in await list_categories():
        category_tokens = _tokens(category)
        matched = sum(any(_matches(q, c) for c in category_tokens) for q in query_tokens)
        score = matched / len(query_tokens)
        # Prefer the most specific match when scores tie: "watches" alone should not beat
        # "mens-watches" for the query "men watches".
        if score > best_score or (
            score == best_score and best and len(category_tokens) > len(_tokens(best))
        ):
            best, best_score = category, score

    return best if best_score == 1.0 else None


# Upstream search requires *every* word to match, so one surplus qualifier yields nothing:
# "gucci bloom perfume" returns 0 results while "gucci bloom" returns the product. Models routinely
# append the user's category word to a product name, so the strictness surfaces as "no such
# product" for items the catalogue plainly has.
_MAX_QUERY_RELAXATIONS = 2


def _relaxed_queries(query: str) -> list[str]:
    """The query, then progressively shorter prefixes of it.

    Trailing words are dropped because the surplus term is nearly always a qualifier appended to a
    distinctive name ("... perfume", "... laptop"), not part of it. Bounded so a long phrase cannot
    fan out into many upstream requests, and never reduced to nothing.
    """
    words = query.split()
    droppable = min(_MAX_QUERY_RELAXATIONS, len(words) - 1)
    return [query] + [" ".join(words[:-drop]) for drop in range(1, droppable + 1)]


async def search(query: str, max_price: float | None = None) -> list[dict]:
    """Free-text product search, relaxing an over-specified query rather than giving up on it.

    Only the query is relaxed; matching itself is still upstream's job, so this does not
    reimplement fuzzy search locally. A `max_price` that filters every match is left empty rather
    than relaxed further — the query did match, the price ceiling is what excluded it.
    """
    for candidate in _relaxed_queries(query):
        resp = await _get_client().get("/products/search", params={"q": candidate})
        resp.raise_for_status()
        results = resp.json().get("products", [])
        if results:
            return _filter_max_price(results, max_price)
    return []


async def browse_category(category: str, max_price: float | None = None) -> list[dict]:
    """List a category's products. `category` must be a real slug — see `resolve_category`."""
    # category is tool-call input (ultimately LLM/user-controlled) — percent-encode it as a single
    # path segment so it can't inject extra path components into the request against dummyjson.com.
    resp = await _get_client().get(f"/products/category/{quote(category, safe='')}")
    resp.raise_for_status()
    results = resp.json().get("products", [])
    return _filter_max_price(results, max_price)


# Enough to convey the gist of opinion without bloating tool results.
_MAX_COMMENTS = 5

# Fields carrying no answerable information. Image URLs and QR/barcode metadata cannot inform any
# product question but cost ~220 tokens per result, so they are stored yet withheld from the model.
_UNANSWERABLE_FIELDS = frozenset({"images", "meta", "sku", "id"})


def describe_product(data: dict) -> dict:
    """Render a stored product into the view the model sees.

    Everything that could answer a question is kept — brand, stock, warranty, shipping, returns,
    discount, dimensions, tags — because withholding a field silently makes questions about it
    unanswerable, which is far worse than the tokens it costs. Reviewer names and emails are
    dropped as personal data the agent has no use for.
    """
    described = {k: v for k, v in data.items() if k not in _UNANSWERABLE_FIELDS}
    reviews = data.get("reviews", [])
    described["reviews"] = [
        {"rating": r["rating"], "comment": r["comment"]}
        for r in reviews[:_MAX_COMMENTS]
        if r.get("comment")
    ]
    described["reviewCount"] = len(reviews)
    return described


def summarize_product(data: dict) -> dict:
    """Render one entry of a search result list.

    Deliberately narrower than `describe_product`: a browse can return a dozen products, so full
    records would dominate the context. Carries enough to compare options and to name a product
    for a follow-up lookup.
    """
    return {
        "name": data["title"],
        "brand": data.get("brand"),
        "price": data["price"],
        "rating": data["rating"],
        "availability": data.get("availabilityStatus"),
    }
