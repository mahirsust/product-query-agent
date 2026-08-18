"""Tests for what the model is shown about a product.

Withholding a field silently makes every question about it unanswerable, which is the failure
these guard against: the agent previously returned only price, rating and a description, so
"is it in stock?" or "what's the warranty?" could not be answered at all.
"""

from mcp_servers.dummyjson import dummyjson_client as dj

RAW = {
    "id": 78,
    "title": "Apple MacBook Pro 14 Inch Space Grey",
    "description": "A powerful and sleek laptop.",
    "category": "laptops",
    "price": 1999.99,
    "discountPercentage": 4.69,
    "rating": 3.65,
    "stock": 24,
    "tags": ["laptops", "apple"],
    "brand": "Apple",
    "sku": "LAP-APP-APP-078",
    "weight": 9,
    "dimensions": {"width": 20.03, "height": 9.54, "depth": 14.82},
    "warrantyInformation": "3 year warranty",
    "shippingInformation": "Ships in 2 weeks",
    "availabilityStatus": "In Stock",
    "returnPolicy": "90 days return policy",
    "minimumOrderQuantity": 1,
    "images": ["https://example.com/1.webp", "https://example.com/2.webp"],
    "thumbnail": "https://example.com/thumb.webp",
    "meta": {"barcode": "5275211560367", "qrCode": "https://example.com/qr.png"},
    "reviews": [
        {
            "rating": 5,
            "comment": "Very happy with my purchase!",
            "reviewerName": "Hazel Evans",
            "reviewerEmail": "hazel.evans@x.dummyjson.com",
        },
        {
            "rating": 4,
            "comment": "Very satisfied!",
            "reviewerName": "Aubrey Garcia",
            "reviewerEmail": "aubrey.garcia@x.dummyjson.com",
        },
    ],
}


def test_every_answerable_field_reaches_the_model():
    described = dj.describe_product(RAW)
    for field in (
        "title",
        "brand",
        "category",
        "price",
        "discountPercentage",
        "rating",
        "stock",
        "availabilityStatus",
        "warrantyInformation",
        "shippingInformation",
        "returnPolicy",
        "minimumOrderQuantity",
        "dimensions",
        "weight",
        "tags",
        "description",
    ):
        assert field in described, f"{field} would be unanswerable"


def test_review_comments_are_included():
    described = dj.describe_product(RAW)
    assert [r["comment"] for r in described["reviews"]] == [
        "Very happy with my purchase!",
        "Very satisfied!",
    ]
    assert described["reviewCount"] == 2


def test_reviewer_personal_data_is_dropped():
    """Names and emails answer nothing and would land in the database and in model-bound traffic."""
    described = dj.describe_product(RAW)
    rendered = str(described)
    assert "Hazel Evans" not in rendered
    assert "hazel.evans@x.dummyjson.com" not in rendered


def test_unanswerable_fields_are_withheld():
    """Image URLs and barcode metadata cannot inform any question but are expensive to carry."""
    described = dj.describe_product(RAW)
    for field in ("images", "meta", "sku", "id"):
        assert field not in described


def test_comment_count_is_capped_but_total_is_reported():
    raw = {**RAW, "reviews": [{"rating": 5, "comment": f"c{i}"} for i in range(20)]}
    described = dj.describe_product(raw)
    assert len(described["reviews"]) == dj._MAX_COMMENTS
    assert described["reviewCount"] == 20


def test_search_summary_stays_compact():
    """A browse can return a dozen products; full records would dominate the context."""
    summary = dj.summarize_product(RAW)
    assert set(summary) == {"name", "brand", "price", "rating", "availability"}
    assert summary["name"] == RAW["title"]


def test_summary_tolerates_missing_optional_fields():
    minimal = {"title": "Thing", "price": 1.0, "rating": 2.0}
    summary = dj.summarize_product(minimal)
    assert summary["brand"] is None and summary["availability"] is None
