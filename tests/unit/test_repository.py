from datetime import UTC, datetime, timedelta

import pytest

from app.db.repository import (
    create_user,
    find_product,
    find_review,
    get_product,
    get_review,
    get_user_by_id,
    get_user_by_username,
    is_stale,
    upsert_product,
    upsert_review,
)


def test_get_product_missing_returns_none(db_session):
    assert get_product(db_session, "nonexistent") is None


def test_upsert_product_is_case_insensitive_and_creates(db_session):
    upsert_product(db_session, "MacBook Pro", price=1999.99, rating=3.65, description="laptop")
    product = get_product(db_session, "macbook pro")
    assert product is not None
    assert product.price == 1999.99


def test_upsert_product_updates_existing(db_session):
    upsert_product(db_session, "MacBook Pro", price=1999.99, rating=3.65, description="laptop")
    upsert_product(db_session, "MacBook Pro", price=1899.99, rating=3.7, description="laptop v2")
    updated = get_product(db_session, "MacBook Pro")
    assert updated.price == 1899.99
    assert updated.rating == 3.7
    assert updated.description == "laptop v2"


def test_upsert_review_requires_existing_product(db_session):
    with pytest.raises(ValueError):
        upsert_review(db_session, "nonexistent", reviews_count=3, rating=4.5)


def test_upsert_review_roundtrip(db_session):
    upsert_product(db_session, "Gucci Bloom", price=79.99, rating=2.74, description="perfume")
    upsert_review(db_session, "Gucci Bloom", reviews_count=3, rating=2.74)
    review = get_review(db_session, "gucci bloom")
    assert review is not None
    assert review.reviews_count == 3
    assert review.rating == 2.74


def test_upsert_review_updates_existing(db_session):
    upsert_product(db_session, "Gucci Bloom", price=79.99, rating=2.74, description="perfume")
    upsert_review(db_session, "Gucci Bloom", reviews_count=3, rating=2.74)
    upsert_review(db_session, "Gucci Bloom", reviews_count=5, rating=3.0)
    review = get_review(db_session, "Gucci Bloom")
    assert review.reviews_count == 5
    assert review.rating == 3.0


def test_get_review_without_product_returns_none(db_session):
    assert get_review(db_session, "nonexistent") is None


def test_create_and_get_user_roundtrip(db_session):
    user = create_user(db_session, "alice", "hashed_pw_value")
    db_session.commit()

    fetched = get_user_by_username(db_session, "alice")
    assert fetched is not None
    assert fetched.id == user.id
    assert fetched.hashed_password == "hashed_pw_value"

    by_id = get_user_by_id(db_session, user.id)
    assert by_id.username == "alice"


def test_get_user_by_username_missing_returns_none(db_session):
    assert get_user_by_username(db_session, "ghost") is None


def test_get_user_by_id_missing_returns_none(db_session):
    assert get_user_by_id(db_session, 999_999) is None


# --- resolving user-typed names against canonically-titled cache rows ---


def _seed_macbook(db_session):
    upsert_product(
        db_session,
        "Apple MacBook Pro 14 Inch Space Grey",
        price=1999.99,
        rating=3.65,
        description="laptop",
    )
    upsert_review(db_session, "Apple MacBook Pro 14 Inch Space Grey", reviews_count=3, rating=3.65)
    db_session.commit()


def test_find_product_matches_a_fragment_of_the_canonical_title(db_session):
    """Rows are cached under the catalogue title, but users type fragments — without the
    substring fallback every "macbook" query would miss and refetch."""
    _seed_macbook(db_session)
    found = find_product(db_session, "macbook")
    assert found is not None
    assert found.price == 1999.99


def test_find_product_prefers_an_exact_match(db_session):
    upsert_product(db_session, "Laptop Stand", price=25.0, rating=4.0, description="accessory")
    upsert_product(db_session, "Stand", price=10.0, rating=4.0, description="generic")
    db_session.commit()
    assert find_product(db_session, "Stand").price == 10.0


def test_find_product_is_case_insensitive(db_session):
    _seed_macbook(db_session)
    assert find_product(db_session, "MACBOOK") is not None


def test_find_product_missing_returns_none(db_session):
    _seed_macbook(db_session)
    assert find_product(db_session, "gucci bloom") is None


def test_find_product_treats_wildcards_as_literal_text(db_session):
    """A user typing '%' must not turn the substring fallback into a match-everything query."""
    _seed_macbook(db_session)
    assert find_product(db_session, "%") is None


def test_find_review_resolves_the_same_product_as_find_product(db_session):
    """get_product and get_review must never disagree about which product a phrasing means."""
    _seed_macbook(db_session)
    review = find_review(db_session, "macbook")
    assert review is not None
    assert review.reviews_count == 3


def test_find_review_missing_returns_none(db_session):
    _seed_macbook(db_session)
    assert find_review(db_session, "nonexistent") is None


# --- cache staleness ---


def test_is_stale_false_for_a_fresh_row(db_session):
    _seed_macbook(db_session)
    assert is_stale(find_product(db_session, "macbook"), ttl_seconds=3600) is False


def test_is_stale_true_once_past_the_ttl(db_session):
    _seed_macbook(db_session)
    product = find_product(db_session, "macbook")
    product.updated_at = datetime.now(UTC) - timedelta(seconds=7200)
    db_session.commit()
    assert is_stale(product, ttl_seconds=3600) is True


def test_is_stale_treats_a_naive_timestamp_as_utc(db_session):
    """SQLite hands back naive datetimes where Postgres returns aware ones; a naive value must not
    be compared against an aware 'now' (TypeError) or misread as local time."""
    _seed_macbook(db_session)
    product = find_product(db_session, "macbook")
    product.updated_at = datetime.now(UTC).replace(tzinfo=None)
    assert is_stale(product, ttl_seconds=3600) is False
