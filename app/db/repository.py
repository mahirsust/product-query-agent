"""Database queries for the product cache and user accounts."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Product, Review, User


def get_product(session: Session, name: str) -> Product | None:
    """Look up a product by its exact canonical name.

    Used by the upsert helpers, which must not match a different product by substring and
    overwrite it. Use `find_product` to resolve user-supplied names.
    """
    return session.scalar(select(Product).where(Product.name == name.lower()))


def find_product(session: Session, name: str) -> Product | None:
    """Resolve a user-supplied product name against the cache.

    Rows are stored under the catalogue's canonical title, but users type fragments, so an exact
    match falls back to a substring match. A miss simply costs one upstream call — the API's own
    search remains the authority on fuzzy matching.
    """
    exact = get_product(session, name)
    if exact is not None:
        return exact
    return session.scalar(
        select(Product)
        .where(Product.name.contains(name.lower(), autoescape=True))
        .order_by(Product.id)
    )


def is_stale(record: Product | Review, ttl_seconds: int) -> bool:
    """Whether a cached row is older than `ttl_seconds`.

    Naive timestamps are read as UTC, since SQLite returns them without a timezone where Postgres
    does not.
    """
    updated_at = record.updated_at
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return (datetime.now(UTC) - updated_at).total_seconds() > ttl_seconds


def upsert_product(
    session: Session,
    name: str,
    price: float,
    rating: float,
    description: str,
    data: str | None = None,
) -> Product:
    """Insert or refresh a cached product, keyed on the lowercased canonical name."""
    product = get_product(session, name)
    if product is None:
        product = Product(
            name=name.lower(),
            price=price,
            rating=rating,
            description=description,
            data=data,
        )
        session.add(product)
    else:
        product.price = price
        product.rating = rating
        product.description = description
        product.data = data
    session.flush()
    return product


def get_review(session: Session, name: str) -> Review | None:
    """Exact counterpart to `get_product`; `find_review` is the user-facing lookup."""
    product = get_product(session, name)
    if product is None:
        return None
    return session.scalar(select(Review).where(Review.product_id == product.id))


def find_review(session: Session, name: str) -> Review | None:
    """Look up reviews for a user-supplied product name.

    Resolves the product exactly as `find_product` does, so the two tools cannot disagree about
    which product a phrasing refers to.
    """
    product = find_product(session, name)
    if product is None:
        return None
    return session.scalar(select(Review).where(Review.product_id == product.id))


def upsert_review(
    session: Session,
    name: str,
    reviews_count: int,
    rating: float,
    comments: str | None = None,
) -> Review:
    """Insert or refresh a product's reviews.

    Raises:
        ValueError: The product is not cached; reviews cannot exist without it.
    """
    product = get_product(session, name)
    if product is None:
        raise ValueError(f"Cannot upsert review for unknown product: {name}")
    review = get_review(session, name)
    if review is None:
        review = Review(
            product_id=product.id,
            reviews_count=reviews_count,
            rating=rating,
            comments=comments,
        )
        session.add(review)
    else:
        review.reviews_count = reviews_count
        review.rating = rating
        review.comments = comments
    session.flush()
    return review


def get_user_by_username(session: Session, username: str) -> User | None:
    """Look up an account for login and duplicate-signup checks."""
    return session.scalar(select(User).where(User.username == username))


def get_user_by_id(session: Session, user_id: int) -> User | None:
    """Resolve the account referenced by a token's subject claim."""
    return session.get(User, user_id)


def create_user(session: Session, username: str, hashed_password: str) -> User:
    """Persist a new user. The password must already be hashed by `app.security.auth`."""
    user = User(username=username, hashed_password=hashed_password)
    session.add(user)
    session.flush()
    return user
