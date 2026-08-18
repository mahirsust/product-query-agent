"""SQLAlchemy tables: the product cache and user accounts."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Product(Base):
    """Cache table for DummyJSON product data. `name` is the lowercased natural key, matching
    today's case-insensitive dict lookup in main.py."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    # The complete upstream payload as JSON. Stored whole so a question about a field nobody
    # anticipated never requires a schema change; the columns above are denormalised from it for
    # lookup and filtering. Nullable so rows written before this column existed still load.
    data: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    review: Mapped["Review"] = relationship(
        back_populates="product", uselist=False, cascade="all, delete-orphan"
    )


class Review(Base):
    """Cache table for DummyJSON review data, one-to-one with Product (DummyJSON's product
    response embeds reviews, so a single fetch populates both tables)."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), unique=True, nullable=False)
    reviews_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    # JSON array of {rating, comment} objects. Reviewer names are dropped — they are personal data
    # the agent has no use for. Nullable so rows written before this column existed still load.
    comments: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    product: Mapped["Product"] = relationship(back_populates="review")


class User(Base):
    """An account. Only the password *hash* is stored, never the password."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
