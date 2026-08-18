"""add review comments

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable so existing rows remain valid; they refill from upstream as their TTL expires.
    op.add_column("reviews", sa.Column("comments", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("reviews", "comments")
