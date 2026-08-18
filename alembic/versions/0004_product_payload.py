"""store the full product payload

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable so existing rows remain valid; they refill from upstream on first use.
    op.add_column("products", sa.Column("data", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("products", "data")
