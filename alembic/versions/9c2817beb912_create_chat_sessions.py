"""create chat_sessions

Revision ID: 9c2817beb912
Revises:
Create Date: 2026-07-03 05:04:46.919263

Chat-session metadata (title/timestamps/turn count) used to list and resume
past chats. Conversation *content* lives in the LangGraph Postgres checkpointer,
not here. Previously created at runtime by ``session_store.ensure_schema``;
that DDL now lives in this migration.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c2817beb912"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "chat_sessions",
        sa.Column("session_id", sa.Text(), primary_key=True, nullable=False),
        sa.Column("title", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "last_active",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("message_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("chat_sessions")
