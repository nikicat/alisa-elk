"""drop pending_requests table

Phase 3 of the handler → LangGraph migration. The asyncio.Task parked
in PendingTaskRegistry is the only authority on slow-path results;
nothing has been written to pending_requests since Phase 2 landed.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-18

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("turn_log") as batch:
        batch.drop_column("pending_request_id")
    op.drop_index("ix_pending_status_updated", table_name="pending_requests")
    op.drop_index("ix_pending_session", table_name="pending_requests")
    op.drop_table("pending_requests")


def downgrade() -> None:
    op.create_table(
        "pending_requests",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("request_text", sa.Text, nullable=False),
        sa.Column("messages_json", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("response_text", sa.Text),
        sa.Column("error_text", sa.Text),
        sa.Column("wait_turns", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_pending_session", "pending_requests", ["session_id"])
    op.create_index(
        "ix_pending_status_updated", "pending_requests", ["status", "updated_at"]
    )
    with op.batch_alter_table("turn_log") as batch:
        batch.add_column(
            sa.Column(
                "pending_request_id",
                sa.String(32),
                sa.ForeignKey(
                    "pending_requests.id",
                    name="fk_turn_log_pending_request_id",
                ),
            )
        )
