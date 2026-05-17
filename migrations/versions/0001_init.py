"""init

Revision ID: 0001
Revises:
Create Date: 2026-05-17

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("display_name", sa.String(200)),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("balance_kop", sa.Integer, nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
    )

    op.create_table(
        "linked_accounts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "yandex_application_id", sa.String(128), nullable=False, unique=True
        ),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("linked_at", sa.DateTime, nullable=False),
    )
    op.create_index(
        "ix_linked_accounts_user_id", "linked_accounts", ["user_id"]
    )

    op.create_table(
        "link_codes",
        sa.Column("code", sa.String(6), primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("used_at", sa.DateTime),
        sa.Column("used_by_app_id", sa.String(128)),
    )
    op.create_index("ix_link_codes_user_id", "link_codes", ["user_id"])
    op.create_index("ix_link_codes_expires_at", "link_codes", ["expires_at"])

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
    op.create_index(
        "ix_pending_session", "pending_requests", ["session_id"]
    )
    op.create_index(
        "ix_pending_status_updated", "pending_requests", ["status", "updated_at"]
    )

    op.create_table(
        "turn_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("ts", sa.DateTime, nullable=False),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id")),
        sa.Column("yandex_application_id", sa.String(128), nullable=False),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("message_id", sa.Integer, nullable=False),
        sa.Column("request_text", sa.Text, nullable=False),
        sa.Column("response_text", sa.Text, nullable=False),
        sa.Column("llm_input_tokens", sa.Integer),
        sa.Column("llm_output_tokens", sa.Integer),
        sa.Column("llm_ms", sa.Integer),
        sa.Column("total_ms", sa.Integer, nullable=False),
        sa.Column(
            "pending_request_id", sa.String(32), sa.ForeignKey("pending_requests.id")
        ),
    )
    op.create_index("ix_turn_log_ts", "turn_log", ["ts"])
    op.create_index("ix_turn_session_ts", "turn_log", ["session_id", "ts"])
    op.create_index(
        "ix_turn_appid_ts", "turn_log", ["yandex_application_id", "ts"]
    )


def downgrade() -> None:
    op.drop_table("turn_log")
    op.drop_table("pending_requests")
    op.drop_table("link_codes")
    op.drop_table("linked_accounts")
    op.drop_table("users")
