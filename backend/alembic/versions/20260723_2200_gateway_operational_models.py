"""gateway operational models

Revision ID: b7e4d9f21c03
Revises: a3f6c2d18e01
Create Date: 2026-07-23 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b7e4d9f21c03"
down_revision: Union[str, Sequence[str], None] = "a3f6c2d18e01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add APIKey, UsageRecord, and ProviderHealth operational tables."""
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("key_identifier", sa.String(length=255), nullable=False),
        sa.Column("api_key_env", sa.String(length=255), nullable=False),
        sa.Column("organization", sa.String(length=255), nullable=True),
        sa.Column("project", sa.String(length=255), nullable=True),
        sa.Column("is_default", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_id", "name", name="uq_api_keys_provider_id_name"),
    )
    op.create_index(op.f("ix_api_keys_is_active"), "api_keys", ["is_active"], unique=False)
    op.create_index(op.f("ix_api_keys_is_default"), "api_keys", ["is_default"], unique=False)
    op.create_index(op.f("ix_api_keys_key_identifier"), "api_keys", ["key_identifier"], unique=False)
    op.create_index(op.f("ix_api_keys_provider_id"), "api_keys", ["provider_id"], unique=False)

    op.create_table(
        "provider_health",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_failure_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("health_score", sa.Float(), nullable=True),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_provider_health_checked_at"), "provider_health", ["checked_at"], unique=False)
    op.create_index(op.f("ix_provider_health_provider_id"), "provider_health", ["provider_id"], unique=False)
    op.create_index(op.f("ix_provider_health_status"), "provider_health", ["status"], unique=False)

    op.create_table(
        "usage_records",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chat_session_id", sa.Integer(), nullable=True),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("ai_model_id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("request_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["ai_model_id"], ["ai_models.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["chat_session_id"], ["chat_sessions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_usage_records_ai_model_id"), "usage_records", ["ai_model_id"], unique=False)
    op.create_index(op.f("ix_usage_records_chat_session_id"), "usage_records", ["chat_session_id"], unique=False)
    op.create_index(op.f("ix_usage_records_provider_id"), "usage_records", ["provider_id"], unique=False)
    op.create_index(op.f("ix_usage_records_request_id"), "usage_records", ["request_id"], unique=False)
    op.create_index(op.f("ix_usage_records_request_timestamp"), "usage_records", ["request_timestamp"], unique=False)
    op.create_index(op.f("ix_usage_records_status"), "usage_records", ["status"], unique=False)


def downgrade() -> None:
    """Remove operational tables."""
    op.drop_index(op.f("ix_usage_records_status"), table_name="usage_records")
    op.drop_index(op.f("ix_usage_records_request_timestamp"), table_name="usage_records")
    op.drop_index(op.f("ix_usage_records_request_id"), table_name="usage_records")
    op.drop_index(op.f("ix_usage_records_provider_id"), table_name="usage_records")
    op.drop_index(op.f("ix_usage_records_chat_session_id"), table_name="usage_records")
    op.drop_index(op.f("ix_usage_records_ai_model_id"), table_name="usage_records")
    op.drop_table("usage_records")

    op.drop_index(op.f("ix_provider_health_status"), table_name="provider_health")
    op.drop_index(op.f("ix_provider_health_provider_id"), table_name="provider_health")
    op.drop_index(op.f("ix_provider_health_checked_at"), table_name="provider_health")
    op.drop_table("provider_health")

    op.drop_index(op.f("ix_api_keys_provider_id"), table_name="api_keys")
    op.drop_index(op.f("ix_api_keys_key_identifier"), table_name="api_keys")
    op.drop_index(op.f("ix_api_keys_is_default"), table_name="api_keys")
    op.drop_index(op.f("ix_api_keys_is_active"), table_name="api_keys")
    op.drop_table("api_keys")
