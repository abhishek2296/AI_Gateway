"""initial schema

Revision ID: a3f6c2d18e01
Revises:
Create Date: 2026-07-23 21:08:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a3f6c2d18e01"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all AI Gateway domain tables."""
    op.create_table(
        "prompt_templates",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("template", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("variables", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "version", name="uq_prompt_templates_name_version"),
    )
    op.create_index(op.f("ix_prompt_templates_category"), "prompt_templates", ["category"], unique=False)
    op.create_index(op.f("ix_prompt_templates_is_active"), "prompt_templates", ["is_active"], unique=False)
    op.create_index(op.f("ix_prompt_templates_name"), "prompt_templates", ["name"], unique=False)

    op.create_table(
        "providers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("provider_type", sa.String(length=100), nullable=False),
        sa.Column("base_url", sa.Text(), nullable=True),
        sa.Column("api_version", sa.String(length=50), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("is_local", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("supports_streaming", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("supports_embeddings", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("supports_function_calling", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("supports_vision", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("supports_audio", sa.Boolean(), server_default=sa.text("false"), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_providers_is_active"), "providers", ["is_active"], unique=False)
    op.create_index(op.f("ix_providers_name"), "providers", ["name"], unique=True)
    op.create_index(op.f("ix_providers_provider_type"), "providers", ["provider_type"], unique=False)

    op.create_table(
        "ai_models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("context_window", sa.Integer(), nullable=True),
        sa.Column("max_output_tokens", sa.Integer(), nullable=True),
        sa.Column("supports_tools", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("supports_json", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("supports_images", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("supports_streaming", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("is_default", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
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
        sa.UniqueConstraint("provider_id", "model_name", name="uq_ai_models_provider_id_model_name"),
    )
    op.create_index(op.f("ix_ai_models_is_active"), "ai_models", ["is_active"], unique=False)
    op.create_index(op.f("ix_ai_models_is_default"), "ai_models", ["is_default"], unique=False)
    op.create_index(op.f("ix_ai_models_model_name"), "ai_models", ["model_name"], unique=False)
    op.create_index(op.f("ix_ai_models_provider_id"), "ai_models", ["provider_id"], unique=False)

    op.create_table(
        "provider_configurations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("api_key_env", sa.String(length=255), nullable=True),
        sa.Column("endpoint", sa.Text(), nullable=True),
        sa.Column("organization", sa.String(length=255), nullable=True),
        sa.Column("region", sa.String(length=100), nullable=True),
        sa.Column("project", sa.String(length=255), nullable=True),
        sa.Column("timeout_seconds", sa.Integer(), nullable=True),
        sa.Column("max_retries", sa.Integer(), nullable=True),
        sa.Column("verify_ssl", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("proxy_url", sa.Text(), nullable=True),
        sa.Column("extra_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
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
        sa.UniqueConstraint("provider_id", name="uq_provider_configurations_provider_id"),
    )
    op.create_index(
        op.f("ix_provider_configurations_is_active"),
        "provider_configurations",
        ["is_active"],
        unique=False,
    )
    op.create_index(
        op.f("ix_provider_configurations_provider_id"),
        "provider_configurations",
        ["provider_id"],
        unique=False,
    )

    op.create_table(
        "ai_model_configurations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("ai_model_id", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("top_p", sa.Float(), nullable=True),
        sa.Column("top_k", sa.Integer(), nullable=True),
        sa.Column("frequency_penalty", sa.Float(), nullable=True),
        sa.Column("presence_penalty", sa.Float(), nullable=True),
        sa.Column("max_tokens", sa.Integer(), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("system_prompt_template", sa.Text(), nullable=True),
        sa.Column("json_mode_default", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("tool_choice_default", sa.String(length=100), nullable=True),
        sa.Column("stream_default", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("extra_parameters", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
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
        sa.ForeignKeyConstraint(["ai_model_id"], ["ai_models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ai_model_id", name="uq_ai_model_configurations_ai_model_id"),
    )
    op.create_index(
        op.f("ix_ai_model_configurations_ai_model_id"),
        "ai_model_configurations",
        ["ai_model_id"],
        unique=False,
    )

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("session_uuid", sa.Uuid(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("ai_model_id", sa.Integer(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("temperature_override", sa.Float(), nullable=True),
        sa.Column("max_tokens_override", sa.Integer(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("is_archived", sa.Boolean(), server_default=sa.text("false"), nullable=False),
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
        sa.ForeignKeyConstraint(["ai_model_id"], ["ai_models.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_sessions_ai_model_id"), "chat_sessions", ["ai_model_id"], unique=False)
    op.create_index(op.f("ix_chat_sessions_is_archived"), "chat_sessions", ["is_archived"], unique=False)
    op.create_index(op.f("ix_chat_sessions_provider_id"), "chat_sessions", ["provider_id"], unique=False)
    op.create_index(op.f("ix_chat_sessions_session_uuid"), "chat_sessions", ["session_uuid"], unique=True)

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("provider_response_id", sa.String(length=255), nullable=True),
        sa.Column("finish_reason", sa.String(length=100), nullable=True),
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
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_messages_provider_response_id"), "messages", ["provider_response_id"], unique=False)
    op.create_index(op.f("ix_messages_role"), "messages", ["role"], unique=False)
    op.create_index(op.f("ix_messages_session_id"), "messages", ["session_id"], unique=False)


def downgrade() -> None:
    """Drop all AI Gateway domain tables."""
    op.drop_index(op.f("ix_messages_session_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_role"), table_name="messages")
    op.drop_index(op.f("ix_messages_provider_response_id"), table_name="messages")
    op.drop_table("messages")

    op.drop_index(op.f("ix_chat_sessions_session_uuid"), table_name="chat_sessions")
    op.drop_index(op.f("ix_chat_sessions_provider_id"), table_name="chat_sessions")
    op.drop_index(op.f("ix_chat_sessions_is_archived"), table_name="chat_sessions")
    op.drop_index(op.f("ix_chat_sessions_ai_model_id"), table_name="chat_sessions")
    op.drop_table("chat_sessions")

    op.drop_index(op.f("ix_ai_model_configurations_ai_model_id"), table_name="ai_model_configurations")
    op.drop_table("ai_model_configurations")

    op.drop_index(op.f("ix_provider_configurations_provider_id"), table_name="provider_configurations")
    op.drop_index(op.f("ix_provider_configurations_is_active"), table_name="provider_configurations")
    op.drop_table("provider_configurations")

    op.drop_index(op.f("ix_ai_models_provider_id"), table_name="ai_models")
    op.drop_index(op.f("ix_ai_models_model_name"), table_name="ai_models")
    op.drop_index(op.f("ix_ai_models_is_default"), table_name="ai_models")
    op.drop_index(op.f("ix_ai_models_is_active"), table_name="ai_models")
    op.drop_table("ai_models")

    op.drop_index(op.f("ix_providers_provider_type"), table_name="providers")
    op.drop_index(op.f("ix_providers_name"), table_name="providers")
    op.drop_index(op.f("ix_providers_is_active"), table_name="providers")
    op.drop_table("providers")

    op.drop_index(op.f("ix_prompt_templates_name"), table_name="prompt_templates")
    op.drop_index(op.f("ix_prompt_templates_is_active"), table_name="prompt_templates")
    op.drop_index(op.f("ix_prompt_templates_category"), table_name="prompt_templates")
    op.drop_table("prompt_templates")
