"""phase 3 hardening constraints

Revision ID: c8f5e2a31d04
Revises: b7e4d9f21c03
Create Date: 2026-07-23 23:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c8f5e2a31d04"
down_revision: Union[str, Sequence[str], None] = "b7e4d9f21c03"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add billing and default-selection integrity constraints."""
    op.drop_index(op.f("ix_usage_records_request_id"), table_name="usage_records")
    op.create_unique_constraint(
        "uq_usage_records_request_id",
        "usage_records",
        ["request_id"],
    )

    op.create_index(
        "uq_ai_models_one_default_per_provider",
        "ai_models",
        ["provider_id"],
        unique=True,
        postgresql_where=sa.text("is_default IS TRUE"),
    )

    op.create_index(
        "uq_api_keys_one_default_per_provider",
        "api_keys",
        ["provider_id"],
        unique=True,
        postgresql_where=sa.text("is_default IS TRUE"),
    )


def downgrade() -> None:
    """Remove billing and default-selection integrity constraints."""
    op.drop_index(
        "uq_api_keys_one_default_per_provider",
        table_name="api_keys",
        postgresql_where=sa.text("is_default IS TRUE"),
    )
    op.drop_index(
        "uq_ai_models_one_default_per_provider",
        table_name="ai_models",
        postgresql_where=sa.text("is_default IS TRUE"),
    )
    op.drop_constraint("uq_usage_records_request_id", "usage_records", type_="unique")
    op.create_index(
        op.f("ix_usage_records_request_id"),
        "usage_records",
        ["request_id"],
        unique=False,
    )
