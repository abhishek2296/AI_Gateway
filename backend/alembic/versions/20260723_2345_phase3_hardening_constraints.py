"""phase 3 hardening constraints

Revision ID: c8f5e2a31d04
Revises: b7e4d9f21c03
Create Date: 2026-07-23 23:45:00.000000

Tightens two categories of data integrity that the initial schema
(``a3f6c2d18e01``) and operational tables migration (``b7e4d9f21c03``) left
unenforced at the database level:

1. ``usage_records.request_id`` becomes a true unique constraint (it was
   only a non-unique index before), so a duplicate request id — e.g. from a
   retried request being recorded twice — is rejected outright rather than
   silently allowed.
2. "At most one default" invariants for ``ai_models`` and ``api_keys`` are
   now enforced via PostgreSQL partial unique indexes (``WHERE is_default
   IS TRUE``), guaranteeing a provider can never end up with two rows both
   flagged as the default — previously this was only guaranteed by
   application logic, which is easy to bypass via a raw SQL update or a
   application-level race condition.
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
    # The plain (non-unique) index created in b7e4d9f21c03 must be dropped
    # first: PostgreSQL will not let a unique constraint and a regular index
    # both exist on the exact same column with the same implicit name
    # collision potential, and the unique constraint below supersedes it
    # (a unique constraint also serves as a lookup index).
    op.drop_index(op.f("ix_usage_records_request_id"), table_name="usage_records")
    op.create_unique_constraint(
        "uq_usage_records_request_id",
        "usage_records",
        ["request_id"],
    )

    # Partial unique index: only rows where is_default IS TRUE participate
    # in the uniqueness check, so any number of non-default models per
    # provider are allowed, but at most one may be the default. A regular
    # (non-partial) unique constraint on provider_id alone would incorrectly
    # limit each provider to a single model overall.
    op.create_index(
        "uq_ai_models_one_default_per_provider",
        "ai_models",
        ["provider_id"],
        unique=True,
        postgresql_where=sa.text("is_default IS TRUE"),
    )

    # Same partial-unique-index pattern as above, applied to API keys.
    op.create_index(
        "uq_api_keys_one_default_per_provider",
        "api_keys",
        ["provider_id"],
        unique=True,
        postgresql_where=sa.text("is_default IS TRUE"),
    )


def downgrade() -> None:
    """Remove billing and default-selection integrity constraints."""
    # Reverse in the opposite order to upgrade() so each drop only ever
    # references objects that still exist at that point.
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
    # Restore the original non-unique index so the schema exactly matches
    # its pre-upgrade state.
    op.create_index(
        op.f("ix_usage_records_request_id"),
        "usage_records",
        ["request_id"],
        unique=False,
    )
