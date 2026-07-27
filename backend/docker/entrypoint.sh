#!/bin/sh
set -eu

if [ "${SKIP_MIGRATIONS:-0}" != "1" ]; then
    echo "Waiting for database..."
    python /app/docker/wait_for_db.py

    echo "Running database migrations..."
    alembic upgrade head
fi

exec "$@"
