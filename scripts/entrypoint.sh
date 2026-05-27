#!/bin/sh
set -e

# Run migrations if the command is for the app (uvicorn)
if [ "$1" = "uvicorn" ]; then
    echo "Running database migrations..."
    alembic upgrade head
    echo "Migrations complete."
fi

exec "$@"
