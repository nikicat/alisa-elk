#!/bin/sh
set -e

# Apply any pending migrations against the (possibly fresh) bind-mounted DB.
alembic upgrade head

# ROOT_PATH lets the app know it's mounted under a subpath (e.g. /elk).
# Only needed for accurate OpenAPI URLs; webhook routing works either way.
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8080 \
    --proxy-headers \
    --forwarded-allow-ips='*' \
    ${ROOT_PATH:+--root-path "$ROOT_PATH"}
