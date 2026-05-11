#!/usr/bin/env bash
# scripts/setup_timescaledb.sh
# One-command TimescaleDB setup via Docker.
# Usage: bash scripts/setup_timescaledb.sh

set -e

CONTAINER=timescaledb
DB_URL="postgresql://postgres:ascent@localhost:5432/ascent"

echo "[Setup] Starting TimescaleDB container..."
docker run -d --name "$CONTAINER" \
    -p 5432:5432 \
    -e POSTGRES_PASSWORD=ascent \
    -e POSTGRES_DB=ascent \
    timescale/timescaledb:latest-pg15

echo "[Setup] Waiting for PostgreSQL to be ready..."
for i in $(seq 1 30); do
    if docker exec "$CONTAINER" pg_isready -U postgres -q 2>/dev/null; then
        echo "[Setup] PostgreSQL ready."
        break
    fi
    echo "[Setup]   attempt $i/30..."
    sleep 2
done

echo "[Setup] Running schema..."
docker exec -i "$CONTAINER" psql -U postgres -d ascent < "$(dirname "$0")/../ascent/data/store/schema.sql"

echo "[Setup] TimescaleDB is ready at $DB_URL"
echo "[Setup] Add to .env: TIMESCALEDB_URL=$DB_URL"
