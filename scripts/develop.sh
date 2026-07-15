#!/bin/bash
set -ex

cleanup() {
    trap - INT TERM
    echo "Shutting down Docker containers..."
    set +e
    docker compose --profile dev down
}

trap cleanup EXIT INT TERM

docker compose --profile dev up --build -d --wait db tts stt

uv run alembic upgrade head
uv run uvicorn src.api.app:app --reload --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}"
