#!/bin/bash
set -ex

uv run alembic upgrade head

uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 120 --loop uvloop
