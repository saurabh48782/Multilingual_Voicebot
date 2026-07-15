# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS python_base

# System deps: audio decoding, FAISS threading, postgres client, pre-commit tooling
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        libpq-dev \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

# HF_HOME lives under /app/data - the app resolves its data dir relative to
# the repo root (/app), and docker-compose mounts ./data there.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/app/data/hf_cache \
    LOG_TARGET=stdout

COPY pyproject.toml uv.lock /app/
WORKDIR /app
RUN uv venv /opt/venv

FROM python_base AS api_base
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY . /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python_base AS api_base_test
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project
COPY .pre-commit-config.yaml /app/
RUN git init && uv run pre-commit install-hooks
COPY . /app/

FROM api_base AS api
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app/data
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthcheck || exit 1
CMD ["scripts/start_api.sh"]

FROM api_base AS unit_test
CMD ["scripts/unit_test.sh"]

FROM api_base AS integration_test
CMD ["scripts/integration_test.sh"]

FROM api_base_test AS precommit_check
CMD ["scripts/precommit_check.sh"]
