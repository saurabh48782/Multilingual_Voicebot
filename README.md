# Multilingual Vernacular Voicebot

Voice + text RAG bot for rural / semi-urban India. It accepts spoken or typed
questions in **Hindi, Bengali, or English** about Indian government
schemes / banking / agriculture, runs a grounded retrieval-augmented generation
(RAG) pipeline with multi-turn memory, and replies by voice in the same
vernacular language.

## Table of Contents
- [Multilingual Vernacular Voicebot](#multilingual-vernacular-voicebot)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. LangGraph Execution Flow](#2-langgraph-execution-flow)
  - [3. Main Entry Points](#3-main-entry-points)
  - [4. Installation](#4-installation)
    - [Docker (full stack)](#docker-full-stack)
    - [Local (uv, host-run app)](#local-uv-host-run-app)
  - [5. Data \& Corpus Management](#5-data--corpus-management)
    - [Ingesting the corpus](#ingesting-the-corpus)
  - [6. Local Development](#6-local-development)
    - [Running the Dev Server](#running-the-dev-server)
    - [Running Tests](#running-tests)
    - [Developer Commands](#developer-commands)
  - [7. CI/CD](#7-cicd)
    - [Docker Build Targets](#docker-build-targets)
    - [Pipeline Flow](#pipeline-flow)
    - [What Each Stage Validates](#what-each-stage-validates)
  - [8. Project Structure](#8-project-structure)

## 1. Project Overview

A LangGraph-orchestrated StateGraph that transcribes a vernacular query, grounds
it in a retrieved corpus, and speaks the answer back in the same language. The
pipeline is confidence-gated and hallucination-guarded end to end, with
conversation state persisted per session for multi-turn memory.

Key components:

  - **FastAPI REST API** (`src/api/`) -- serves voice/chat/session/document
    endpoints plus the single-page UI, gated by API-key + rate-limit middleware.

  - **LangGraph Workflow** (`src/graph/`) -- stateful pipeline of one-file-per-node
    stages (transcribe → detect_lang → pii_scrub → translate → rewrite →
    classify_intent → retrieve → confidence → generate → verify → synthesize),
    with providers injected via a `Deps` container.

  - **Hybrid RAG** (`src/rag/`) -- semantic chunking, `multilingual-e5-large`
    embeddings in FAISS, BM25 lexical index, RRF fusion, and a
    `bge-reranker-v2-m3` cross-encoder with a confidence gate.

  - **Model sidecars** (`services/`) -- IndicConformer STT and Indic Parler-TTS
    run as isolated Docker images (conflicting deps / remote code) reached over
    HTTP; Ollama serves every LLM + translation call.

## 2. LangGraph Execution Flow

Each turn runs a single confidence-gated path from START. Voice-only stages
(STT/TTS) are skipped for text-in turns; `input_mode` is recorded on the first
node and drives whether audio is produced.

```
Audio/Text --> transcribe --> detect_lang --> pii_scrub --> translate->en
                                                                 |
                                                          rewrite_query (coreference)
                                                                 |
                                             classify_intent --[general]--> smalltalk
                                                     |                          |
                                                 [scheme]              (skips RAG)
                                                     |                          |
                                              retrieve (FAISS + BM25)           |
                                                     |                          |
                                     confidence_check --[fail]--> fallback      |
                                                     |               |          |
                                                 generate            |          |
                                                     |               |          |
                                   verify_groundedness --[fail]------+          |
                                                     |                          |
                                          translate->vernacular <---------------+
                                                     |
                                             synthesize (TTS on voice turns only)
                                                     |
                                                Audio/Text response
```

Three hallucination guards run in series: the retriever confidence gate, the
grounded generation prompt, and an LLM self-check in `verify_groundedness` (which
**fails closed** -- unparseable verdicts route to `fallback`). `general` turns
short-circuit retrieval via `smalltalk` and go straight to vernacular translation.

## 3. Main Entry Points

- **`src/api/app.py`** -- FastAPI application built by `create_app()`. Its lifespan
  opens the Postgres pool + `AsyncPostgresSaver` checkpointer and builds the graph
  once; serves the API + SPA on port 8000.
- **`src/graph/builder.py`** -- `build_graph(checkpointer, deps)` compiles the
  StateGraph. Nodes pull providers from the injected `Deps` container; there is no
  standalone `__main__` entry point.

## 4. Installation

### Docker (full stack)

Requires Docker + Docker Compose. For GPU paths, also the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
All services are gated behind compose profiles -- a bare `docker compose up`
starts nothing.

```bash
# 1. clone + cd into the repo, then set env (POSTGRES_PASSWORD + HF_TOKEN)
cp .env.example .env

# 2. drop your corpus (PDF / text) into data/corpus/

# 3. start the full stack (db + api + tts + stt)
docker compose --profile all up -d --build

# 4. one-shot: ingest the corpus into FAISS
docker compose --profile all exec api python -m src.data.ingestion.ingest --force

# 5. open http://localhost:8000
```

Compose profiles: `all` (db + api + tts + stt -- the full app), `dev`
(db + tts + stt, for a host-run app), and the CI profiles `unit_test`,
`integration_test`, `precommit_check`.

### Local (uv, host-run app)

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). The heavy STT/TTS
sidecars still run in Docker; only the API runs on the host.

```bash
# 1. sync deps
uv sync

# 2. env + activate pre-commit hooks
cp .env.example .env
uv run pre-commit install

# 3. bring up the dev stack + run the reloading server (see §6)
./scripts/develop.sh
```

## 5. Data & Corpus Management

The RAG corpus is **not stored in git**. Drop `*.pdf` or `*.txt` files into
`data/corpus/` (the fixed `corpus_dir`) -- one document per file, where the full
filename (including extension) becomes the `doc_id`, so `a.pdf` and `a.txt` are
distinct documents. Supported extensions are defined by `SUPPORTED_EXTENSIONS` in
`src/rag/ingestor.py`.

Ingestion builds three artifacts under `data/index/`, upserted in lockstep:

- `faiss.index` + `metadata.parquet` -- dense `multilingual-e5-large` vectors.
- `bm25_corpus.pkl` + `bm25/` -- lexical BM25 index (Devanagari/Bengali aware).
- `manifest.json` -- per-file SHA256 hashes so unchanged docs are skipped.

### Ingesting the corpus

```bash
uv run python -m src.data.ingestion.ingest              # incremental (default)
uv run python -m src.data.ingestion.ingest --force      # re-ingest unchanged docs
uv run python -m src.data.ingestion.ingest --file data/corpus/pm_kisan.pdf  # one file
uv run python -m src.data.ingestion.ingest --stats      # show index chunk counts
uv run python -m src.data.ingestion.ingest --clear      # wipe FAISS + BM25 + manifest
```

> The API keeps FAISS + BM25 as process-local singletons behind a
> readers-writer lock, so it **must run single-worker** (`WORKERS=1`, the
> `scripts/start_api.sh` default) -- multiple workers desync on the first upload.

## 6. Local Development

### Running the Dev Server

```bash
./scripts/develop.sh
```

This script:
1. Starts the `dev` compose stack (Postgres `db` + the `tts` + `stt` sidecars)
   via Docker Compose, waiting until they are healthy (`--wait`).
2. Runs Alembic database migrations (`alembic upgrade head`) to create the
   `chat_sessions` metadata schema.
3. Launches Uvicorn with hot-reload on `API_HOST:API_PORT` (default
   `0.0.0.0:8000`), running the API on the host against the containerised deps.
4. Tears the compose stack down again on exit (`Ctrl-C` / `EXIT` trap).

The API is talking to the sidecars over `STT_REMOTE_URL` / `TTS_REMOTE_URL` and
to Postgres over `CHECKPOINT_DSN` (published on `127.0.0.1:5433`) -- all set in
`.env`.

### Running Tests

**Unit tests** -- mocked providers, no external services or credentials:

```bash
uv run pytest -vv tests/unit
# or:
./scripts/unit_test.sh
```

**Integration tests** -- Tavern-based API tests that drive a live stubbed server:

```bash
uv run pytest -vv tests/integration
# or:
./scripts/integration_test.sh
```

### Developer Commands

```bash
uv run ruff check --fix src/ tests/    # lint
uv run ruff format src/ tests/         # format
uv run mypy src/                       # type check
uv run pytest                          # full suite
```

Pre-commit hooks (`ruff`, `mypy`, `bandit`, `detect-secrets`, `hadolint`) run
automatically on commit once installed (`uv run pre-commit install`); run them
across the tree with `./scripts/precommit_check.sh`.

## 7. CI/CD

### Docker Build Targets

The Dockerfile uses multi-stage builds off a shared `python:3.13-slim` base. Each
target is independent -- build and run only the one you need. They are also wired
into `docker-compose.yml` under matching profiles.

| Target             | What it does                        | Includes dev deps? |
|--------------------|-------------------------------------|--------------------|
| `api`              | Production API server (port 8000)   | No                 |
| `unit_test`        | Runs pytest on `tests/unit/`        | Yes                |
| `integration_test` | Runs Tavern integration tests       | Yes                |
| `precommit_check`  | Runs pre-commit hooks               | Yes                |

```bash
# Pre-commit checks
docker compose --profile precommit_check up --build

# Unit tests (no external services needed)
docker compose --profile unit_test up --build

# Integration tests (needs db + api)
docker compose --profile integration_test up --build

# Production stack (db + api + tts + stt)
docker compose --profile all up -d --build
```

### Pipeline Flow

All stages must pass in sequence before the production image is shipped:

```
Pre-commit Check -> Unit Tests -> Integration Tests -> Build & Push Production Image -> Deploy
```

If any stage fails, the pipeline stops. The production `api` target installs only
runtime dependencies -- no dev tools or test frameworks are included.

### What Each Stage Validates

- **Pre-commit check**: formatting (ruff format), linting (ruff check), type
  checking (mypy), security (bandit), secret scanning (detect-secrets), and
  Dockerfile linting (hadolint).
- **Unit tests**: node/provider logic with mocked STT/TTS/LLM/retriever -- no
  Postgres, sidecars, or model downloads needed.
- **Integration tests**: Tavern YAML flows driving a live stubbed API server
  (booted by the test conftest) to exercise the router + graph wiring end to end.

## 8. Project Structure

```
├── src/
│   ├── api/                   # FastAPI app, routers, security, services, audio cache
│   │   ├── app.py             # create_app() + lifespan (db pool, checkpointer, graph)
│   │   ├── routers/           # health, voice, chat, sessions, documents, audio
│   │   └── security.py        # ApiKeyMiddleware + RateLimitMiddleware
│   ├── graph/                 # LangGraph workflow engine
│   │   ├── builder.py         # build_graph(checkpointer, deps) -> compiled StateGraph
│   │   ├── state.py           # VoicebotState TypedDict (full graph state shape)
│   │   ├── deps.py            # Deps DI container + _default_* provider factories
│   │   ├── nodes/             # One file per node (transcribe, retrieve, generate, ...)
│   │   └── prompts.py         # LLM prompt templates + sanitize_untrusted()
│   ├── rag/                   # Chunker, embeddings, FAISS store, BM25, reranker, retriever
│   ├── stt/                   # STT Protocol + IndicConformer sidecar client
│   ├── tts/                   # TTS Protocol + Indic Parler-TTS sidecar client
│   ├── translation/           # Translation Protocol + Ollama (only provider)
│   ├── llm/                   # LLM Protocol + Ollama (only provider)
│   ├── db/                    # Postgres lifespan helpers + chat_sessions metadata store
│   ├── evaluation/            # RAGAS offline eval harness (CLI: python -m src.evaluation)
│   ├── frontend/              # index.html + static/{css,js}
│   └── utils/                 # config loader, logger, providers, pii, observability
├── services/
│   ├── stt_conformer/         # IndicConformer STT Docker sidecar (POST /stt)
│   └── tts_parler/            # Indic Parler-TTS Docker sidecar (POST /tts)
├── tests/
│   ├── unit/                  # Unit tests (mocked providers, no credentials)
│   └── integration/           # Tavern-based API integration tests
├── scripts/
│   ├── develop.sh             # Local dev: dev compose stack + Alembic + hot-reload uvicorn
│   ├── start_api.sh           # Production API startup (alembic upgrade + single-worker uvicorn)
│   ├── unit_test.sh           # Run unit test suite
│   ├── integration_test.sh    # Run integration test suite
│   └── precommit_check.sh     # Run pre-commit hooks across the tree
├── alembic/                   # Database migrations (own the chat_sessions DDL)
├── data/corpus/               # Drop-in PDF/TXT corpus (one doc per file)
├── data/index/                # FAISS + BM25 + manifest artifacts (built by ingestion)
├── params.yaml                # Provider choices, thresholds, and runtime config
├── Dockerfile                 # Multi-stage build (api, unit_test, integration_test, precommit_check)
└── docker-compose.yml         # Profiled service stack (all / dev / CI targets)
```
