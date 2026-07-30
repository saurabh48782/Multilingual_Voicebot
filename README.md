# Multilingual Vernacular Voicebot

<div align="center">

[![CI](https://github.com/saurabh48782/Multilingual_Voicebot/actions/workflows/ci.yml/badge.svg)](https://github.com/saurabh48782/Multilingual_Voicebot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Core**

![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?style=flat-square&logo=langgraph&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-checkpointer-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose%20sidecars-2496ED?style=flat-square&logo=docker&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-tracing-1C3C3C?style=flat-square&logo=langchain&logoColor=white)

**Models & Speech**

![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Ollama](https://img.shields.io/badge/Ollama-Gemma-000000?style=flat-square&logo=ollama&logoColor=white)
![IndicConformer](https://img.shields.io/badge/AI4Bharat-IndicConformer%20STT-FF9933?style=flat-square)
![Indic Parler TTS](https://img.shields.io/badge/Indic%20Parler--TTS-138808?style=flat-square)

**Retrieval & Evaluation**

![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF?style=flat-square&logo=meta&logoColor=white)
![Hybrid retrieval](https://img.shields.io/badge/BM25%20%2B%20RRF-hybrid-4B8BBE?style=flat-square)
![Reranker](https://img.shields.io/badge/Reranker-bge--v2--m3-FF6F00?style=flat-square)
![Docling](https://img.shields.io/badge/Docling-layout--aware%20PDF-1F6FEB?style=flat-square)
![RAGAS](https://img.shields.io/badge/RAGAS-offline%20eval-8A2BE2?style=flat-square)

**Quality Gates**

![uv](https://img.shields.io/badge/uv-DE5FE9?style=flat-square&logo=uv&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-lint%20%26%20format-D7FF64?style=flat-square&logo=ruff&logoColor=black)
![mypy](https://img.shields.io/badge/mypy-strict-2A6DB2?style=flat-square)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-FAB040?style=flat-square&logo=precommit&logoColor=black)

**Languages**

![Hindi](https://img.shields.io/badge/हिन्दी-Hindi-FF9933?style=flat-square)
![Bengali](https://img.shields.io/badge/বাংলা-Bengali-006A4E?style=flat-square)
![English](https://img.shields.io/badge/English-en-012169?style=flat-square)

</div>

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
  - [7. Evaluation](#7-evaluation)
    - [How the pipeline is measured](#how-the-pipeline-is-measured)
    - [The metrics](#the-metrics)
    - [Latest results](#latest-results)
    - [Running an evaluation](#running-an-evaluation)
    - [How to read these numbers](#how-to-read-these-numbers)
  - [8. CI/CD](#8-cicd)
    - [Docker Build Targets](#docker-build-targets)
    - [Pipeline Flow](#pipeline-flow)
    - [What Each Stage Validates](#what-each-stage-validates)
  - [9. Project Structure](#9-project-structure)
  - [10. License](#10-license)
    - [Model licenses](#model-licenses)

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

## 7. Evaluation

Retrieval and generation quality are measured offline with
[RAGAS](https://docs.ragas.io/). The harness lives in `src/evaluation/` and is
internal-only -- no HTTP surface, never imported from the request path. It runs
entirely against the local Ollama server, so there is no hosted judge and no API
key involved.

**[`src/evaluation/README.md`](src/evaluation/README.md) is the design doc** --
per-metric formulas, judge configuration rationale, and known limitations.

### How the pipeline is measured

Two phases, deliberately separated:

1. **`generate`** -- synthesizes a testset *once* from the already-ingested index
   (`metadata.parquet`, not the raw PDFs, so every question is answerable from
   text the retriever can actually return). RAGAS builds a knowledge graph over
   the chunks and emits questions + reference answers. Questions are synthesized
   in English and machine-translated to `hi`/`bn`, so all three languages share
   one semantic testset -- a vernacular score gap is attributable to the
   translation + retrieval path, not to different questions.
2. **`run`** -- replays the *real* production retrieval + generation path over
   that fixed testset (same prompts, same confidence gate, same reranker, same
   `INSUFFICIENT_CONTEXT` refusal behaviour) and has the judge score the output.
   Because the testset is a committed artifact, scores are comparable across
   retrieval and prompt changes.

The judge is `gemma4:12b` on Ollama with `temperature=0.0` and `format=json`;
embeddings are the same `multilingual-e5-large` model the index is built with, so
similarity metrics are measured in the index's own embedding space.

### The metrics

All scored 0.0--1.0, higher is better.

| Metric | Question it answers | Denominator |
|---|---|---|
| `context_precision` | Are the *useful* chunks ranked first? (rank-aware average precision) | all rows |
| `context_recall` | Did retrieval bring back every claim the reference answer needs? | all rows |
| `faithfulness` | Is every statement in the answer inferable from the retrieved context? (the hallucination metric) | answered rows |
| `answer_relevancy` | Does the answer actually address the question, without hedging or drift? | answered rows |
| `answer_correctness` | Composite: `0.75 × claim-F1 + 0.25 × answer_similarity` | answered rows |
| `answer_similarity` | Embedding cosine vs. the reference answer (no LLM) | answered rows |
| `answered_rate` | Fraction of rows that produced a grounded answer rather than a refusal | all rows |

Answer-quality metrics are averaged over **answered rows only**. An empty
fallback is a *non-answer*, not a wrong answer; folding refusals in conflates two
independent axes (it dragged `answer_correctness` from 0.74 to 0.56 on one run).
`answered_rate` is therefore a first-class metric with its own threshold -- a bot
that refuses everything would otherwise score perfectly.

### Latest results

100-row testset, all three languages, judged 2026-07-22
(`data/eval/reports/report-20260722T230459Z.json`; `retrieval_threshold=0.55`,
`retrieval_gap_threshold=0.03`, generator and judge both `gemma4:12b`).

| Metric | Threshold | en | hi | bn |
|---|---|---|---|---|
| `context_precision` | 0.70 | **0.908** | 0.858 | 0.853 |
| `context_recall` | 0.70 | **0.970** | 0.943 | 0.923 |
| `faithfulness` | 0.80 | **0.996** | 0.994 | 0.989 |
| `answer_relevancy` | 0.75 | **0.892** | 0.889 | 0.887 |
| `answer_correctness` | 0.60 | **0.738** | 0.747 | 0.738 |
| `answer_similarity` | -- | **0.951** | 0.947 | 0.952 |
| `answered_rate` | 0.70 | **0.84** | 0.79 | 0.74 |

Every metric clears its threshold in all three languages. The visible pattern is
the expected one: quality metrics are near-flat across languages (the pipeline
translates to English before retrieving and generating), while **coverage**
degrades down the language ladder -- 0.84 → 0.79 → 0.74 -- because translation
noise in the vernacular query costs retrieval confidence and trips the gate more
often. `context_precision` shows the same ~5-point vernacular drop for the same
reason.

Wall-clock for the full 300-row (100 × 3 languages) run was ~11.5 h on a single
24 GB GPU co-hosting the RAG, STT and TTS models; the run is dominated by
serialized Ollama judge calls, not by retrieval.

### Running an evaluation

```bash
uv run python -m src.evaluation generate                  # synth testset -> data/eval/testset.jsonl
uv run python -m src.evaluation generate --size 30 --languages en,hi
uv run python -m src.evaluation run                       # score all languages from params.yaml
uv run python -m src.evaluation run --languages en --sample-size 10
uv run pytest -m eval                                     # regression gate (manual)
```

Prerequisites: an ingested corpus (`data/index/metadata.parquet`) and a running
Ollama with `evaluation.judge_model`, `llm.model` and `translation.ollama_model`
pulled. All tuning -- judge options, metric list, thresholds, sampling -- lives
under `evaluation:` in `params.yaml`. Reports land in `data/eval/reports/` with
per-language aggregates, every per-row score, and a `config_snapshot` so an old
report stays interpretable after config drift.

`tests/integration/test_ragas_regression.py` asserts the thresholds against a
fresh English-only run. It is marked `eval` and excluded from the default
`pytest` invocation (`addopts = "-m 'not eval'"`) -- it needs a live Ollama and
burns hundreds of local LLM calls, so it is run deliberately, not in CI.

### How to read these numbers

Points an external reviewer should weigh:

- **The judge shares a model family with the generator.** A same-family judge
  tends to rate the generator's own hallucinations as faithful, so
  `faithfulness` (0.99) is optimistic and should be read as "no gross
  ungrounded output" rather than a precise figure. The harness logs a warning
  when `judge_model == llm.model`. The constraint is GPU budget, not preference
  -- see the README's judge section for the alternatives that were tried.
- **Reference answers are LLM-synthesized**, so exact claim overlap is noisy.
  That is why `answer_correctness` carries the lowest quality floor (0.60)
  despite being the closest single number to "was the citizen served correctly".
- **`answer_similarity` is a smoke signal, not correctness.** It is pure cosine:
  a fluent answer that inverts one eligibility number still scores high. It
  intentionally carries no threshold.
- **Read `answered_rate` first.** Quality metrics exclude refusals by design, so
  they are only meaningful next to coverage.
- **`hi`/`bn` scores fold in translation quality.** There is no ablation that
  isolates retrieval from translation.
- **Thresholds are floors we intend to hold, not measured baselines**, and scores
  are not comparable across testset regenerations -- regenerate deliberately and
  keep the old report for the diff.

## 8. CI/CD

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

## 9. Project Structure

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

## 10. License

This project's **source code** is released under the [MIT License](LICENSE) —
free to use, modify, and distribute, with attribution and no warranty.

### Model licenses

The MIT license covers this repository's code only. It does **not** relicense
the pretrained models the pipeline downloads at runtime, each of which carries
its own terms. Anyone deploying this stack is bound by the terms below, not by
this repository's license.

| Model | Role | License |
|---|---|---|
| [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) | Dense embeddings (FAISS index) | MIT |
| [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Cross-encoder reranker | Apache-2.0 |
| [`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | Speech-to-text (STT sidecar) | MIT |
| [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts) | Text-to-speech (TTS sidecar) | Apache-2.0 |
| `gemma4:12b`, `translategemma:4b` (via Ollama) | Generation, query rewrite, groundedness verification, translation, RAGAS judge | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |

> **Note on the Gemma models.** The Gemma Terms of Use are *not* an OSI-approved
> open-source licence. They permit commercial use but impose a use-restriction
> policy and require those restrictions to be passed downstream. If that is
> unacceptable for your deployment, `llm.provider` / `translation.provider` in
> `params.yaml` point at Ollama, so any other Ollama-served model can be
> substituted by changing `llm.model` and `translation.ollama_model` — no code
> change required.

The corpus in `data/corpus/` is **not** distributed with this repository; the
licensing of any documents you ingest is yours to verify.
