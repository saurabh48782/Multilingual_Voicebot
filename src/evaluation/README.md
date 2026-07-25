# RAG Evaluation (`src/evaluation`)

Offline, RAGAS-based quality measurement for the voicebot's retrieval + generation
path. Internal only — no HTTP surface, no import from the request path. Runs
entirely on the local Ollama server (no hosted judge, no API key).

```bash
# 1. synthesize a testset from the already-ingested corpus
uv run python -m src.evaluation generate
uv run python -m src.evaluation generate --size 30 --languages en,hi

# 2. score the pipeline against it
uv run python -m src.evaluation run
uv run python -m src.evaluation run --languages en --sample-size 10

# 3. regression gate (manual, opt-in marker)
uv run pytest -m eval
```

Prerequisites: an ingested corpus (`data/index/metadata.parquet` must exist) and a
running Ollama with `evaluation.judge_model` + `llm.model` + `translation.ollama_model`
pulled. All tuning lives under `evaluation:` in `params.yaml`.

| File | Role |
|---|---|
| `__main__.py` | argparse CLI (`generate` / `run`); sets up logging + LangSmith tracing |
| `__init__.py` | `langchain_community.chat_models.vertexai` import shim (see below) |
| `adapters.py` | `ChatOllama` judge + E5 embedder, wrapped for RAGAS |
| `testset_gen.py` | synthetic Q/ground-truth generation → `data/eval/testset.jsonl` |
| `ragas_eval.py` | retrieve+generate per row → RAGAS metrics → JSON report |

---

## The two phases, and why they are separate

**Phase 1 — `generate`** builds the testset *once*. RAGAS reads corpus chunks,
extracts headlines/summaries/themes via LLM ("transforms"), builds a knowledge
graph over them, then synthesizes questions with reference answers. This is the
expensive, LLM-heavy, one-off step.

**Phase 2 — `run`** replays our real pipeline over that fixed testset and asks
the judge to score the results. This is the step you repeat after every retrieval
or prompt change, so its inputs must not drift — which is exactly why the testset
is a committed artifact on disk rather than regenerated per run.

## Testset generation decisions

**Source is the ingested index, not the raw corpus.** `_load_corpus_documents()`
reads English chunk text out of `metadata.parquet` rather than re-parsing PDFs.
Two reasons: it skips a full Docling re-parse, and it guarantees every question is
answerable from text the retriever can actually return. If we generated questions
from raw PDF text, chunking artifacts would show up as retrieval failures that
have nothing to do with retrieval quality.

**Chunks under 200 chars are dropped.** RAGAS's synthesizers cannot extract a
question from a one-sentence fragment; those rows fail the transform stage and
pollute the run with parse errors.

**English-first, then translated.** Questions and ground truths are synthesized in
English and round-tripped through `OllamaTranslator` into `hi`/`bn`, stored under
`translations` on the same row. So all three languages share one semantic testset —
a hi/bn score difference is attributable to the translation+retrieval path, not to
a different set of questions. A per-language translation failure is logged and that
language is simply absent for the row (never fatal).

**`query_distribution` is coarser than it looks.** RAGAS ≥0.2 replaced the old
evolution types with synthesizers. `simple` and `reasoning` both map to
`SingleHopSpecificQuerySynthesizer`; only `multi_context` maps to
`MultiHopSpecificQuerySynthesizer`. The weights are honored, so the effective split
is 75% single-hop / 25% multi-hop. Keeping three keys preserves the intent (and the
report's `evolution_type` column) for when distinct synthesizers are wired up.

## Judge configuration (`adapters.py`)

The judge is `langchain_ollama.ChatOllama` on `evaluation.judge_model`, sharing
`llm.ollama_base_url` with production. Every option is set deliberately:

- **`format="json"`** — every RAGAS metric parses the judge's reply as JSON into a
  Pydantic model. Without this, prose preambles break the parse.
- **`reasoning=False`** (`evaluation.judge_reasoning`) — a reasoning model spends
  its `num_predict` budget inside `<think>` and returns *empty* answer content.
  Empty string → invalid JSON → `ValidationError` mid-transform. This was the
  concrete failure that killed `generate` runs on `qwen3.5:4b`.
- **`num_ctx`** (`evaluation.judge_num_ctx`, falling back to `llm.num_ctx`) — always
  sent. Ollama's default window (2048/4096) is smaller than a RAGAS prompt plus a
  corpus chunk, and it truncates **from the front**, silently discarding the
  instructions before the model ever sees them. Same reasoning as production's
  `llm.num_ctx`.
- **`temperature=0.0`**, **`num_predict=2048`** — deterministic verdicts, enough
  room for multi-claim faithfulness output.

**Model choice: `gemma4:12b`, with a known caveat.** Ideally the judge is a
*different* model family from the generator — a same-family judge tends to rate the
generator's own hallucinations as faithful, inflating faithfulness and correctness.
`evaluate_testset` logs a warning when `judge_model == llm.model` for exactly this
reason. We still ship the same model because (a) the 24 GB GPU is already
co-hosting e5-large, bge-reranker, TTS and STT and cannot keep a second 12B family
resident, and (b) the alternative we tried, `qwen3.5:4b`, over-escapes JSON in ~20%
of replies even with `format=json` and produced empty content in reasoning mode —
a broken judge is worse than a correlated one. Override `evaluation.judge_model`
when GPU budget allows.

**Embeddings: `E5Embeddings` wrapper, not raw `HuggingFaceEmbeddings`.** E5 requires
`passage: ` on documents and `query: ` on queries; mixing them collapses cosine
quality. `src/rag/embedder.py` enforces this at ingest time, so the wrapper mirrors
it here — otherwise `answer_similarity` and `answer_relevancy` would be measured in
a different embedding space than the index lives in.

Always hand RAGAS `ragas_llm_wrapper()` / `ragas_embeddings_wrapper()`, never the
raw `ChatOllama` or `Embeddings` object. RAGAS's retry, timeout and prompt-callback
machinery lives in the wrapper; passing the raw client bypasses it.

## Why two separate `run_config` blocks in `params.yaml`

`evaluation.testset.run_config` governs phase 1 (`TestsetGenerator`); `evaluation.run_config`
governs phase 2 (`ragas.evaluate`). They are separate because the two phases have
different failure semantics and different tuning pressure:

- Phase 1 is a long one-off over hundreds of transform tasks, and needs
  **`raise_exceptions: false`** so a single unrecoverable node is skipped and logged
  instead of aborting a 40-minute run. `ragas.evaluate()` has no such knob, so the
  key exists only under `testset`.
- Phase 2 is re-run constantly and its row count is controlled by `--sample-size`,
  so its concurrency is worth tuning independently of generation.

Both override RAGAS's defaults of `timeout=180` / `max_workers=16`, which are wrong
for a single-GPU Ollama: 16 concurrent judge calls do not run concurrently — Ollama
serializes them — so back-of-queue calls blow past 180 s. RAGAS then nulls the
affected results, and since a metric's aggregate is computed over its column, whole
metrics come back `NaN` (`context_precision`, `answer_similarity` were the ones we
lost). `timeout: 300` + `max_workers: 4` keeps each call inside its budget;
`max_retries: 10` with `max_wait: 60` absorbs transient parse failures.

Omitting `run_config` from `ragas.evaluate()` entirely — which is what the code did
before — is the same bug with no way to configure around it.

## `run` phase decisions (`ragas_eval.py`)

**Eval mirrors production, including its refusals.** `_generate_answer` reuses the
production `GENERATE_SYSTEM` / `GENERATE_PROMPT`, `sanitize_untrusted()` on each
context, `temperature=0.3`, and native thinking when `llm.think_on_generate` — a
thinking generator answers differently than a plain one, so eval must match. Three
production behaviors are reproduced as an **empty answer**:

1. the confidence gate failing (`not result.passed`) → user sees the fallback,
2. the model emitting `INSUFFICIENT_CONTEXT` → same,
3. an Ollama error mid-generation → prod's `llm_error` fallback path.

Case 3 is caught per row rather than propagated, so one `ReadTimeout` does not
discard every already-completed row in the batch.

**Vernacular rows are judged in English space.** For `hi`/`bn` the stored vernacular
question is translated to English (mirroring the graph's `translate→en` node), used
for retrieval, and the answer is kept in English to be judged against the English
ground truth. The judge never has to reason cross-lingually — otherwise a drop in
hi/bn scores could be the judge's Hindi weakness rather than the pipeline's.

**`answered_rate` is a first-class metric, and answer-quality metrics are averaged
over answered rows only.** An empty fallback is a *non-answer*, not a wrong answer.
Averaging `faithfulness`/`answer_relevancy`/`answer_correctness`/`answer_similarity`
over empty rows conflates two independent axes and understates real quality — this
measurably dragged `answer_correctness` from 0.74 to 0.56 on one run. So:

| axis | metrics | denominator |
|---|---|---|
| coverage | `answered_rate` | all rows |
| retrieval | `context_precision`, `context_recall` | all rows (the retriever runs regardless of the gate) |
| answer quality | `faithfulness`, `answer_relevancy`, `answer_correctness`, `answer_similarity` | answered rows only |

A pipeline that refuses everything therefore shows perfect quality and a floored
`answered_rate` — which is why `answered_rate` carries its own threshold (0.70) in
`evaluation.thresholds`. Read the two together, never quality alone. If RAGAS
returns a different row count than we sent (dropped rows), the answered-only mask
is abandoned with a warning and quality falls back to all rows rather than
mis-aligning.

**`--sample-size` samples randomly with a fixed seed.** `sample_strategy: random`
+ `sample_seed: 42`. A head-first slice would inherit the testset's generation
order (all `simple` queries first), overstating easy cases. `sample_strategy: head`
exists for reproducible smoke tests.

**Reports** land in `data/eval/reports/report-<UTC timestamp>.json`: per-language
aggregates, every per-row score, and a `config_snapshot` (judge model, generation
model, embedding model, both gate thresholds) so an old report stays interpretable
after config drift. `list_reports()` / `load_report()` are read helpers over that
directory.

## The VertexAI import shim (`__init__.py`)

`ragas` imports `langchain_community.chat_models.vertexai` at module import time;
newer `langchain-community` releases removed it, so `import ragas` raises
`ModuleNotFoundError` before any of our code runs. `__init__.py` registers a stub
module exposing a `ChatVertexAI` placeholder if the real one is missing. It is never
instantiated — evaluation runs entirely on Ollama — it exists only so the import
resolves. Delete it once `ragas` drops the reference.

## The metrics

Configured in `evaluation.metrics`; each name maps to a RAGAS class via
`_METRIC_REGISTRY` in `ragas_eval.py`. RAGAS names its own output columns differently
from our config keys (e.g. `context_precision` →
`llm_context_precision_with_reference`), which is why `metric_columns` pairs them up
before aggregation. All six are scored on **0.0–1.0, higher is better**. Values below
are RAGAS 0.4.3 defaults.

| our key | RAGAS class | judged by | inputs | denominator |
|---|---|---|---|---|
| `context_precision` | `LLMContextPrecisionWithReference` | LLM | question, contexts, reference | all rows |
| `context_recall` | `LLMContextRecall` | LLM | question, contexts, reference | all rows |
| `faithfulness` | `Faithfulness` | LLM | contexts, answer | answered rows |
| `answer_relevancy` | `ResponseRelevancy` | LLM + embeddings | question, answer | answered rows |
| `answer_correctness` | `AnswerCorrectness` | LLM + embeddings | answer, reference | answered rows |
| `answer_similarity` | `SemanticSimilarity` | embeddings only | answer, reference | answered rows |
| `answered_rate` | *(ours, not RAGAS)* | — | answer | all rows |

Note the cost asymmetry: `answer_similarity` is two embedding calls, while
`answer_correctness` and `faithfulness` each fan out into several judge calls per row.
Dropping metrics from `evaluation.metrics` is the main lever on run time.

### `context_precision` — are the *useful* chunks ranked first?

The judge labels each retrieved chunk relevant or not (`v_k ∈ {0,1}`) with respect to
answering the question given the reference answer. The score is average precision over
the ranked list:

```
Precision@k = (relevant chunks within the top k) / k
context_precision = Σ_k (Precision@k · v_k) / (number of relevant chunks in top K)
```

Rank-aware: one relevant chunk at position 1 scores far better than the same chunk at
position 5. This is the metric that moves when the **reranker** changes — it measures
ordering, not whether the answer was retrieved at all. Independent of our generated
answer, hence all rows.

### `context_recall` — did retrieval bring back everything the answer needs?

The judge splits the **reference** answer into individual claims and, per claim, decides
whether it is attributable to the retrieved contexts:

```
context_recall = (reference claims attributable to the contexts) / (total reference claims)
```

The complement of `context_precision`: recall catches *missing* evidence, precision
catches *badly ranked or padded* evidence. A low recall with high precision means
`top_k` or the chunker is starving the generator. Also answer-independent.

### `faithfulness` — is the answer grounded, or hallucinated?

The judge decomposes our answer into atomic statements, then checks each one for
inferability from the retrieved contexts:

```
faithfulness = (statements inferable from the contexts) / (total statements)
```

This is the hallucination metric, and it is the offline analogue of the runtime
`verify_groundedness` node — it says nothing about whether the answer is *correct*,
only whether it stays inside the retrieved evidence. An answer can be perfectly
faithful to a wrong chunk. Its threshold (0.80) is the highest we set, because for a
government-scheme bot a confident invention is the worst failure mode.

### `answer_relevancy` — does the answer actually address the question?

The judge reads only our answer (not the question) and reverse-engineers `strictness=3`
candidate questions it could be answering. Each is embedded with our E5 embedder and
compared to the real question:

```
answer_relevancy = mean( cosine(embed(generated_question_i), embed(user_input)) )
```

The prompt also flags **noncommittal** answers ("I don't know", hedges), which zero the
score. This penalises evasion, off-topic drift, and padding — but not factual error, and
it never looks at the contexts. Because it consumes embeddings, the E5 prefix wrapper
matters here as much as for `answer_similarity`.

### `answer_similarity` — how close is the answer to the reference, semantically?

Pure embedding cosine, **no LLM involved**:

```
answer_similarity = cosine( embed(answer), embed(reference) )
```

Cheap and fully deterministic, which makes it the most stable metric across runs — and
the most easily fooled: paraphrase scores high, and so does a fluent answer that
inverts a single eligibility number. Treat it as a smoke signal, never as correctness.
It carries no threshold in `evaluation.thresholds` for that reason.

### `answer_correctness` — the composite verdict

A weighted blend of a factual component and the similarity above, `weights=[0.75, 0.25]`:

```
TP = claims in both answer and reference
FP = claims in the answer but not the reference     (invented / unsupported)
FN = claims in the reference but not the answer     (omitted)

F1  = TP / (TP + 0.5·(FP + FN))                     (beta = 1.0)

answer_correctness = 0.75·F1 + 0.25·answer_similarity
```

The judge does the TP/FP/FN classification. Because F1 punishes omissions and inventions
symmetrically, this is the closest single number to "would a citizen be correctly
served?", and it is the metric most sensitive to judge quality — the same-family judge
caveat above bites hardest here. Its threshold is deliberately the lowest quality floor
(0.60): the reference answers are themselves LLM-synthesized, so exact claim overlap is
noisy.

### `answered_rate` — our own coverage metric

```
answered_rate = (rows with a non-empty answer) / (all rows)
```

Not a RAGAS metric; computed in `evaluate_testset` from the same emptiness rule the
three fallback paths produce. It exists because every metric above except the two
retrieval ones is averaged over answered rows, so quality scores alone cannot detect a
pipeline that refuses most traffic. Always read it first.

## Thresholds and the regression gate

`tests/integration/test_ragas_regression.py` asserts every entry in
`evaluation.thresholds` against a fresh English-only run (default 10 rows, override
with `RAGAS_REGRESSION_SIZE`). It carries `pytest.mark.eval` and `pyproject.toml`
sets `addopts = "-m 'not eval'"`, so it never runs in CI or a normal `pytest`
invocation — it needs a live Ollama, an ingested corpus, a generated testset, and
burns hundreds of local LLM calls. Run it deliberately.

Thresholds are floors we intend to hold, not measured baselines. Note that they are
scored against the *same* thresholds regardless of language, but the gate test only
checks `en`.

Related calibration: `rag.retrieval_threshold` was moved 0.65 → 0.55 on 2026-07-22
using this harness — a 300-row sweep of reranker sigmoid scores showed 0.65 blocked
26% of rows, and 83% of those blocked rows did contain the answer. The confidence
gate and `answered_rate` are directly coupled, so re-run `run` after touching
either threshold.

## Known limitations

- Judge and generator share a model family (see above) — faithfulness is optimistic.
- `simple` vs `reasoning` are not actually distinct synthesizers.
- `hi`/`bn` scores fold in translation quality; there is no ablation that isolates
  retrieval from translation.
- Scores are not comparable across testset regenerations — regenerate deliberately,
  and keep the old report for the diff.
