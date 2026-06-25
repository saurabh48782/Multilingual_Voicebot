# IndicConformer STT sidecar

A standalone HTTP service wrapping [`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) - a Conformer hybrid CTC+RNNT ASR model covering 22 Indian languages.

## Why a separate service?

The model loads custom modeling code via `trust_remote_code=True` and pulls an
`onnxruntime` / `torchaudio` stack we'd rather not add to the app image. Running
it here keeps those dependencies - and the remote-code execution - isolated; the
app talks to it over HTTP. `pyproject.toml` / `uv.lock` for the main app are
untouched. (Same pattern as the `tts_parler` sidecar.)

## No language auto-detection

IndicConformer does **not** detect the spoken language - the
caller must pass a language code. The app's UI mandates a language selection and
threads it through to `/stt`. Supported end to end today: Hindi (`hi`),
Bengali (`bn`).

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/healthz` | – | `{"status":"ok","model_loaded":bool}` |
| `POST` | `/stt` | multipart: `audio`=<file>, `language`=`hi`/`bn`/…, `decode_strategy`=`rnnt`/`ctc` | `{"text": str, "language": str}` |

The 600M model loads lazily on the first `/stt` request and stays resident.

## Run via docker-compose (recommended - works on any machine with Docker)

```bash
docker compose --profile all up        # brings up db + api + tts + stt
```

The `api` service is wired with `STT_REMOTE_URL=http://stt:8002`, so the app
routes transcription here automatically (see `src/stt/indic_conformer_remote.py`).

## Run standalone

```bash
docker build -t voicebot-stt ./services/stt_conformer
docker run --rm -p 8002:8002 -v "$PWD/data/hf_cache:/cache/hf" voicebot-stt
```

## CPU

The default image is built for **GPU** (`onnxruntime-gpu` + CUDA torch, not
portable everywhere). Inference is ONNX, so the CPU build must swap *both* the
ONNX runtime and the torch wheel index:

```bash
docker build \
  --build-arg ONNXRUNTIME=onnxruntime==1.20.1 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
  -t voicebot-stt-cpu ./services/stt_conformer
```

and comment out the `deploy.resources.reservations.devices` block on the `stt`
service in `docker-compose.yml`.
