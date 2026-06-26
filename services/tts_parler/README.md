# Indic Parler-TTS sidecar

A standalone HTTP service wrapping [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts).

## Why a separate service?

`parler-tts` hard-pins `transformers==4.46.1`, which conflicts with the main
app's stack (e5-large + bge-reranker + sentence-transformers want a newer
transformers). Two versions of `transformers` can't be imported into one
process, so this model runs in its **own environment** and the app talks to it
over HTTP. The conflicting pin lives only in this image - `pyproject.toml` /
`uv.lock` for the main app are untouched.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/healthz` | – | `{"status":"ok","model_loaded":bool}` |
| `POST` | `/tts` | `{"text": str, "description": str?}` | `audio/wav` bytes |

The 0.9B model loads lazily on the first `/tts` request and stays resident.

## Run via docker-compose (recommended - works on any machine with Docker)

```bash
docker compose --profile all up        # brings up db + api + tts
```

The `api` service is wired with `TTS_REMOTE_URL=http://tts:8001`, so the app
routes synthesis here automatically (see `src/tts/indic_parler_remote.py`).

## Run standalone

```bash
docker build -t voicebot-tts ./services/tts_parler
docker run --rm -p 8001:8001 -v "$PWD/data/hf_cache:/cache/hf" voicebot-tts
```

## CPU

The default image is built for **GPU** (CUDA torch). For a portable CPU build:

```bash
docker build \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
  -t voicebot-tts-cpu ./services/tts_parler
```

and comment out the `deploy.resources.reservations.devices` block on the `tts`
service in `docker-compose.yml`. GPU additionally requires the host NVIDIA
container runtime - that part can't be containerized away.
