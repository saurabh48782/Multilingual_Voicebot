# IndicConformer STT sidecar image.
# GPU by default; build a portable CPU image with:
#   --build-arg ONNXRUNTIME=onnxruntime==1.20.1
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
# GPU also needs the host NVIDIA container runtime at run time.
# Python 3.11 for the widest onnxruntime/torchaudio wheel coverage.
FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG ONNXRUNTIME=onnxruntime-gpu==1.20.1

# ffmpeg: torchaudio decodes compressed uploads (webm/opus, mp3) via it.
# libsndfile1: soundfile. git: trust_remote_code may fetch modeling deps.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libsndfile1 \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/cache/hf \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv
COPY requirements.txt /srv/
RUN pip install --extra-index-url "${TORCH_INDEX}" -r requirements.txt "${ONNXRUNTIME}"
COPY api /srv/api
COPY server.py /srv/

EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8002/healthz || exit 1
CMD ["python", "server.py"]
