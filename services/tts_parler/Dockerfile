# Indic Parler-TTS sidecar image.
# GPU by default; build a portable CPU image with:
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
# GPU also needs the host NVIDIA container runtime at run time.
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121

# git: parler-tts (+ descript-audiotools) install from git. libsndfile1: soundfile.
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
RUN pip install --extra-index-url "${TORCH_INDEX}" -r requirements.txt
COPY server.py /srv/

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8001/healthz || exit 1
CMD ["python", "server.py"]
