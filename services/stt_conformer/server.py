"""IndicConformer STT sidecar - service entrypoint.

Standalone image; the main app talks to it over HTTP
(src/stt/indic_conformer_remote.py).

API:
  GET  /healthz   → {"status": "ok", "model_loaded": bool}
  POST /stt       → {"text": str, "language": str}
                    multipart: audio=<file>, language=<code>, decode_strategy=rnnt|ctc
"""

from __future__ import annotations

import uvicorn

from api.app import create_app

app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
