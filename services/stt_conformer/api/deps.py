"""FastAPI dependencies for the STT sidecar.

The IndicConformer ``Transcriber`` is built once in the app lifespan and stashed
on ``app.state``; each request pulls it via the dependency callable here.
"""

from __future__ import annotations

from fastapi import Depends, Request

from api.transcriber import Transcriber


def get_transcriber(request: Request) -> Transcriber:
    transcriber: Transcriber = request.app.state.transcriber
    return transcriber


TranscriberDep = Depends(get_transcriber)
