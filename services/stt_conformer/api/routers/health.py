"""GET /healthz - liveness + lazy-model readiness flag.

The container reports healthy immediately; ``model_loaded`` flips to true only
after the first /stt request warms the resident model, so callers can poll it
before sending latency-sensitive traffic.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.deps import TranscriberDep
from api.schemas import HealthStatus
from api.transcriber import Transcriber

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthStatus)
def healthz(transcriber: Transcriber = TranscriberDep) -> HealthStatus:
    return HealthStatus(status="ok", model_loaded=transcriber.loaded)
