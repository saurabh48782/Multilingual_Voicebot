"""GET /healthcheck - quick liveness + readiness check."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas import HealthStatus
from src.utils.observability import tracing_enabled

router = APIRouter(tags=["System Status"])


@router.get("/healthcheck", response_model=HealthStatus)
def healthcheck(request: Request) -> HealthStatus:
    store = getattr(request.app.state, "faiss_store", None)
    total = store.total_chunks if store is not None else 0
    return HealthStatus(
        status="ok",
        faiss_loaded=store is not None,
        total_chunks=total,
        tracing_enabled=tracing_enabled(),
    )
