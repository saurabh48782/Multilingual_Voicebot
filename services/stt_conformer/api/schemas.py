"""Pydantic response schemas for the STT sidecar."""

from __future__ import annotations

from pydantic import BaseModel


class HealthStatus(BaseModel):
    status: str
    model_loaded: bool


class TranscriptionResponse(BaseModel):
    text: str
    language: str
