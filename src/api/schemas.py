"""Pydantic request/response schemas for the public API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.translation.base import SUPPORTED_LANGUAGES


class ContextChunk(BaseModel):  # type: ignore[misc]
    text: str
    score: float
    source: str
    page_num: int


class ConfidenceReport(BaseModel):  # type: ignore[misc]
    retrieval_score: float
    retrieval_gap: float
    retrieval_passed: bool
    grounded: bool
    overall: Literal["green", "yellow", "red"]


class VoiceResponse(BaseModel):  # type: ignore[misc]
    session_id: str
    transcript: str
    detected_language: str
    english_query: str
    rewritten_query: str
    retrieved_context: list[ContextChunk]
    response_text: str  # vernacular
    response_text_english: str
    audio_url: str | None
    audio_content_type: str | None = None
    confidence: ConfidenceReport
    fallback_triggered: bool
    fallback_reason: str | None = None


class ChatRequest(BaseModel):  # type: ignore[misc]
    text: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=128)
    language: str = Field(..., min_length=1)

    @field_validator("language")  # type: ignore[misc]
    @classmethod
    def language_must_be_supported(cls, v: str) -> str:
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language must be one of {sorted(SUPPORTED_LANGUAGES)}")
        return v


class SessionHistoryEntry(BaseModel):  # type: ignore[misc]
    role: Literal["user", "assistant"]
    content: str


class SessionHistory(BaseModel):  # type: ignore[misc]
    session_id: str
    messages: list[SessionHistoryEntry]


class SessionMeta(BaseModel):  # type: ignore[misc]
    session_id: str
    title: str
    created_at: datetime
    last_active: datetime
    message_count: int


class SessionList(BaseModel):  # type: ignore[misc]
    sessions: list[SessionMeta]


class HealthStatus(BaseModel):  # type: ignore[misc]
    status: str
    faiss_loaded: bool
    total_chunks: int
    tracing_enabled: bool = False


class DocumentUploadResponse(BaseModel):  # type: ignore[misc]
    filename: str
    chunks_added: int
    message: str
