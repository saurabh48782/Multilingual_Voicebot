"""LangGraph state schema for the voicebot pipeline."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.rag.store import SearchResult


class VoicebotState(TypedDict, total=False):
    """Full state mutated by every node in the graph.

    `total=False` so individual nodes can return partial updates that LangGraph
    merges into the running state.
    """

    # inputs
    audio_input: bytes | None
    text_input: str | None
    requested_language: str  # hi | bn | en

    # transcription / language
    transcript: str
    source_language: str  # "hi" | "bn" | "en"

    # translation / rewrite
    english_query: str
    rewritten_query: str

    # retrieval
    retrieved_docs: list[SearchResult]
    retrieval_confidence: float
    retrieval_gap: float
    retrieval_passed: bool

    # generation
    english_response: str
    grounded: bool

    # output
    vernacular_response: str
    audio_output: bytes | None
    audio_content_type: str

    # routing flags
    fallback_triggered: bool
    fallback_reason: str

    # conversation memory
    messages: Annotated[list[BaseMessage], add_messages]
    # rolling LLM summary of turns older than KEEP_RECENT
    conversation_summary: str
