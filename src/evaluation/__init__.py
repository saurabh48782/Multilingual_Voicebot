"""RAGAS-based offline evaluation for the RAG pipeline."""

from __future__ import annotations

import sys as _sys
import types as _types

_VERTEX_MOD = "langchain_community.chat_models.vertexai"
if _VERTEX_MOD not in _sys.modules:
    try:
        __import__(_VERTEX_MOD)
    except ModuleNotFoundError:
        _stub = _types.ModuleType(_VERTEX_MOD)

        class ChatVertexAI:  # noqa: N801 - matches upstream name
            """Placeholder for the removed langchain_community Vertex AI chat model.

            Never used: the voicebot evaluation runs entirely on Ollama. Present
            only so ragas' top-level import succeeds.
            """

        _stub.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
        _sys.modules[_VERTEX_MOD] = _stub

del _sys, _types, _VERTEX_MOD
