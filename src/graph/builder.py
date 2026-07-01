"""Compile the voicebot StateGraph.

Flow:

  START
    → transcribe
    → pii_scrub
    → translate_to_english
    → rewrite_query
    → retrieve
    → (confidence pass) → generate
                            → verify_groundedness
                                → (grounded)   → translate_to_vernacular → synthesize → [summarize?] → END
                                → (ungrounded) → fallback                → synthesize → [summarize?] → END
    → (confidence fail) → fallback             → synthesize → [summarize?] → END

`summarize` fires after `synthesize` when accumulated messages >= SUMMARIZE_THRESHOLD (8).
It compresses the oldest messages into `conversation_summary` and removes them from state,
keeping the checkpoint bounded while giving `rewrite_query` full conversation context.

`build_graph(checkpointer, deps)` returns a compiled `StateGraph`. A
`checkpointer` is required — use `async_postgres_checkpointer()` for
production or pass a `MemorySaver` explicitly in tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from src.graph.deps import Deps
from src.graph.nodes.fallback import fallback
from src.graph.nodes.generate import make_generate
from src.graph.nodes.pii_scrub import pii_scrub
from src.graph.nodes.retrieve import make_retrieve, route_after_retrieve
from src.graph.nodes.rewrite_query import make_rewrite_query
from src.graph.nodes.summarize import make_summarize, should_summarize
from src.graph.nodes.synthesize import make_synthesize
from src.graph.nodes.transcribe import make_transcribe
from src.graph.nodes.translate import make_to_english, make_to_vernacular
from src.graph.nodes.verify_groundedness import (
    make_verify_groundedness,
    route_after_verify,
)
from src.graph.state import VoicebotState
from src.utils.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def build_graph(
    checkpointer: BaseCheckpointSaver[Any],
    deps: Deps | None = None,
) -> Any:
    """Wire the StateGraph and compile it with the provided checkpointer."""
    deps = deps or Deps()

    graph: Any = StateGraph(VoicebotState)

    graph.add_node("transcribe", make_transcribe(deps))
    graph.add_node("pii_scrub", pii_scrub)
    graph.add_node("translate_to_english", make_to_english(deps))
    graph.add_node("rewrite_query", make_rewrite_query(deps))
    graph.add_node("retrieve", make_retrieve(deps))
    graph.add_node("generate", make_generate(deps))
    graph.add_node("verify_groundedness", make_verify_groundedness(deps))
    graph.add_node("translate_to_vernacular", make_to_vernacular(deps))
    graph.add_node("fallback", fallback)
    graph.add_node("synthesize", make_synthesize(deps))
    graph.add_node("summarize", make_summarize(deps))

    graph.add_edge(START, "transcribe")
    graph.add_edge("transcribe", "pii_scrub")
    graph.add_edge("pii_scrub", "translate_to_english")
    graph.add_edge("translate_to_english", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "fallback": "fallback"},
    )
    graph.add_edge("generate", "verify_groundedness")
    graph.add_conditional_edges(
        "verify_groundedness",
        route_after_verify,
        {
            "translate_to_vernacular": "translate_to_vernacular",
            "fallback": "fallback",
        },
    )
    graph.add_edge("translate_to_vernacular", "synthesize")
    graph.add_edge("fallback", "synthesize")
    graph.add_conditional_edges(
        "synthesize",
        should_summarize,
        {"summarize": "summarize", "end": END},
    )
    graph.add_edge("summarize", END)

    return graph.compile(checkpointer=checkpointer)
