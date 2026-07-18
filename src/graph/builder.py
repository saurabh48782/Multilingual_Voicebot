"""Compile the voicebot StateGraph.

Flow:

  START
    → transcribe
    → pii_scrub
    → translate_to_english
    → rewrite_query
    → classify_intent
    → (general) → smalltalk → translate_to_vernacular → synthesize → END
    → (scheme)  → retrieve
        → (confidence pass) → generate → verify_groundedness
            → (grounded)   → translate_to_vernacular → synthesize → END
            → (ungrounded) → fallback                → synthesize → END
        → (confidence fail) → fallback → synthesize → END

History compression (the`summarize` node) is deliberately NOT in this graph:
It now runs off the critical path as a FastAPI BackgroundTask after the response
is returned - see `src/api/services.py :: summarize_session`.

`build_graph(checkpointer, deps)` returns a compiled `StateGraph`. A
`checkpointer` is required - production opens one via `checkpointer_lifespan()`
in `src/db/db.py`; tests pass a `MemorySaver` explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from src.graph.deps import Deps
from src.graph.node_logging import traced_node, traced_router
from src.graph.nodes.classify_intent import make_classify_intent, route_after_classify
from src.graph.nodes.fallback import fallback
from src.graph.nodes.generate import make_generate
from src.graph.nodes.pii_scrub import pii_scrub
from src.graph.nodes.retrieve import make_retrieve, route_after_retrieve
from src.graph.nodes.rewrite_query import make_rewrite_query
from src.graph.nodes.smalltalk import make_smalltalk
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

    def add(name: str, fn: Any) -> None:
        """Register a node wrapped in the per-node stage tracer."""
        graph.add_node(name, traced_node(name, fn))

    add("transcribe", make_transcribe(deps))
    add("pii_scrub", pii_scrub)
    add("translate_to_english", make_to_english(deps))
    add("rewrite_query", make_rewrite_query(deps))
    add("classify_intent", make_classify_intent(deps))
    add("smalltalk", make_smalltalk(deps))
    add("retrieve", make_retrieve(deps))
    add("generate", make_generate(deps))
    add("verify_groundedness", make_verify_groundedness(deps))
    add("translate_to_vernacular", make_to_vernacular(deps))
    add("fallback", fallback)
    add("synthesize", make_synthesize(deps))

    graph.add_edge(START, "transcribe")
    graph.add_edge("transcribe", "pii_scrub")
    graph.add_edge("pii_scrub", "translate_to_english")
    graph.add_edge("translate_to_english", "rewrite_query")
    graph.add_edge("rewrite_query", "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        traced_router("classify_intent", route_after_classify),
        {"smalltalk": "smalltalk", "retrieve": "retrieve"},
    )
    graph.add_edge("smalltalk", "translate_to_vernacular")

    graph.add_conditional_edges(
        "retrieve",
        traced_router("retrieve", route_after_retrieve),
        {"generate": "generate", "fallback": "fallback"},
    )
    graph.add_edge("generate", "verify_groundedness")
    graph.add_conditional_edges(
        "verify_groundedness",
        traced_router("verify_groundedness", route_after_verify),
        {
            "translate_to_vernacular": "translate_to_vernacular",
            "fallback": "fallback",
        },
    )
    graph.add_edge("translate_to_vernacular", "synthesize")
    graph.add_edge("fallback", "synthesize")
    # synthesize is terminal for the request path. History compression runs
    # off-critical-path (see module docstring / src/api/services.py).
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer)
