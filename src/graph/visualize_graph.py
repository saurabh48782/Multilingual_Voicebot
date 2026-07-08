from unittest.mock import MagicMock

from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph
from src.graph.deps import Deps


def main() -> None:
    deps = Deps(
        stt=MagicMock(),
        tts=MagicMock(),
        translator=MagicMock(),
        llm=MagicMock(),
        retriever=MagicMock(),
    )
    app = build_graph(checkpointer=MemorySaver(), deps=deps)

    png_bytes = app.get_graph().draw_mermaid_png()

    output_filename = "langgraph_visualization.png"

    with open(output_filename, "wb") as f:
        f.write(png_bytes)

    print(f"Graph successfully saved to {output_filename}")


if __name__ == "__main__":
    main()
