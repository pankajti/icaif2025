from langgraph.graph import StateGraph, END
from frdr.fed_narratives.agents.theme_classifier.agent import classify_theme
from frdr.fed_narratives.agents.emphasis_scorer.agent import compute_emphasis
from langchain_core.runnables import RunnableLambda

from typing import TypedDict, Annotated, List, Dict

class PipelineState(TypedDict):
    paragraphs: Annotated[List[str], "raw_paragraphs"]
    tags: Annotated[List[Dict], "theme_tags"]  # e.g., [{"paragraph": "...", "theme": "inflation"}]
    emphasis: Annotated[Dict[str, float], "theme_emphasis"]


# -----------------------------
# Graph Definition
# -----------------------------
def create_fed_speech_pipeline():
    builder = StateGraph(PipelineState)

    theme_classifier_node = RunnableLambda(lambda input: classify_theme(input["paragraphs"]))
    # LangGraph-compatible node
    emphasis_scorer_node = RunnableLambda(
        lambda input: compute_emphasis(input["tags"])
    )

    builder.add_node("theme_classifier", theme_classifier_node)
    builder.add_edge("theme_classifier", "emphasis_scorer")

    builder.add_node("emphasis_scorer", emphasis_scorer_node)
    builder.add_edge("emphasis_scorer", END)

    builder.set_entry_point("theme_classifier")

    return builder.compile()

def create_graph_image():
    import os
    graph = create_fed_speech_pipeline()

    def export_graph_image(graph_obj, output_path: str = "assets/graph_structure.png"):
        os.makedirs("assets", exist_ok=True)
        if os.path.exists(output_path):
            print(f"[INFO] Graph image already exists at {output_path}. Skipping regeneration.")
            return
        try:
            g = graph_obj.get_graph()
            g.draw_png(output_path)
            print(f"[INFO] Graph image saved at {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to export graph: {e}")

    export_graph_image(graph)

if __name__ == '__main__':
    create_graph_image()
