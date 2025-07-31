from langgraph.graph import StateGraph, END
from frdr.fed_narratives.agents.theme_classifier.agent import calculate_surprise_index
from frdr.fed_narratives.agents.emphasis_scorer.agent import compute_emphasis_llm
from frdr.fed_narratives.agents.theme_averager.theme_averager_agent import ThemeAveragerAgent
from frdr.fed_narratives.agents.surprise_detector.surprise_detector_agent import SurpriseDetectorAgent
from langchain_core.runnables import RunnableLambda

from typing import TypedDict, Annotated, List, Dict

import pandas as pd

class PipelineState(TypedDict):
    paragraphs: Annotated[List[str], "raw_paragraphs"]
    tags: Annotated[List[Dict], "theme_tags"]  # [{'paragraph': "...", 'theme': "..."}]
    emphasis: Annotated[List[Dict], "theme_emphasis"]  # [{'paragraph': '...', 'theme': '...', 'emphasis_score': ...}]
    baseline: Annotated[List[Dict], "theme_baseline"]  # [{'theme': '...', 'baseline_score': ...}]
    surprise: Annotated[List[Dict], "surprise_index"]  # [{'paragraph': ..., 'theme': ..., 'emphasis_score': ..., 'baseline_score': ..., 'surprise_score': ..., ...}]

def create_fed_speech_pipeline(history_df: pd.DataFrame):

    builder = StateGraph(PipelineState)

    theme_classifier_node = RunnableLambda(lambda input: {
        "paragraphs": input["paragraphs"],
        "tags": calculate_surprise_index(input["paragraphs"]).tags
    })

    emphasis_scorer_node = RunnableLambda(lambda input: {
        **input,
        # expects input["tags"], returns a list of dicts (per paragraph) with emphasis_score
        "emphasis": compute_emphasis_llm(input["tags"]).emphasis_tags
    })

    # Use ThemeAveragerAgent and SurpriseDetectorAgent
    theme_averager_agent = ThemeAveragerAgent(history_df)
    theme_averager_node = RunnableLambda(lambda input: {
        **input,
        # expects input["emphasis"], adds baseline score per paragraph (list of dicts)
        "baseline": [
            {
                "paragraph": em["paragraph"],
                "theme": em["theme"],
                "emphasis_score": em["emphasis_score"],
                "baseline_score": theme_averager_agent.compute_baseline(em["theme"], input.get("date"))  # pass date if available
            }
            for em in input["emphasis"]
        ]
    })

    surprise_detector_agent = SurpriseDetectorAgent(threshold=0.1)
    surprise_detector_node = RunnableLambda(lambda input: {
        **input,
        "surprise": [
            {
                **b,
                **surprise_detector_agent.detect(b["emphasis_score"], b["baseline_score"])
            }
            for b in input["baseline"]
        ]
    })

    # Build pipeline
    builder.add_node("theme_classifier", theme_classifier_node)
    builder.add_node("emphasis_scorer", emphasis_scorer_node)
    builder.add_node("theme_averager", theme_averager_node)
    builder.add_node("surprise_detector", surprise_detector_node)

    builder.add_edge("theme_classifier", "emphasis_scorer")
    builder.add_edge("emphasis_scorer", "theme_averager")
    builder.add_edge("theme_averager", "surprise_detector")
    builder.add_edge("surprise_detector", END)

    builder.set_entry_point("theme_classifier")
    return builder.compile()

# Optionally, for visualizing the graph
def create_graph_image():
    import os
    # You must pass a real `history_df` here for a complete graph!
    history_df = pd.read_csv("fed_narratives/results_df.csv", parse_dates=["date"])
    graph = create_fed_speech_pipeline(history_df)

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
