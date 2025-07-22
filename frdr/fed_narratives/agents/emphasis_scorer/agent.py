from pydantic import BaseModel, Field
from typing import List, Literal, Dict

# Input from ThemeClassifierAgent
class ThemeTag(BaseModel):
    paragraph: str
    theme: Literal['inflation', 'employment', 'other']

class EmphasisScoreOutput(BaseModel):
    emphasis: Dict[str, float]  # e.g., {"inflation": 0.6, "employment": 0.3, "other": 0.1}

def compute_emphasis(tags: List[ThemeTag]) -> EmphasisScoreOutput:
    total = len(tags)
    if total == 0:
        return EmphasisScoreOutput(emphasis={"inflation": 0.0, "employment": 0.0, "other": 0.0})

    counts = {"inflation": 0, "employment": 0, "other": 0}
    for tag in tags:
        counts[tag.theme] += 1

    emphasis = {k: round(v / total, 3) for k, v in counts.items()}
    return EmphasisScoreOutput(emphasis=emphasis)

