from pydantic import BaseModel, Field
from typing import List, Literal

class ThemeTag(BaseModel):
    paragraph: str
    theme: Literal['inflation', 'employment', 'other']

class EmphasisTag(BaseModel):
    paragraph: str
    theme: Literal['inflation', 'employment', 'other']
    emphasis_score: float  # value between 0 and 1

class EmphasisScoreOutput(BaseModel):
    emphasis_tags: List[EmphasisTag]

def compute_emphasis_llm(tags: List[ThemeTag], llm=None) -> EmphasisScoreOutput:
    # (Pseudo-LLM step, replace with real LLM batch call)
    emphasis_tags = []
    for tag in tags:
        # For demo, assign a dummy value or query LLM with paragraph content
        # e.g., prompt: "Rate the emphasis of this paragraph on {tag.theme} from 0 to 1: {tag.paragraph}"
        score = 0.5  # Replace with LLM call result
        emphasis_tags.append(
            EmphasisTag(paragraph=tag.paragraph, theme=tag.theme, emphasis_score=score)
        )
    return EmphasisScoreOutput(emphasis_tags=emphasis_tags)
