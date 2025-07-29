from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from frdr.fed_narratives.utilities.llm import get_llm

# Output schema for surprise calculation
class SurpriseScore(BaseModel):
    paragraph: str = Field(..., description="Paragraph of text")
    theme: str = Field(..., description="Thematic classification, e.g., 'inflation', 'employment', or 'other'")
    emphasis_score: float = Field(..., description="Emphasis score for this paragraph on a 0-1 scale")
    baseline_score: float = Field(..., description="Rolling mean emphasis score for this theme")
    surprise_score: float = Field(..., description="Absolute difference between emphasis_score and baseline_score")
    direction: str = Field(..., description="'upward' or 'downward' depending on whether emphasis_score > baseline_score")
    significant: bool = Field(..., description="True if surprise_score > threshold")

class SurpriseIndexOutput(BaseModel):
    scores: List[SurpriseScore]

parser = PydanticOutputParser(pydantic_object=SurpriseIndexOutput)

prompt_template = """
You are a Federal Reserve policy analyst.

For each paragraph below:
- Classify its primary theme ('inflation', 'employment', or 'other').
- Assign an 'emphasis_score' from 0 (not emphasized) to 1 (very strongly emphasized) for the assigned theme.
- The 'baseline_score' for each theme is provided for each paragraph.
- Compute 'surprise_score' = abs(emphasis_score - baseline_score).
- Set 'direction' as 'upward' if emphasis_score > baseline_score, else 'downward'.
- If surprise_score > 0.1, set 'significant' = true; else false.

Return your output as JSON:
{format_instructions}

Input:
Paragraphs: {paragraphs}
Baselines: {baselines}
"""

def calculate_surprise_index(paragraphs: List[str], theme_baselines: List[float]) -> SurpriseIndexOutput:
    llm = get_llm('llama3_local')
    para = "\n\n".join(paragraphs)
    baseline_str = ", ".join([str(b) for b in theme_baselines])

    response = call_llm(llm, para, baseline_str)
    response_text = response.content if not isinstance(response, str) else response
    try:
        result = parser.parse(response_text)
    except Exception as e:
        try:
            response = call_llm(llm, response_text, baseline_str)
            response_text = response.content if not isinstance(response, str) else response
            result = parser.parse(response_text)
        except Exception as e:
            print("Error parsing response for surprise index.")
            result = None
    return result

def call_llm(llm, para, baselines):
    formatted = parser.get_format_instructions()
    full_prompt = prompt_template.format(
        format_instructions=formatted,
        paragraphs=para,
        baselines=baselines
    )
    response = llm.invoke(full_prompt)
    return response
