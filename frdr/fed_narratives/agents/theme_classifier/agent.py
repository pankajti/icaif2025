from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from frdr.fed_narratives.utilities.llm import get_llm  # assumes you have a shared LLM loader

# Define output schema
class ThemeTag(BaseModel):
    paragraph: str = Field(..., description="Paragraph of text")
    theme: str = Field(..., description="One of: 'inflation', 'employment', or 'other'")

class ThemeClassificationOutput(BaseModel):
    tags: List[ThemeTag]

# LangChain parser
parser = PydanticOutputParser(pydantic_object=ThemeClassificationOutput)

# Prompt template
prompt_template = """
You are a Federal Reserve policy analyst.

Classify the primary theme of each paragraph below as either:
- "inflation"
- "employment"
- or "other"

Do not skip any paragraph. If unsure, always assign "other".

Return your output in this JSON format:
{format_instructions}

Paragraphs:
{paragraphs}
"""


# Agent logic
def classify_theme(paragraphs: List[str]) -> ThemeClassificationOutput:
    llm = get_llm()
    para = "\n\n".join(paragraphs)
    response = call_llm(llm, para)
    response_text = response.content if not isinstance(response,str) else response
    try:
        classified_theme = parser.parse(response_text)
    except Exception as e:
        try:
            response = call_llm(llm, response_text)
            response_text = response.content if not isinstance(response, str) else response
            classified_theme = parser.parse(response_text)
        except Exception as e :
            print("Error parsing response {} ")
            #raise e

    return classified_theme


def call_llm(llm, para):
    formatted = parser.get_format_instructions()
    full_prompt = prompt_template.format(
        format_instructions=formatted,
        paragraphs=para
    )
    response = llm.invoke(full_prompt)
    return response

