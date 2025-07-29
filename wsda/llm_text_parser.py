from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import re
import json
import os

# Define the Pydantic model for the expected output structure
class CottonAnalysis(BaseModel):
    report_date: str = Field(description="The date the report was published, in a clear format (e.g., January 12, 2025).")
    report_year: str = Field(description="The year the report was published (e.g., 2025).")
    output: str = Field(description="Total cotton output in million bales.")
    supply: str = Field(description="Total cotton supply in million bales.")
    trade: str = Field(description="Total cotton trade in million bales.")
    use: str = Field(description="Total cotton use in million bales.")
    ending_stocks: str = Field(description="Total cotton ending stocks in million bales.")
    trend: str = Field(description="Summary of increase/decrease trends mentioned in the report for cotton.")
    sentiment: str = Field(description="Overall sentiment for cotton (Bullish, Bearish, or Neutral).")

# Initialize the LLM
llm = ChatOllama(model='llama3.1', temperature=0, format="json")

# --- 1. Load XLS and extract text (mock function since XLS parsing isn't implemented) ---
# Note: Since PyMuPDF (fitz) is for PDFs, we'll assume the input is the provided text document for now.
# In a real scenario, use a library like pandas or openpyxl to parse XLS.
def extract_all_text_from_xls(xls_path: str) -> str:
    # For this implementation, we'll use the provided document text directly
    # In practice, replace this with actual XLS parsing logic
    return """{document_text}""".format(document_text=open(xls_path, 'r').read() if os.path.exists(xls_path) else "Error: XLS file not found")

# --- 2. Create the Pydantic output parser ---
from langchain_core.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=CottonAnalysis)

# --- 3. Define Prompt Template ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "You are a financial analyst. Extract the requested information into a JSON object. "
            "Only output the JSON object."
        )),
        ("human", (
            "Document:\n---\n{document}\n---\n"
            "{format_instructions}"
        )),
    ]
)

# --- 4. Create the chain ---
chain = prompt_template | llm | parser

# --- 5. Run on your WASDE file ---
if __name__ == "__main__":
    try:
        # Path to the WASDE XLS file
        xls_path = r'/Users/pankajti/dev/git/icaif2025/data/downloaded_reports/2025/wasde0125_XLS.xls'

        # Extract the full text of the document
        full_document_text = extract_all_text_from_xls(xls_path)

        # Pass the full document text and format instructions to the invoke method
        structured_analysis = chain.invoke({"document": full_document_text, "format_instructions": parser.get_format_instructions()})

        print("\n📘 Structured Cotton Report Analysis:\n")
        # Convert the Pydantic object to a dictionary and then pretty-print as JSON
        print(json.dumps(structured_analysis.dict(), indent=2))

    except FileNotFoundError:
        print(f"Error: The file '{xls_path}' was not found. Please ensure the path is correct.")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
        raise e
    except Exception as e:
        if "Invalid json output" in str(e):
            print(f"Output Parsing Error: The LLM did not provide valid JSON. This usually means the model's response did not conform to the expected format.")
            print(f"LLM's raw output (if available): {e.llm_output if hasattr(e, 'llm_output') else 'N/A'}")
        else:
            print(f"An unexpected error occurred: {e}")
        raise e