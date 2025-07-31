from langchain_ollama.chat_models import ChatOllama
from langchain_together import Together
import os
from langchain_openai.chat_models import ChatOpenAI

from dotenv import load_dotenv

env_path = '/Users/pankajti/dev/git/gloomy-dreams/gloomy_dreams/config/.env'
if os.path.exists(env_path):
    load_dotenv(env_path)

together_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
llama_model_name = 'llama3.1'
openai_model_name = "gpt-3.5-turbo"
#openai_model_name = "gpt-4.1"



models = {'together':Together(model=together_model_name), 'llama3_local':ChatOllama(model=llama_model_name),
          'openai':ChatOpenAI(model_name=openai_model_name)}


def get_llm(model_name: str = "together"):
    """Returns a Together model if available, else falls back to Ollama local model."""

    try:
        llm_provider = os.getenv("LLM_PROVIDER", model_name).lower()
        print(f"loading llm for {llm_provider}")

        model = models.get(llm_provider,None)

        if model is None:
            raise ValueError(f"Unknown LLM_PROVIDER: {llm_provider}")

        return model

    except Exception as e:
        print(f"[WARNING] Failed to load Together model: {e}")
        print("[INFO] Falling back to local Ollama model...")
        model = ChatOllama(model=model_name)
        return model


if __name__ == '__main__':
    from dotenv import load_dotenv

    env_path = '/Users/pankajti/dev/git/gloomy-dreams/gloomy_dreams/config/.env'
    if os.path.exists(env_path):
        load_dotenv(env_path)

    llm = get_llm('openai')
    resp = llm.invoke("What is your name?")
    print(resp)