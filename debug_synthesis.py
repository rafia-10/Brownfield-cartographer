
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

def test_synthesis():
    model = "gemini-pro"
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    
    prompt = "You are an Expert Software Architect. Summarize the importance of code modularity in 2 sentences."
    
    try:
        print(f"Calling LLM ({model})...")
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"SUCCESS: {response.content}")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test_synthesis()
