import os
import logging
from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv

log = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def create_llm(model_name: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any) -> BaseChatModel:
        load_dotenv()
        # Strictly Groq as requested
        from langchain_groq import ChatGroq
        key = api_key or os.getenv("GROQ_API_KEY")
        
        if not key:
            log.warning("LLMFactory: GROQ_API_KEY not found.")
            raise ValueError("GROQ_API_KEY is missing. Please set it in .env or your environment.")
        
            
        model = model_name or os.getenv("CARTOGRAPHER_MODEL") or "llama-3.3-70b-versatile"
        return ChatGroq(model=model, groq_api_key=key, **kwargs)

    @staticmethod
    def create_embeddings() -> Optional[Any]:
        # Embeddings are disabled in strict Groq mode (no native API)
        log.warning("LLMFactory: Embeddings are disabled (Groq-only mode). Semantic features will be skiped.")
        return None
