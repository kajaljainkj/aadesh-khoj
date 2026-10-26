"""Central settings, read from .env."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # LLM: ollama (local, default) | groq | gemini | deepseek | grok
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    # If the primary provider fails (no key, quota/rate limit, network), use this one instead
    llm_fallback: str = os.getenv("LLM_FALLBACK", "ollama").lower()
    llm_fallback_cooldown: float = float(os.getenv("LLM_FALLBACK_COOLDOWN", "60"))
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    xai_api_key: str = os.getenv("XAI_API_KEY", "")
    xai_model: str = os.getenv("XAI_MODEL", "grok-3-mini")
    llm_min_interval: float = float(os.getenv("LLM_MIN_INTERVAL", "4"))

    # Embeddings (local)
    embed_model: str = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))

    database_url: str = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:55432/rag")

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    collection: str = os.getenv("COLLECTION", "main")

    # Reranker: a cross-encoder re-scores the top RERANK_CANDIDATES hits (local, ~2 GB model)
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "false").lower() == "true"
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "20"))
    # Max tokens the cross-encoder reads per (question, chunk) pair; lower = faster on CPU
    rerank_max_length: int = int(os.getenv("RERANK_MAX_LENGTH", "512"))
    # If no candidate scores above this, the reranker is unsure: keep the first-stage order
    rerank_min_score: float = float(os.getenv("RERANK_MIN_SCORE", "0.05"))

    # OCR (Windows: set TESSERACT_CMD to tesseract.exe if it's not on PATH)
    ocr_enabled: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
    ocr_lang: str = os.getenv("OCR_LANG", "eng+hin")


settings = Settings()
