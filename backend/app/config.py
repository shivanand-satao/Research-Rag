"""Load backend secrets without exposing them to the browser."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
PRODUCTION = os.getenv("APP_ENV", "").strip().lower() == "production" or os.getenv("RENDER") == "true"
load_dotenv(BACKEND / (".env.production" if PRODUCTION else ".env"))

QDRANT_MINILM_MODEL = "sentence-transformers/all-minilm-l6-v2"
FRONTEND_URL = (os.getenv("FRONTEND_URL") or "http://127.0.0.1:5173").rstrip("/")


@dataclass(frozen=True)
class Settings:
    frontend_url: str = FRONTEND_URL
    supabase_url: str = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_key: str = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY", "")
    redirect_url: str = os.getenv("RESEARCH_REDIRECT_URL") or f"{FRONTEND_URL}/auth/callback"
    storage_bucket: str = os.getenv("RESEARCH_STORAGE_BUCKET", "research-documents")
    qdrant_url: str = os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLUSTER_ENDPOINT", "")
    qdrant_key: str = os.getenv("QDRANT_API_KEY", "")
    collection: str = os.getenv("RESEARCH_QDRANT_COLLECTION", "research_rag_chunks")
    groq_key: str = os.getenv("GROQ_API_KEY", "")
    groq_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    generation_model: str = os.getenv("GENERATION_MODEL", "openai/gpt-oss-120b")
    answer_input_tokens: int = max(512, int(os.getenv("RESEARCH_ANSWER_INPUT_TOKENS", "3500")))
    answer_max_tokens: int = max(256, int(os.getenv("RESEARCH_ANSWER_MAX_TOKENS", "1400")))
    generation_retries: int = max(0, min(3, int(os.getenv("RESEARCH_GENERATION_RETRIES", "2"))))
    generation_retry_wait: float = max(0, min(10, float(os.getenv("RESEARCH_GENERATION_RETRY_WAIT", "8"))))
    ocr: bool = os.getenv("OCR", "false").lower() == "true"
    ocr_language: str = os.getenv("OCR_LANGUAGE", "eng")
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")
    rerank: bool = os.getenv("RESEARCH_RERANK_ENABLED", "false").lower() == "true"
    rerank_model: str = os.getenv("RESEARCH_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    child_tokens: int = int(os.getenv("RESEARCH_CHILD_TOKENS", "360"))
    parent_tokens: int = int(os.getenv("RESEARCH_PARENT_TOKENS", "1600"))
    context_tokens: int = int(os.getenv("RESEARCH_CONTEXT_TOKENS", "12000"))
    origins: tuple[str, ...] = tuple(dict.fromkeys([FRONTEND_URL, "http://127.0.0.1:5173", "http://localhost:5173",
        *(x.strip().rstrip("/") for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip())]))
    max_upload_bytes: int = 25 * 1024 * 1024

    @property
    def embedding_model(self) -> str:
        return QDRANT_MINILM_MODEL

    @property
    def embedding_child_tokens(self) -> int:
        # Hosted MiniLM accepts 256 tokens, including [CLS] and [SEP].
        return min(self.child_tokens, 254)


settings = Settings()

