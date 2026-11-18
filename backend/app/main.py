from contextlib import asynccontextmanager
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import client
from .routes import router
from .rag import prepare_reranker
from .document_jobs import start_cleanup


@asynccontextmanager
async def lifespan(app):
    start_cleanup()
    if settings.rerank:
        threading.Thread(target=prepare_reranker, name="research-reranker", daemon=True).start()
    yield
    client.close()


app = FastAPI(title="Research RAG", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.origins),
    allow_credentials=False, allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"])
app.include_router(router)


@app.head("/")
@app.get("/")
@app.get("/api/health")
def health():
    return {"status": "ok", "app": "Research RAG", "supabase_configured": bool(settings.supabase_url and settings.supabase_key),
        "generation_configured": bool(settings.groq_key),
        "embedding_provider": "qdrant-cloud" if settings.qdrant_all_minilm_l6_v2 else "local-bge",
        "rerank_enabled": settings.rerank}

