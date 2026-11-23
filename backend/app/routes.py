"""Authenticated research workspace endpoints."""
import logging
import threading
import weakref
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from .config import settings
from .database import Database, Identity, authenticate
from .processing import MIME_TYPES
from .preview import source_pdf
from .rag import answer_question
from .schemas import ChatRequest, ConversationCreate, ConversationUpdate
from .vectors import vector_store
from .document_jobs import list_documents, schedule_index, schedule_delete, file_lock

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)
_locks = weakref.WeakValueDictionary()
_registry_lock = threading.Lock()


def conversation_lock(key: str):
    with _registry_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


@router.get("/config")
def config():
    # Only the publishable/anon key is public. Service and provider keys stay on the server.
    key = settings.supabase_key
    if key.startswith("sb_secret_"):
        raise HTTPException(503, "Use a Supabase publishable or anon key for SUPABASE_KEY.")
    if key.count(".") == 2:
        import base64
        import json
        try:
            role = json.loads(base64.urlsafe_b64decode(key.split(".")[1] + "==")).get("role")
            if role == "service_role":
                raise HTTPException(503, "Use a Supabase publishable or anon key for SUPABASE_KEY.")
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(503, "The Supabase public key is invalid.")
    if not settings.supabase_url or not key:
        raise HTTPException(503, "Configure Supabase in backend/.env before starting Research RAG.")
    return {"supabase_url": settings.supabase_url, "supabase_key": key, "redirect_url": settings.redirect_url}


@router.get("/documents")
def documents(user: Identity = Depends(authenticate)):
    return list_documents(Database(user))


@router.post("/documents", status_code=202)
def upload(file: UploadFile = File(...), user: Identity = Depends(authenticate)):
    filename = (file.filename or "").replace("\\", "/").split("/")[-1][:180]
    suffix = Path(filename).suffix.lower()
    if suffix not in MIME_TYPES:
        raise HTTPException(400, "Upload a PDF, DOCX, TXT, or Markdown file.")
    content = file.file.read(settings.max_upload_bytes + 1)
    file.file.close()
    if not content or len(content) > settings.max_upload_bytes:
        raise HTTPException(413, "Choose a nonempty file smaller than 25 MB.")
    db = Database(user)
    document_id = str(uuid4())
    path = f"{user.id}/{document_id}/source{suffix}"
    row = db.insert("documents", {"id": document_id, "title": filename, "source": filename,
        "mime_type": MIME_TYPES[suffix], "status": "processing", "storage_path": path, "size_bytes": len(content)})
    try:
        return schedule_index(db, row, content)
    except Exception:
        db.update("documents", row["id"], {"status": "failed", "error": "Could not queue this file. Retry the upload."})
        raise


@router.get("/documents/{document_id}/content")
def download(document_id: UUID, user: Identity = Depends(authenticate)):
    db = Database(user)
    document = db.get("documents", str(document_id))
    response = original_file(db, document)
    return Response(content=response.content, media_type=document["mime_type"],
        headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff"})


@router.get("/documents/{document_id}")
def document_metadata(document_id: UUID, user: Identity = Depends(authenticate)):
    return Database(user).get("documents", str(document_id))


def original_file(db: Database, document: dict):
    try:
        return db.storage("GET", document["storage_path"])
    except HTTPException as exc:
        if exc.status_code == 404 and document.get("status") == "processing":
            raise HTTPException(409, "The original is still uploading. Keep working and retry opening this source shortly.") from exc
        raise


@router.get("/documents/{document_id}/preview")
def preview(document_id: UUID, user: Identity = Depends(authenticate)):
    db = Database(user)
    document = db.get("documents", str(document_id))
    original = original_file(db, document)
    try:
        pdf = source_pdf(original.content, document["source"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(content=pdf, media_type="application/pdf", headers={"Content-Disposition": "inline",
        "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


def ensure_current_sources(db: Database, document_ids: list[str]):
    """Queue outdated sources without doing file processing inside a chat request."""
    store = vector_store()
    pending = False
    for document_id in document_ids:
        document = db.get("documents", document_id)
        if document["status"] == "processing":
            pending = True
            continue
        lock = file_lock(db.user.id, document_id)
        if not lock.acquire(blocking=False):
            pending = True
            continue
        try:
            if document["status"] != "ready":
                raise HTTPException(400, "Select a ready source. Failed files can be retried in the source panel.")
            if not store.needs_reindex(db.user.id, document_id):
                continue
            schedule_index(db, document)
            pending = True
        finally:
            lock.release()
    if pending:
        raise HTTPException(409, "These sources are indexing in the background. You can use other ready sources while they finish, then retry this question.")


@router.post("/documents/{document_id}/reindex", status_code=202)
def reindex_document(document_id: UUID, user: Identity = Depends(authenticate)):
    db = Database(user)
    return schedule_index(db, db.get("documents", str(document_id)))


@router.post("/documents/{document_id}/retry", status_code=202)
def retry_document(document_id: UUID, file: UploadFile | None = File(None), user: Identity = Depends(authenticate)):
    db = Database(user)
    document = db.get("documents", str(document_id))
    content = None
    if file:
        filename = (file.filename or "").replace("\\", "/").split("/")[-1][:180]
        if Path(filename).suffix.lower() != Path(document["source"]).suffix.lower():
            raise HTTPException(400, "Choose a replacement with the same file type as the original.")
        content = file.file.read(settings.max_upload_bytes + 1)
        file.file.close()
        if not content or len(content) > settings.max_upload_bytes:
            raise HTTPException(413, "Choose a nonempty file smaller than 25 MB.")
    return schedule_index(db, document, content)


@router.delete("/documents/{document_id}", status_code=202)
def delete_document(document_id: UUID, user: Identity = Depends(authenticate)):
    db = Database(user)
    row = db.get("documents", str(document_id))
    return schedule_delete(db, row)


@router.get("/conversations")
def conversations(user: Identity = Depends(authenticate)):
    return Database(user).list("conversations", order="updated_at.desc")


@router.post("/conversations", status_code=201)
def create_conversation(body: ConversationCreate, user: Identity = Depends(authenticate)):
    db = Database(user)
    ids = db.validate_documents([str(value) for value in body.document_ids])
    return db.insert("conversations", {"title": body.title, "document_ids": ids})


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: UUID, user: Identity = Depends(authenticate)):
    db = Database(user)
    conversation = db.get("conversations", str(conversation_id))
    return {"conversation": conversation, "messages": db.messages(str(conversation_id))}


@router.patch("/conversations/{conversation_id}")
def update_conversation(conversation_id: UUID, body: ConversationUpdate, user: Identity = Depends(authenticate)):
    db = Database(user)
    changes = body.model_dump(exclude_none=True)
    if "document_ids" in changes:
        changes["document_ids"] = db.validate_documents([str(value) for value in changes["document_ids"]])
    if not changes:
        raise HTTPException(400, "Provide a title or source selection.")
    return db.update("conversations", str(conversation_id), changes)


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: UUID, user: Identity = Depends(authenticate)):
    db = Database(user)
    db.get("conversations", str(conversation_id))
    db.delete("conversations", str(conversation_id))


@router.post("/conversations/{conversation_id}/chat")
def chat(conversation_id: UUID, body: ChatRequest, user: Identity = Depends(authenticate)):
    db, cid = Database(user), str(conversation_id)
    with conversation_lock(f"{user.id}/{cid}"):
        conversation = db.get("conversations", cid)
        existing = db.request("GET", "research_messages", params={"conversation_id": f"eq.{cid}",
            "request_id": f"eq.{body.request_id}", "order": "created_at.asc,id.asc"})
        if existing:
            return {"conversation": conversation, "messages": existing}
        selected = conversation["document_ids"]
        if not selected:
            raise HTTPException(400, "Add and select at least one document before asking a question.")
        try:
            ensure_current_sources(db, selected)
            db.validate_documents(selected)
            answer = answer_question(body, user.id, selected, db.messages(cid, recent=True))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Retrieval failed for conversation %s (%s)", cid, type(exc).__name__)
            raise HTTPException(503, "Search is unavailable. Check Qdrant and the embedding model, then retry.") from exc
        return db.request("POST", "rpc/research_save_turn", body={
            "p_conversation_id": cid, "p_request_id": str(body.request_id), "p_question": body.content,
            "p_answer": answer["content"], "p_metadata": {"citations": answer["citations"],
                "grounded": answer["grounded"], "model": answer["model"], "mode": body.mode,
                "document_ids": selected},
        })

