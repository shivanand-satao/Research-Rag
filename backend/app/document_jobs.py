"""Independent, bounded file jobs with cancellation, retry, and user-scoped progress."""
import json
import logging
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from fastapi import HTTPException

from .config import BACKEND
from .database import Database
from .processing import extract
from .vectors import embedding_tokenizer, vector_store

logger = logging.getLogger(__name__)
cache = BACKEND / ".cache" / "uploads"
cleanup_path = BACKEND / ".cache" / "vector-cleanup.json"
pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="research-file")
delete_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-delete")
registry_lock = threading.Lock()
locks = weakref.WeakValueDictionary()
activities: dict[tuple[str, str], "Activity"] = {}
_cleanup_running = False


@dataclass
class Activity:
    operation: str
    phase: str = "Queued"
    progress: int = 0
    cancelled: threading.Event = field(default_factory=threading.Event)


def file_lock(user_id: str, document_id: str):
    with registry_lock:
        key = (user_id, document_id)
        lock = locks.get(key)
        if lock is None:
            lock = threading.Lock(); locks[key] = lock
        return lock


def failure_message(exc: Exception, phase: str) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    if isinstance(exc, ValueError):
        return str(exc)[:350]
    code = getattr(exc, "status_code", None)
    source = getattr(exc, "source", exc)
    if code in {401, 403}:
        return "The vector service denied access. Check that QDRANT_API_KEY can write to this cluster, then retry."
    if code == 400:
        return "Qdrant rejected cloud indexing (HTTP 400). Enable inference on the cluster and check that sentence-transformers/all-minilm-l6-v2 is available, then retry."
    if isinstance(source, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException)):
        return "Cannot reach the vector service over HTTPS. Check the backend network connection and cluster availability, then retry."
    if phase == "Preparing tokenizer":
        return "The tokenizer could not be loaded. Check the backend tokenizer cache or Hugging Face connection, then retry."
    if phase == "Reading document":
        return "The document could not be read. Check that the PDF opens correctly and is not password protected."
    return f"{phase} failed ({type(exc).__name__}). Retry this file; its original is retained."


def _progress(activity: Activity, phase: str, progress: int):
    if activity.cancelled.is_set():
        raise InterruptedError("File processing cancelled")
    activity.phase, activity.progress = phase, progress


def _process(db: Database, document: dict, activity: Activity):
    key = (db.user.id, document["id"])
    local = cache / f"{document['id']}.bin"
    stored = False
    try:
        with file_lock(*key):
            _progress(activity, "Saving original", 5)
            if local.exists():
                content = local.read_bytes()
                # A retry may already have an original; replace it safely through Storage.
                db.storage("DELETE", document["storage_path"])
                db.storage("POST", document["storage_path"], content=content, mime=document["mime_type"])
            else:
                content = db.storage("GET", document["storage_path"]).content
            stored = True
            _progress(activity, "Preparing tokenizer", 15)
            tokenizer = embedding_tokenizer()
            _progress(activity, "Reading document", 25)
            chunks = extract(content, document["source"], tokenizer=tokenizer,
                progress=lambda done, total: _progress(activity, "Reading document", 25 + int(20 * done / total)))
            _progress(activity, "Embedding in Qdrant and indexing", 45)
            vector_store().index(db.user.id, document["id"], document["source"], chunks,
                cancelled=activity.cancelled.is_set,
                progress=lambda done, total: _progress(activity, "Indexing passages", 65 + int(30 * done / total)))
            _progress(activity, "Finishing", 98)
            db.update("documents", document["id"], {"status": "ready", "chunk_count": len(chunks), "error": None})
            activity.progress = 100
    except InterruptedError:
        logger.info("File processing cancelled for %s", document["id"])
    except Exception as exc:
        # Keep the original file so failures can be retried without another upload.
        logger.warning("File %s failed during %s: %s (HTTP %s)", document["id"], activity.phase,
            type(exc).__name__, getattr(exc, "status_code", None))
        if not activity.cancelled.is_set():
            try:
                db.update("documents", document["id"], {"status": "failed", "error": failure_message(exc, activity.phase)})
            except Exception:
                logger.warning("Unable to save file failure state for %s", document["id"])
    finally:
        if stored:
            try:
                local.unlink(missing_ok=True)
            except OSError:
                logger.warning("Unable to clear upload cache for %s", document["id"])
        with registry_lock:
            if activities.get(key) is activity:
                activities.pop(key, None)


def schedule_index(db: Database, document: dict, content: bytes | None = None):
    key = (db.user.id, document["id"])
    with registry_lock:
        if document["status"] == "deleting":
            raise HTTPException(409, "This file is being deleted. Retry deleting it from the source panel.")
        if key in activities:
            raise HTTPException(409, "This file already has an action in progress.")
        if content is not None:
            cache.mkdir(parents=True, exist_ok=True)
            (cache / f"{document['id']}.bin").write_bytes(content)
        activity = Activity("index")
        activities[key] = activity
    try:
        row = db.update("documents", document["id"], {"status": "processing", "error": None})
        pool.submit(_process, db, row, activity)
    except Exception:
        with registry_lock:
            activities.pop(key, None)
        raise
    return {**row, "activity": {"phase": "Queued", "progress": 0}}


def _save_cleanup(pending: list[dict]):
    cleanup_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cleanup_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(pending), encoding="utf-8")
    temporary.replace(cleanup_path)


def _defer_vectors(user_id: str, document_id: str):
    with registry_lock:
        pending = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else []
        entry = {"user_id": user_id, "document_id": document_id}
        if entry not in pending:
            pending.append(entry)
        _save_cleanup(pending)


def clean_deferred_vectors():
    with registry_lock:
        pending = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else []
    done = []
    for entry in pending:
        try:
            vector_store().delete(entry["user_id"], entry["document_id"])
            done.append(entry)
        except Exception:
            break
    if done:
        with registry_lock:
            latest = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else []
            _save_cleanup([entry for entry in latest if entry not in done])


def start_cleanup():
    global _cleanup_running
    with registry_lock:
        if _cleanup_running or not cleanup_path.exists():
            return
        _cleanup_running = True
    def run():
        global _cleanup_running
        try:
            clean_deferred_vectors()
        finally:
            with registry_lock:
                _cleanup_running = False
    threading.Thread(target=run, name="research-vector-cleanup", daemon=True).start()


def _delete(db: Database, document: dict, activity: Activity):
    key = (db.user.id, document["id"])
    try:
        with file_lock(*key):
            activity.phase, activity.progress = "Removing original", 25
            db.storage("DELETE", document["storage_path"])
            activity.phase, activity.progress = "Removing indexed passages", 65
            if document["chunk_count"] > 0:
                vector_store().delete(db.user.id, document["id"])
            else:
                # Failed/cancelled uploads must be removable even if vector connectivity failed.
                _defer_vectors(db.user.id, document["id"])
            db.delete("documents", document["id"])
            (cache / f"{document['id']}.bin").unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Deletion failed for %s (%s)", document["id"], type(exc).__name__)
        try:
            db.update("documents", document["id"], {"status": "deleting", "error": failure_message(exc, activity.phase)})
        except Exception:
            pass
    finally:
        with registry_lock:
            if activities.get(key) is activity:
                activities.pop(key, None)
    start_cleanup()


def schedule_delete(db: Database, document: dict):
    key = (db.user.id, document["id"])
    with registry_lock:
        previous = activities.get(key)
        if previous and previous.operation == "delete":
            return {**document, "activity": {"phase": previous.phase, "progress": previous.progress}}
        if previous:
            previous.cancelled.set()
        activity = Activity("delete", "Cancelling processing" if previous else "Queued")
        activities[key] = activity
    try:
        row = db.update("documents", document["id"], {"status": "deleting", "error": None})
        delete_pool.submit(_delete, db, row, activity)
    except Exception:
        with registry_lock:
            activities.pop(key, None)
        raise
    return {**row, "activity": {"phase": activity.phase, "progress": 0}}


def list_documents(db: Database):
    rows = db.list("documents", order="created_at.desc")
    for row in rows:
        key = (db.user.id, row["id"])
        with registry_lock:
            activity = activities.get(key)
        if activity:
            row["activity"] = {"phase": activity.phase, "progress": activity.progress}
        elif row["status"] == "processing" or (row["status"] == "deleting" and not row.get("error")):
            # Resume interrupted jobs with the current user's fresh token, without storing tokens.
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))).total_seconds()
            if age > 15:
                try:
                    row.update(schedule_index(db, row) if row["status"] == "processing" else schedule_delete(db, row))
                except HTTPException as exc:
                    if exc.status_code != 409:
                        raise
    return rows
