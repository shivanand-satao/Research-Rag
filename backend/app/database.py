"""User-scoped Supabase REST and private storage adapter."""
from __future__ import annotations
from dataclasses import dataclass
from urllib.parse import quote

import httpx
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

client = httpx.Client(timeout=45, headers={"User-Agent": "research-rag/1.0"})
bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Identity:
    id: str
    token: str


def headers(token: str) -> dict[str, str]:
    if not settings.supabase_url or not settings.supabase_key:
        raise HTTPException(503, "Configure SUPABASE_URL and SUPABASE_KEY in backend/.env.")
    return {"apikey": settings.supabase_key, "Authorization": f"Bearer {token}"}


def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Identity:
    if credentials is None:
        raise HTTPException(401, "Sign in to continue.")
    try:
        response = client.get(f"{settings.supabase_url}/auth/v1/user", headers=headers(credentials.credentials))
    except httpx.RequestError as exc:
        raise HTTPException(503, "Authentication service is unavailable.") from exc
    if response.status_code in {401, 403}:
        raise HTTPException(401, "Your session expired. Please sign in again.")
    if not response.is_success or not response.json().get("id"):
        raise HTTPException(503, "Authentication service is unavailable.")
    return Identity(response.json()["id"], credentials.credentials)


class Database:
    def __init__(self, identity: Identity):
        self.user = identity

    def request(self, method: str, resource: str, *, params=None, body=None):
        try:
            response = client.request(method, f"{settings.supabase_url}/rest/v1/{resource}",
                headers={**headers(self.user.token), "Prefer": "return=representation"}, params=params, json=body)
        except httpx.RequestError as exc:
            raise HTTPException(503, "Research storage is unavailable. Try again shortly.") from exc
        if not response.is_success:
            if response.status_code == 401:
                raise HTTPException(401, "Your session expired. Please sign in again.")
            code = response.json().get("code", "") if response.headers.get("content-type", "").startswith("application/json") else ""
            if code in {"42P01", "PGRST205", "PGRST202"}:
                raise HTTPException(503, "Apply the Research RAG Supabase migration before using the workspace.")
            if code == "23505":
                raise HTTPException(409, "This request already exists.")
            raise HTTPException(502, "Research storage could not complete this request.")
        return response.json() if response.content else None

    def list(self, table: str, **params):
        scoped = {"user_id": f"eq.{self.user.id}", **params}
        if "limit" in params:
            return self.request("GET", f"research_{table}", params=scoped)
        rows, offset = [], 0
        while True:
            page = self.request("GET", f"research_{table}", params={**scoped, "limit": "500", "offset": str(offset)})
            rows.extend(page)
            if len(page) < 500:
                return rows
            offset += 500

    def get(self, table: str, item_id: str):
        rows = self.list(table, id=f"eq.{item_id}", limit="1")
        if not rows:
            raise HTTPException(404, f"{table.title().rstrip('s')} not found.")
        return rows[0]

    def insert(self, table: str, body: dict):
        return self.request("POST", f"research_{table}", body={**body, "user_id": self.user.id})[0]

    def update(self, table: str, item_id: str, body: dict):
        rows = self.request("PATCH", f"research_{table}",
            params={"id": f"eq.{item_id}", "user_id": f"eq.{self.user.id}"}, body=body)
        if not rows:
            raise HTTPException(404, "Item not found.")
        return rows[0]

    def delete(self, table: str, item_id: str):
        self.request("DELETE", f"research_{table}", params={"id": f"eq.{item_id}", "user_id": f"eq.{self.user.id}"})

    def messages(self, conversation_id: str, *, recent=False):
        params = {
            "conversation_id": f"eq.{conversation_id}", "order": "created_at.desc,id.desc" if recent else "created_at.asc,id.asc",
        }
        if recent:
            rows = self.request("GET", "research_messages", params={**params, "limit": "20"})
            return list(reversed(rows))
        rows, offset = [], 0
        while True:
            page = self.request("GET", "research_messages", params={**params, "limit": "500", "offset": str(offset)})
            rows.extend(page)
            if len(page) < 500:
                return rows
            offset += 500

    def validate_documents(self, document_ids: list[str]) -> list[str]:
        ids = list(dict.fromkeys(document_ids))
        if not ids:
            return []
        rows = self.list("documents", id=f"in.({','.join(ids)})", select="id,status")
        if len(rows) != len(ids) or any(row["status"] != "ready" for row in rows):
            raise HTTPException(400, "Select only ready documents that belong to your account.")
        return ids

    def storage(self, method: str, path: str, *, content=None, mime=None):
        base = f"{settings.supabase_url}/storage/v1/object/{settings.storage_bucket}"
        url = base if method == "DELETE" else f"{base}/{quote(path, safe='/')}"
        try:
            response = client.request(method, url, headers={**headers(self.user.token),
                **({"Content-Type": mime} if mime else {})},
                **({"json": {"prefixes": [path]}} if method == "DELETE" else {"content": content}))
        except httpx.RequestError as exc:
            raise HTTPException(503, "Document storage is unavailable.") from exc
        status = response.status_code
        if not response.is_success:
            try:
                # Storage may wrap a missing-object 404 inside an HTTP 400 response.
                if str(response.json().get("statusCode")) == "404":
                    status = 404
            except (ValueError, AttributeError):
                pass
        if method == "DELETE" and status == 404:
            return None
        if not response.is_success:
            if status in {401, 403}:
                raise HTTPException(status, "Document storage access was denied. Sign in again and check the storage policies.")
            if status == 404:
                raise HTTPException(404, "The saved original is unavailable. Choose the file again to retry.")
            raise HTTPException(502, "Document storage failed. Check that the migration and storage policies are installed.")
        return response

