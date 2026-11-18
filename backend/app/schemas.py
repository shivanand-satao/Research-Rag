from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ConversationCreate(BaseModel):
    title: str = Field(default="New research", min_length=1, max_length=120)
    document_ids: list[UUID] = Field(default_factory=list, max_length=30)

    @field_validator("title")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    document_ids: list[UUID] | None = Field(default=None, max_length=30)

    @field_validator("title")
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip() if value else value


class ChatRequest(BaseModel):
    request_id: UUID
    content: str = Field(min_length=1, max_length=6000)
    mode: Literal["hybrid", "vector", "text"] = "hybrid"
    scope: Literal["auto", "focused", "all_documents"] = "auto"
    top_k: int = Field(default=8, ge=2, le=16)

    @field_validator("content")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value.strip()

