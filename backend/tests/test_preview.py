import io
from uuid import uuid4

from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from app.database import Identity, authenticate
from app.main import app
from app.preview import source_pdf
from app import routes


def test_original_pdf_is_returned_unchanged():
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buffer = io.BytesIO()
    writer.write(buffer)
    original = buffer.getvalue()
    assert source_pdf(original, "report.pdf") == original


def test_text_sources_have_a_readable_pdf_preview():
    output = source_pdf(b"# Findings\nA detector achieved 27.4 millivolts per ppm.\n\nThe result was repeatable.", "findings.md")
    reader = PdfReader(io.BytesIO(output))
    assert len(reader.pages) == 1
    assert len(reader.pages[0].images) == 1


def test_pdf_preview_is_user_scoped_and_served_inline(monkeypatch):
    document_id = str(uuid4())
    calls = []
    class Database:
        def __init__(self, user):
            assert user.id == "alice"
        def get(self, table, item_id):
            assert table == "documents" and item_id == document_id
            return {"source": "report.pdf", "storage_path": "alice/report.pdf"}
        def storage(self, method, path):
            calls.append((method, path))
            return type("Original", (), {"content": b"%PDF-original"})()
    monkeypatch.setattr(routes, "Database", Database)
    app.dependency_overrides[authenticate] = lambda: Identity("alice", "test")
    try:
        response = TestClient(app).get(f"/api/documents/{document_id}/preview")
        assert response.status_code == 200
        assert response.content == b"%PDF-original"
        assert response.headers["content-type"] == "application/pdf"
        assert response.headers["content-disposition"] == "inline"
        assert calls == [("GET", "alice/report.pdf")]
    finally:
        app.dependency_overrides.clear()


def test_pdf_preview_requires_login():
    assert TestClient(app).get(f"/api/documents/{uuid4()}/preview").status_code == 401
