"""Cloud startup, production origins, and tokenizer-only deployment regressions."""
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app import vectors
from app.main import app


def test_root_and_render_health_probe_are_available():
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.head("/").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"


def test_production_startup_without_local_ml_or_ocr_dependencies():
    env = dict(os.environ)
    for key in ["FRONTEND_URL", "CORS_ORIGINS", "RESEARCH_REDIRECT_URL", "RESEARCH_RERANK_ENABLED", "OCR", "QDRANT_ALL_MINILM_L6_V2"]:
        env.pop(key, None)
    env["APP_ENV"] = "production"
    code = '''
import importlib.abc
import sys
class WithoutOptionalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'sentence_transformers', 'pdfplumber', 'pytesseract', 'fastembed'}:
            raise ImportError('Optional dependency unavailable')
sys.meta_path.insert(0, WithoutOptionalPackages())
from fastapi.testclient import TestClient
from app import main
from app.config import settings
main.start_cleanup = lambda: None
assert settings.frontend_url == 'https://research-rag-seven.vercel.app'
assert settings.redirect_url == 'https://research-rag-seven.vercel.app/auth/callback'
assert settings.qdrant_all_minilm_l6_v2 and not settings.rerank and not settings.ocr
with TestClient(main.app) as client:
    assert client.get('/api/health').status_code == 200
    assert client.head('/').status_code == 200
    response = client.options('/api/documents', headers={
        'Origin': 'https://research-rag-seven.vercel.app',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'authorization,content-type'})
    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == 'https://research-rag-seven.vercel.app'
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print('Cloud startup and production preflight passed without optional dependencies')
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env, cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_cloud_tokenizer_uses_json_without_truncation_or_padding(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    import huggingface_hub
    raw = Tokenizer(models.WordPiece(vocab={"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    raw.enable_truncation(max_length=256)
    raw.enable_padding(length=256)
    path = tmp_path / "tokenizer.json"
    raw.save(str(path))
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    tokenizer = vectors._minilm_tokenizer.__wrapped__()
    assert len(tokenizer.encode("hello " * 600)) == 600
    assert len(tokenizer.encode("hello", add_special_tokens=True)) == 3
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"
    assert calls[0]["filename"] == "tokenizer.json"
