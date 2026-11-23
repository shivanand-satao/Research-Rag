"""Cloud startup, production origins, and optional reranker failure handling."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import vectors
from app.main import app


def test_root_and_render_health_probe_are_available():
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.head("/").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"


def test_reranker_download_receives_clean_dashboard_values():
    env = dict(os.environ)
    env.update({
        "APP_ENV": "production",
        "RESEARCH_RERANK_ENABLED": " true\r\n",
        "RESEARCH_RERANK_MODEL": " cross-encoder/ms-marco-MiniLM-L-2-v2\n",
        "RESEARCH_RERANK_REVISION": "\t1b5cd67b15209f24824c50370e0397743aa9b787\r\n",
        "RESEARCH_RERANK_ONNX_FILE": " onnx/model_quint8_avx2.onnx\n",
    })
    code = '''
from app.config import settings, RERANK_MODEL, RERANK_REVISION
from app import onnx_reranker
received = []
onnx_reranker.OnnxReranker = lambda model, revision, filename, cache: received.append((model, revision, filename))
onnx_reranker.load_reranker()
assert settings.rerank
assert received == [(RERANK_MODEL, RERANK_REVISION, 'onnx/model_quint8_avx2.onnx')]
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env,
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("enabled", [True, False])
def test_production_startup_without_local_ml_or_ocr_dependencies(enabled):
    env = dict(os.environ)
    for key in ["FRONTEND_URL", "CORS_ORIGINS", "RESEARCH_REDIRECT_URL", "RESEARCH_RERANK_ENABLED", "OCR"]:
        env.pop(key, None)
    env["APP_ENV"] = "production"
    if not enabled:
        env["RESEARCH_RERANK_ENABLED"] = "false"
    env["EXPECTED_RERANK_ENABLED"] = str(enabled).lower()
    code = '''
import importlib.abc
import os
import sys
class WithoutOptionalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'onnxruntime', 'torch', 'transformers', 'sentence_transformers', 'pdfplumber', 'pytesseract', 'fastembed'}:
            raise ImportError('Optional dependency unavailable')
sys.meta_path.insert(0, WithoutOptionalPackages())
from fastapi.testclient import TestClient
from app import main
from app.config import settings
main.start_cleanup = lambda: None
assert settings.frontend_url == 'https://research-rag-steel.vercel.app'
assert settings.redirect_url == 'https://research-rag-steel.vercel.app/auth/callback'
assert settings.embedding_model == 'sentence-transformers/all-minilm-l6-v2'
enabled = os.environ['EXPECTED_RERANK_ENABLED'] == 'true'
assert settings.rerank == enabled and not settings.ocr
if enabled:
    main.prepare_reranker()
with TestClient(main.app) as client:
    health = client.get('/api/health')
    assert health.status_code == 200
    assert health.json()['rerank_enabled'] == enabled
    assert health.json()['rerank_status'] == ('unavailable' if enabled else 'disabled')
    assert client.head('/').status_code == 200
    response = client.options('/api/documents', headers={
        'Origin': 'https://research-rag-steel.vercel.app',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'authorization,content-type'})
    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == 'https://research-rag-steel.vercel.app'
assert not {'onnxruntime', 'torch', 'transformers', 'sentence_transformers'} & sys.modules.keys()
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
