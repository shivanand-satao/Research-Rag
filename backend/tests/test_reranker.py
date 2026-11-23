"""Query/passage encoding, bounded inference, and reranker readiness."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from app import rag
from app.main import health
from app.onnx_reranker import OnnxReranker


@pytest.fixture
def onnx(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    tokenizer = Tokenizer(models.WordPiece(vocab={"[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
        "[PAD]": 3, "query": 4, "passage": 5, "word": 6}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    downloads, calls = [], []
    state = SimpleNamespace(active=0, peak=0, scored=0, invalid=None, options=None, providers=None)

    def download(**kwargs):
        downloads.append(kwargs)
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("Not cached yet")
        return str(tokenizer_path if kwargs["filename"] == "tokenizer.json" else tmp_path / "model.onnx")

    class Session:
        def __init__(self, path, sess_options, providers):
            state.options, state.providers = sess_options, providers

        def get_inputs(self):
            return [SimpleNamespace(name=name) for name in ("input_ids", "attention_mask", "token_type_ids")]

        def get_outputs(self):
            return [SimpleNamespace(name="logits")]

        def run(self, outputs, inputs):
            state.active += 1
            state.peak = max(state.peak, state.active)
            try:
                time.sleep(.005)
                calls.append(inputs)
                count = len(inputs["input_ids"])
                scores = np.arange(state.scored, state.scored + count, dtype=np.float32).reshape(-1, 1)
                state.scored += count
                if state.invalid == "nan":
                    scores[0, 0] = np.nan
                elif state.invalid == "multiple_labels":
                    scores = np.repeat(scores, 2, axis=1)
                elif state.invalid == "missing_score":
                    scores = scores[:-1]
                return [scores]
            finally:
                state.active -= 1

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=SimpleNamespace,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"), InferenceSession=Session))
    model = OnnxReranker("example/model", "fixed-revision", "onnx/model.onnx", tmp_path)
    return model, downloads, calls, state


def test_pairs_preserve_passage_segments_with_padding_and_truncation(onnx):
    model, downloads, calls, state = onnx
    pairs = [("query", "passage " + "word " * 600), ("query", "passage"), ("query", "word")]
    assert model.predict(pairs, batch_size=2) == [0., 1., 2.]
    assert [len(call["input_ids"]) for call in calls] == [2, 1]
    assert calls[0]["input_ids"].shape == (2, 512)
    assert calls[0]["token_type_ids"][0, :4].tolist() == [0, 0, 0, 1]
    assert calls[0]["attention_mask"][1, :5].tolist() == [1] * 5
    assert not calls[0]["attention_mask"][1, 5:].any()
    assert all(array.dtype == np.int64 for call in calls for array in call.values())
    assert {item["filename"] for item in downloads} == {"tokenizer.json", "onnx/model.onnx"}
    assert all(item["revision"] == "fixed-revision" for item in downloads)
    assert state.providers == ["CPUExecutionProvider"]
    assert state.options.intra_op_num_threads == state.options.inter_op_num_threads == 1


def test_concurrent_questions_do_not_run_inference_simultaneously(onnx):
    model, _, _, state = onnx
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: model.predict([("query", "passage")] * 2, batch_size=1), range(3)))
    assert all(len(result) == 2 for result in results)
    assert state.peak == 1


@pytest.mark.parametrize("invalid", ["nan", "multiple_labels", "missing_score"])
def test_invalid_model_scores_are_rejected_instead_of_dropping_passages(onnx, invalid):
    model, _, _, state = onnx
    state.invalid = invalid
    with pytest.raises(ValueError, match="one finite relevance score"):
        model.predict([("query", "passage")])


def test_readiness_changes_only_after_the_model_loads(monkeypatch):
    from app import onnx_reranker
    model, loads = object(), []
    monkeypatch.setattr(rag, "settings", replace(rag.settings, rerank=True))
    monkeypatch.setattr(rag, "_rerank_started", False)
    monkeypatch.setattr(rag, "_rerank_model", None)
    monkeypatch.setattr(rag, "_rerank_status", "not_started")
    monkeypatch.setattr(onnx_reranker, "load_reranker", lambda: loads.append(True) or model)
    assert health()["rerank_status"] == "not_started"
    rag.prepare_reranker()
    rag.prepare_reranker()
    assert health()["rerank_status"] == "ready"
    assert rag.reranker() is model and len(loads) == 1
