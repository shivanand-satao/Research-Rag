"""Quantized CPU cross-encoder inference without Torch or Transformers."""
import threading
from pathlib import Path

from .config import BACKEND, settings


class OnnxReranker:
    def __init__(self, model: str, revision: str, filename: str, cache_dir: Path):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        def download(asset):
            kwargs = dict(repo_id=model, revision=revision, filename=asset, cache_dir=str(cache_dir))
            try:
                return hf_hub_download(**kwargs, local_files_only=True)
            except FileNotFoundError:
                return hf_hub_download(**kwargs)

        tokenizer_path = download("tokenizer.json")
        model_path = download(filename)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise ValueError("The reranker tokenizer must define [PAD].")
        self.tokenizer.enable_truncation(max_length=512, strategy="longest_first")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name
        # Serialize requests to bound inference memory and protect tokenizer batching.
        self.lock = threading.Lock()

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 4) -> list[float]:
        import numpy as np

        if not pairs:
            return []
        batch_size = max(1, min(8, batch_size))
        scores = []
        with self.lock:
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                encoded = self.tokenizer.encode_batch(batch)
                features = {
                    "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                    "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                    "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
                }
                inputs = {name: features[name] for name in self.input_names}
                logits = np.asarray(self.session.run([self.output_name], inputs)[0])
                if logits.shape != (len(batch), 1) or not np.isfinite(logits).all():
                    raise ValueError("The reranker must return one finite relevance score per passage.")
                scores.extend(logits[:, 0].tolist())
        return scores


def load_reranker():
    return OnnxReranker(settings.rerank_model, settings.rerank_revision,
        settings.rerank_onnx_file, BACKEND / ".cache" / "reranker")


if __name__ == "__main__":
    # Render runs this during its build. Download/inference failures must fail the build.
    if settings.rerank:
        model = load_reranker()
        question = "What is the capital of France?"
        scores = model.predict([(question, "Paris is the capital of France."),
            (question, "A bread recipe uses flour and yeast.")], batch_size=1)
        if scores[0] <= scores[1]:
            raise ValueError("The reranker failed its passage relevance check.")
        print(f"Validated ONNX reranker: {settings.rerank_model} ({settings.rerank_revision})")
    from .vectors import embedding_tokenizer
    embedding_tokenizer()
    print("Embedding tokenizer cached for startup")
