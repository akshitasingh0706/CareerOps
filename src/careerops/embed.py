"""Local embeddings through fastembed. Needs the `local-embeddings` extra."""
from __future__ import annotations

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedder:
    """Wraps fastembed. The model downloads from Hugging Face on first use."""

    dim = 384

    def __init__(self, model: str = MODEL) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
