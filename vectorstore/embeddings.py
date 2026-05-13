"""
vectorstore/embeddings.py
=========================

Local SentenceTransformer embeddings.  Wrapped in a tiny class so the rest of
the code uses a single `.embed(texts) -> np.ndarray` API regardless of which
underlying model is configured.

We deliberately keep the model in-process (not on a remote API) so the
pipeline works fully offline once the model is cached.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import get_logger, settings

log = get_logger(__name__)


@lru_cache(maxsize=2)
def _load_model(name: str) -> SentenceTransformer:
    log.info("Loading embedding model: %s", name)
    return SentenceTransformer(name)


class Embedder:
    """Embed texts with a cached SentenceTransformer model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.model = _load_model(self.model_name)
        self.dim: int = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an `(n, dim)` float32 ndarray, L2-normalised for cosine search."""
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string and return a `(dim,)` float32 vector."""
        return self.embed([text])[0]
