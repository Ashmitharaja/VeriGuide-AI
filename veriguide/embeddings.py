"""Shared sentence-embedding model (loaded once, downloaded on first use ~90 MB)."""

from functools import lru_cache

import numpy as np

from . import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Return unit-normalised vectors, so dot product == cosine similarity."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    # Small batches keep peak memory low on modest machines.
    return _model().encode(
        texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
