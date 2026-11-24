"""Embedding utilities used by all four assessment tasks."""
from __future__ import annotations

import os
import time
from typing import Sequence

import numpy as np


class LocalEmbedder:
    """Loads all-MiniLM-L6-v2 once and returns normalized 384-D vectors."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


class HashEmbedder:
    """Small deterministic embedder for automated tests; not for real retrieval."""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            v = np.zeros(self.dimension, dtype=np.float32)
            for token in text.lower().split():
                idx = sum((i + 1) * ord(c) for i, c in enumerate(token)) % self.dimension
                v[idx] += 1.0
            norm = np.linalg.norm(v)
            vectors.append(v / norm if norm else v)
        return np.asarray(vectors, dtype=np.float32)


def cosine_similarity(a, b) -> float:
    """Cosine similarity implemented from the mathematical definition."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def optional_api_embeddings(texts):
    """Call an OpenAI-compatible embeddings endpoint if credentials are configured."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "Skipped: OPENAI_API_KEY is not configured."
    from openai import OpenAI
    kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    result = client.embeddings.create(model=model, input=list(texts))
    return np.asarray([item.embedding for item in result.data], dtype=np.float32), model


def timed_encode(embedder, texts):
    start = time.perf_counter()
    vectors = embedder.encode(texts)
    return vectors, time.perf_counter() - start
