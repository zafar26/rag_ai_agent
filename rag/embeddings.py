"""
Shared embedding utility.

Both the PyTorch router classifier and the ChromaDB retrieval pipeline
need to turn text into vectors. We use a single small, fast
sentence-transformers model for both, so we don't load two different
embedding models into memory.
"""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2 is small (~80MB), fast on CPU, and produces
# 384-dimensional embeddings. Good default for a demo/portfolio project.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once and cache it (singleton)."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert a list of strings into a (N, EMBEDDING_DIM) numpy array.
    Used for both training the router and for RAG document/query embedding.
    """
    model = get_embedder()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings
