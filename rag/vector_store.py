"""
ChromaDB wrapper. Each agent gets its own isolated collection so
agents never accidentally retrieve each other's knowledge.
"""

from pathlib import Path
from typing import List

import chromadb

from rag.embeddings import embed_texts

CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")

_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(agent_name: str):
    """Each agent's knowledge base lives in its own Chroma collection."""
    return _client.get_or_create_collection(name=f"agent_{agent_name}")


def add_documents(agent_name: str, documents: List[str]) -> int:
    """
    Chunk-free simple ingestion: each line/paragraph passed in is stored
    as one retrievable document. Embeddings are computed ourselves
    (shared model) rather than Chroma's default embedding function, so
    router and retrieval stay consistent.
    """
    collection = get_or_create_collection(agent_name)
    embeddings = embed_texts(documents).tolist()
    ids = [f"{agent_name}_{collection.count() + i}" for i in range(len(documents))]

    collection.add(documents=documents, embeddings=embeddings, ids=ids)
    return len(documents)


def query_documents(agent_name: str, query: str, top_k: int = 3) -> List[str]:
    """Retrieve the top_k most relevant chunks for a query, for one agent."""
    collection = get_or_create_collection(agent_name)
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=min(top_k, collection.count()))
    return results["documents"][0] if results["documents"] else []
