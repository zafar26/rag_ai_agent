"""
Base class for all specialized agents. Each concrete agent just sets
its own name/persona; retrieval + generation logic is shared here.
"""

from rag.vector_store import query_documents
from llm.generator import generate_response


class BaseAgent:
    name: str = "base"
    system_prompt: str = "You are a helpful assistant."

    def handle(self, query: str, top_k: int = 3) -> dict:
        """
        1. Retrieve relevant chunks from this agent's own knowledge base
        2. Generate a response using the local LLM, grounded in that context
        """
        context_chunks = query_documents(self.name, query, top_k=top_k)
        answer = generate_response(
            system_prompt=self.system_prompt,
            context_chunks=context_chunks,
            user_query=query,
        )
        return {
            "agent": self.name,
            "answer": answer,
            "retrieved_context": context_chunks,
        }
