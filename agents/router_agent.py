"""
Router agent: the entry point for every user query.

Embeds the query -> runs it through the trained PyTorch classifier ->
picks the matching specialized agent -> delegates the query to it.
"""

from functools import lru_cache

from models.router_classifier import load_model, predict_label
from rag.embeddings import embed_texts
from agents.tech_support_agent import TechSupportAgent
from agents.hr_agent import HRAgent
from agents.sales_agent import SalesAgent

AGENT_REGISTRY = {
    "tech_support": TechSupportAgent(),
    "hr": HRAgent(),
    "sales": SalesAgent(),
}


@lru_cache(maxsize=1)
def _get_router_model():
    """Load the trained classifier + label list once (singleton)."""
    return load_model()


def route_and_handle(query: str) -> dict:
    """
    Full pipeline for one user turn:
    1. Classify which agent should handle the query (PyTorch model)
    2. Delegate to that agent, which retrieves + generates its own answer
    """
    model, labels = _get_router_model()
    embedding = embed_texts([query])[0]
    routing_result = predict_label(model, labels, embedding)

    predicted_agent_name = routing_result["predicted_agent"]
    agent = AGENT_REGISTRY.get(predicted_agent_name)

    if agent is None:
        return {
            "error": f"No agent registered for predicted label '{predicted_agent_name}'",
            "routing": routing_result,
        }

    result = agent.handle(query)
    result["routing"] = routing_result
    return result
