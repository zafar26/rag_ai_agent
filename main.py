"""
FastAPI app for the multi-agent RAG system.

Endpoints:
    POST /chat     -> route a query to the right agent and get an answer
    POST /ingest    -> add new documents to a specific agent's knowledge base
    GET  /agents    -> list available agents
    GET  /health    -> basic health check

Before running:
    1. pip install -r requirements.txt
    2. python train_router.py     (trains the PyTorch router classifier)
    3. python ingest.py           (loads sample docs into ChromaDB)
    4. uvicorn main:app --reload
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agents.router_agent import route_and_handle, AGENT_REGISTRY
from rag.vector_store import add_documents

app = FastAPI(
    title="Multi-Agent RAG System",
    description="PyTorch-routed, ChromaDB-backed, locally-hosted LLM agent system",
    version="1.0.0",
)

# Allow the frontend (served from a different origin, e.g. a static file
# server or file://) to call this API directly from the browser.
# Kept even though we now also serve the frontend ourselves, in case you
# host the frontend separately (e.g. Netlify) while the API stays local.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent / "frontend"


@app.get("/")
def serve_frontend():
    """Serve the chatbot UI at the root URL, e.g. http://127.0.0.1:8000/"""
    return FileResponse(FRONTEND_DIR / "index.html")


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    agent: str
    answer: str
    retrieved_context: list[str]
    routing: dict


class IngestRequest(BaseModel):
    agent_name: str
    documents: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/agents")
def list_agents():
    return {"available_agents": list(AGENT_REGISTRY.keys())}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = route_and_handle(request.query)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/ingest")
def ingest(request: IngestRequest):
    if request.agent_name not in AGENT_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent '{request.agent_name}'. Valid agents: {list(AGENT_REGISTRY.keys())}",
        )
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    count = add_documents(request.agent_name, request.documents)
    return {"ingested": count, "agent": request.agent_name}
