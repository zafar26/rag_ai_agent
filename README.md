# Multi-Agent RAG System (FastAPI + PyTorch + ChromaDB + Local LLM)

A small but complete multi-agent RAG application:

<img width="2446" height="1398" alt="image" src="https://github.com/user-attachments/assets/bb80bda2-7986-4aa7-8a33-a2c9bda88763" />


- **FastAPI** — HTTP API layer
- **PyTorch** — trains a small classifier that routes each query to the right agent
- **ChromaDB** — vector database, one isolated collection per agent
- **Local LLM** — `Qwen2.5-0.5B-Instruct` via Hugging Face `transformers` (PyTorch backend), runs on CPU
- **Three specialized agents** — Tech Support, HR, Sales — each with its own knowledge base and persona

## How it works

```
User query
   │
   ▼
Router Agent (PyTorch classifier)
   │  embeds query → predicts category → picks an agent
   ▼
Specialized Agent (Tech Support / HR / Sales)
   │  retrieves relevant docs from ITS OWN ChromaDB collection
   ▼
Local LLM (Qwen2.5-0.5B-Instruct)
   │  generates an answer grounded in the retrieved context
   ▼
Response returned to user (with routing info + retrieved context, for transparency)
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the PyTorch router classifier
#    (embeds the example queries in data/router_training_data.json
#     and trains a small MLP to classify new queries)
python train_router.py

# 3. Ingest sample knowledge base docs into ChromaDB
python ingest.py

# 4. Start the API
uvicorn main:app --reload
```
<img width="1944" height="1230" alt="image" src="https://github.com/user-attachments/assets/12c136e5-3e27-4a41-ae65-2703f5fc0014" />

<img width="1980" height="950" alt="image" src="https://github.com/user-attachments/assets/82767c6a-fbc8-4687-be97-0cd52e8e3e7d" />


The API will be live at `http://127.0.0.1:8000`. Interactive docs (Swagger UI)
are auto-generated at `http://127.0.0.1:8000/docs`.

> Note: the first run of `train_router.py` and the first `/chat` request will
> download the embedding model (~80MB) and the LLM (~1GB) from Hugging Face.
> This requires internet access on whichever machine you run it on.

## Example usage

**Chat with the system:**
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?"}'
```

Response:
```json
{
  "agent": "tech_support",
  "answer": "To reset your password, go to the login page and click 'Forgot Password'...",
  "retrieved_context": ["To reset your password, go to the login page..."],
  "routing": {
    "predicted_agent": "tech_support",
    "confidence_scores": {"tech_support": 0.94, "hr": 0.03, "sales": 0.03}
  }
}
```

**Add new knowledge to an agent:**
```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "sales", "documents": ["We now offer a lifetime deal for $999."]}'
```

**List available agents:**
```bash
curl http://127.0.0.1:8000/agents
```

## Project structure

```
rag_agent_app/
├── main.py                    # FastAPI app + endpoints
├── train_router.py            # Trains the PyTorch router classifier
├── ingest.py                  # Loads sample docs into ChromaDB
├── requirements.txt
├── models/
│   ├── router_classifier.py   # PyTorch MLP model definition + load/save/predict
│   ├── router_classifier.pt   # (created after training)
│   └── router_labels.json     # (created after training)
├── rag/
│   ├── embeddings.py          # Shared sentence-transformers embedding wrapper
│   └── vector_store.py        # ChromaDB wrapper, one collection per agent
├── llm/
│   └── generator.py           # Local LLM generation via transformers
├── agents/
│   ├── base_agent.py          # Shared retrieve + generate logic
│   ├── router_agent.py        # Ties classifier -> agent dispatch together
│   ├── tech_support_agent.py
│   ├── hr_agent.py
│   └── sales_agent.py
└── data/
    ├── router_training_data.json   # Example queries per category, for training
    ├── tech_support_docs.txt
    ├── hr_docs.txt
    └── sales_docs.txt
```

## Frontend chatbot UI

A standalone, dependency-free chatbot page lives at `frontend/index.html`.
It's a "dispatch console" — alongside the chat, it shows the PyTorch
router's live confidence scores for each agent (so you can see *why* a
query got routed where it did) and the exact chunks retrieved from
ChromaDB for the answer.

**FastAPI serves it directly — everything runs on one port.**
Just start the API as usual:

```bash
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000/** in your browser — that's the chatbot
UI. The API endpoints (`/chat`, `/ingest`, `/agents`, `/docs`) are all still
available on the same port, e.g. `http://127.0.0.1:8000/docs` for the
Swagger UI.

No build step, no npm install, no separate frontend server. If you ever
want to host the frontend elsewhere (e.g. Netlify) while the API stays
local, `frontend/index.html` also works as a standalone file — just edit
the API URL field in the page's header, and CORS is already enabled on
the FastAPI side for that case.

## Extending this

- **Add a new agent:** create `agents/<name>_agent.py` subclassing `BaseAgent`,
  register it in `AGENT_REGISTRY` in `agents/router_agent.py`, add training
  examples for it in `data/router_training_data.json`, retrain the router,
  and ingest a knowledge base file for it.
- **Swap the LLM:** change `MODEL_NAME` in `llm/generator.py` to any
  `transformers`-compatible instruct model (bigger models need a GPU for
  reasonable latency).
- **Improve routing:** the router is a simple MLP on sentence embeddings.
  You can add more training examples, adjust `hidden_dim`, or replace it
  with an LLM-based classifier if you want zero-shot routing without
  retraining for new agents.



## 
