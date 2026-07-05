"""
Ingest the sample knowledge base .txt files into ChromaDB, one
collection per agent.

Run once before starting the API:
    python ingest.py
"""

from pathlib import Path

from rag.vector_store import add_documents

DATA_DIR = Path(__file__).parent / "data"

AGENT_DOC_FILES = {
    "tech_support": DATA_DIR / "tech_support_docs.txt",
    "hr": DATA_DIR / "hr_docs.txt",
    "sales": DATA_DIR / "sales_docs.txt",
}


def main():
    for agent_name, file_path in AGENT_DOC_FILES.items():
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        count = add_documents(agent_name, lines)
        print(f"Ingested {count} documents into '{agent_name}' collection.")

    print("\nIngestion complete. You can now start the API with:")
    print("    uvicorn main:app --reload")


if __name__ == "__main__":
    main()
