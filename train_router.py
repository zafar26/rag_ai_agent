"""
Train the router classifier.

Run this once before starting the API:
    python train_router.py

It reads example queries per category from data/router_training_data.json,
embeds them using the shared sentence-transformers model, and trains a
small PyTorch MLP to classify new queries into one of the categories.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split  # only used to split train/val

from models.router_classifier import RouterClassifier, save_model
from rag.embeddings import embed_texts, EMBEDDING_DIM

DATA_PATH = Path(__file__).parent / "data" / "router_training_data.json"
EPOCHS = 60
LEARNING_RATE = 1e-3


def load_training_data():
    with open(DATA_PATH, "r") as f:
        raw = json.load(f)

    labels = sorted(raw.keys())  # e.g. ["hr", "sales", "tech_support"]
    texts, y = [], []
    for label in labels:
        for query in raw[label]:
            texts.append(query)
            y.append(labels.index(label))

    return texts, np.array(y), labels


def main():
    print("Loading training data...")
    texts, y, labels = load_training_data()
    print(f"Loaded {len(texts)} examples across {len(labels)} categories: {labels}")

    print("Embedding queries with sentence-transformers (this downloads the "
          "embedding model on first run)...")
    X = embed_texts(texts)  # shape (N, 384)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    model = RouterClassifier(input_dim=EMBEDDING_DIM, num_classes=len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Training router classifier...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_preds = val_logits.argmax(dim=-1)
                val_acc = (val_preds == y_val_t).float().mean().item()
            print(f"Epoch {epoch:3d} | train loss: {loss.item():.4f} | val acc: {val_acc:.2%}")

    save_model(model, labels)
    print(f"\nSaved trained router to models/router_classifier.pt")
    print(f"Labels: {labels}")


if __name__ == "__main__":
    main()
