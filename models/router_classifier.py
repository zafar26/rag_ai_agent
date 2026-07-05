"""
PyTorch classifier that decides which specialized agent should handle
a given user query.

Input:  a 384-dim sentence embedding of the user's query
Output: a probability distribution over agent categories
        (tech_support, hr, sales)

This is intentionally a small MLP -- the routing task is simple
(it's a text classification problem over a handful of classes), so a
heavy model isn't warranted. This keeps training fast (seconds, on CPU)
while still being a genuine, trainable PyTorch model rather than a
hardcoded if/else.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).parent
WEIGHTS_PATH = MODEL_DIR / "router_classifier.pt"
LABELS_PATH = MODEL_DIR / "router_labels.json"


class RouterClassifier(nn.Module):
    """Simple feed-forward network: embedding -> hidden -> class logits."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_model(model: RouterClassifier, label_list: List[str]) -> None:
    """Persist trained weights and the label ordering to disk."""
    torch.save(model.state_dict(), WEIGHTS_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(label_list, f)


def load_model() -> Tuple[RouterClassifier, List[str]]:
    """Load trained weights + label ordering. Raises if not trained yet."""
    if not WEIGHTS_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Router classifier not trained yet. Run `python train_router.py` first."
        )
    with open(LABELS_PATH, "r") as f:
        label_list = json.load(f)

    model = RouterClassifier(num_classes=len(label_list))
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model, label_list


def predict_label(model: RouterClassifier, label_list: List[str], embedding) -> Dict:
    """
    Run inference for a single query embedding.
    Returns the predicted label plus per-class confidence scores.
    """
    with torch.no_grad():
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

    scored = {label_list[i]: round(probs[i], 4) for i in range(len(label_list))}
    predicted = max(scored, key=scored.get)
    return {"predicted_agent": predicted, "confidence_scores": scored}
