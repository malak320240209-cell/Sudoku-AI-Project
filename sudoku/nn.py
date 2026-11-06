"""Digit classifier (1-9) and helpers for logits and probabilities."""
import json

import numpy as np
import torch
from torch import nn

from .calibration import softmax
from .config import MODELS


class DigitCNN(nn.Module):
    """Two convolution blocks and a dropout classifier. Class index = digit - 1."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 9))

    def forward(self, x):
        return self.classifier(self.features(x))


def _tensor(images):
    return torch.from_numpy(np.asarray(images, dtype=np.float32)).unsqueeze(1)


@torch.no_grad()
def predict_logits(model, images, batch_size=1024):
    model.eval()
    x = _tensor(images)
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]).numpy()


def train_model(data, epochs=8, batch_size=128, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    model = DigitCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    x, y = _tensor(data["x_train"]), torch.from_numpy(data["y_train"] - 1)
    best_acc, best_state, history = 0.0, None, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), batch_size):
            batch = order[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        val_pred = predict_logits(model, data["x_val"]).argmax(axis=1)
        val_acc = float((val_pred == data["y_val"] - 1).mean())
        history.append({"epoch": epoch + 1, "train_loss": total / len(x), "val_accuracy": val_acc})
        print(f"epoch {epoch + 1}: loss {total / len(x):.4f}, validation accuracy {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


class DigitReader:
    """Loads the trained network and its temperature, and returns calibrated probabilities."""

    def __init__(self):
        path = MODELS / "cnn.pt"
        if not path.exists():
            raise FileNotFoundError("Trained model missing. Run: python train.py")
        self.model = DigitCNN()
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.temperature = json.loads((MODELS / "calibration.json").read_text())["temperature"]

    def probabilities(self, images):
        return softmax(predict_logits(self.model, images), self.temperature)
