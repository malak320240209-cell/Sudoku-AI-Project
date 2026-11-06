"""Calibration tools: expected calibration error and temperature scaling."""
import numpy as np


def softmax(logits, temperature=1.0):
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def fit_temperature(logits, y):
    """Temperature minimising validation negative log-likelihood, by grid then refinement."""
    grid = np.linspace(0.3, 5.0, 95)
    for _ in range(3):
        best = min(grid, key=lambda t: nll(softmax(logits, t), y))
        step = grid[1] - grid[0]
        grid = np.linspace(max(0.05, best - step), best + step, 21)
    return float(best)


def reliability(probs, y, bins=15):
    """Per-bin confidence, accuracy and count."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            rows.append((float(confidence[mask].mean()), float(correct[mask].mean()), int(mask.sum())))
    return rows


def ece(probs, y, bins=15):
    n = len(y)
    return sum(count / n * abs(acc - conf) for conf, acc, count in reliability(probs, y, bins))
