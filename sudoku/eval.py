"""Metrics tables and plots shared by training, experiments and the demo."""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .calibration import ece, reliability  # noqa: E402

BLUE, ORANGE, TEAL, GREY = "#2a6fb0", "#d9792b", "#2a9d8f", "#8a8f98"
PALETTE = [BLUE, ORANGE, TEAL, GREY]


def recovered(solution, truth):
    return solution is not None and list(solution) == list(truth)


def plot_confusion(probs, y, path, name):
    """Save the confusion matrix for digits 1-9 and return it as nested lists."""
    matrix = np.zeros((9, 9), dtype=int)
    np.add.at(matrix, (y, probs.argmax(axis=1)), 1)
    off_diagonal = matrix.astype(float)
    np.fill_diagonal(off_diagonal, np.nan)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    image = ax.imshow(off_diagonal, cmap="Oranges")
    for i in range(9):
        for j in range(9):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=7,
                    color="black" if i != j else BLUE)
    ax.set_xticks(range(9), range(1, 10))
    ax.set_yticks(range(9), range(1, 10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(f"{name}: {len(y) - int(np.trace(matrix)):,} errors in {len(y):,} test images")
    fig.colorbar(image, label="Misclassified images", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return matrix.tolist()


def plot_reliability(models, y, path):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0.5, 1], [0.5, 1], color=GREY, linestyle="--", label="Perfect calibration")
    for (name, probs), colour in zip(models.items(), PALETTE):
        rows = [r for r in reliability(probs, y) if r[0] >= 0.5]
        ax.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", ms=3, color=colour,
                label=f"{name} (ECE {ece(probs, y):.4f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability on the test set")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarise(rows, keys, metrics):
    """Group per-seed rows by keys and return mean and sample standard deviation of each metric."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    table = []
    for key, members in groups.items():
        out = dict(zip(keys, key))
        out["seeds"] = len(members)
        for metric in metrics:
            values = np.array([m[metric] for m in members], dtype=float)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        table.append(out)
    return table


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, keys, metrics, digits=2):
    header = keys + metrics
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [str(row[k]) for k in keys]
        cells += [f"{row[f'{m}_mean']:.{digits}f} ± {row[f'{m}_std']:.{digits}f}" for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def line_plot(table, x_key, series_key, metric, path, title, xlabel, ylabel, log_scale=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(dict.fromkeys(r[series_key] for r in table))
    colours = (PALETTE * 3)[:len(names)]
    for name, colour in zip(names, colours):
        rows = sorted((r for r in table if r[series_key] == name), key=lambda r: r[x_key])
        ax.errorbar([r[x_key] for r in rows], [r[f"{metric}_mean"] for r in rows],
                    yerr=[r[f"{metric}_std"] for r in rows], marker="o", ms=4, capsize=3,
                    color=colour, label=str(name))
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(sorted({r[x_key] for r in table}))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
