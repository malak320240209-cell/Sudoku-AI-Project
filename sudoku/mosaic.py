"""Digit pools, misread injection and 9x9 image mosaics.

Injection rule (documented in DATA_DICTIONARY.md): a misread cell uses an image from the hard pool
of its true digit, meaning the network reads it as another digit with confidence below
HARD_CONFIDENCE. Every other given uses an image the network reads correctly, so a puzzle with k
injected misreads has exactly k. Blank cells hold faint noise, well below the brightness threshold.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import BLANK_THRESHOLD, HARD_CONFIDENCE, MIN_HARD_PER_DIGIT, MODELS

CELL = 28
LINE = 2
SIZE = 9 * CELL + 10 * LINE


def perturb(images, rng):
    """Rotate each image 20-35 degrees either way and blank a 7x7 patch. Used only to top up hard pools."""
    n = len(images)
    angle = np.deg2rad(rng.uniform(20, 35, n) * rng.choice([-1, 1], n))
    cos, sin = torch.tensor(np.cos(angle)).float(), torch.tensor(np.sin(angle)).float()
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
    x = torch.from_numpy(images).unsqueeze(1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, align_corners=False).squeeze(1).numpy().copy()
    for i in range(n):
        r, c = rng.integers(6, 15, 2)
        out[i, r:r + 7, c:c + 7] = 0
    return out


def build_pools(x_test, y_test, reader, seed):
    """Split test images by digit into clean (read correctly) and hard (low-confidence misreads)."""
    rng = np.random.default_rng(seed)
    probs = reader.probabilities(x_test)
    pred = probs.argmax(axis=1) + 1
    confidence = probs.max(axis=1)
    pool = {"summary": {}}
    for digit in range(1, 10):
        of_digit = np.flatnonzero(y_test == digit)
        clean = of_digit[pred[of_digit] == digit]
        hard = of_digit[(pred[of_digit] != digit) & (confidence[of_digit] < HARD_CONFIDENCE)]
        hard_x, hard_p, natural = x_test[hard], probs[hard], len(hard)
        attempts = 0
        while len(hard_x) < MIN_HARD_PER_DIGIT and attempts < 20:
            attempts += 1
            copies = perturb(x_test[rng.choice(clean, 400)], rng)
            p = reader.probabilities(copies)
            keep = (p.argmax(axis=1) + 1 != digit) & (p.max(axis=1) < HARD_CONFIDENCE)
            need = MIN_HARD_PER_DIGIT - len(hard_x)
            hard_x = np.concatenate([hard_x, copies[keep][:need]])
            hard_p = np.concatenate([hard_p, p[keep][:need]])
        pool[f"clean_x_{digit}"], pool[f"clean_p_{digit}"] = x_test[clean], probs[clean]
        pool[f"hard_x_{digit}"], pool[f"hard_p_{digit}"] = hard_x, hard_p
        pool["summary"][digit] = {"clean": len(clean), "hard_natural": natural,
                                  "hard_perturbed": len(hard_x) - natural}
    return pool


def save_pools(pool):
    arrays = {k: v for k, v in pool.items() if k != "summary"}
    np.savez_compressed(MODELS / "pools.npz", **arrays)


def load_pools():
    path = MODELS / "pools.npz"
    if not path.exists():
        raise FileNotFoundError("Digit pools missing. Run: python train.py")
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


@dataclass
class Instance:
    images: np.ndarray       # (81, 28, 28)
    probs: np.ndarray        # (81, 9), calibrated; zeros on blanks
    puzzle: list             # true givens
    solution: list
    misread_cells: list


def make_instance(puzzle, solution, k_misreads, pools, rng):
    """Assign an MNIST image to every given, injecting exactly k_misreads hard digits."""
    givens = [i for i in range(81) if puzzle[i]]
    misread = sorted(int(c) for c in rng.choice(givens, k_misreads, replace=False)) if k_misreads else []
    images = rng.uniform(0, 0.01, (81, CELL, CELL)).astype(np.float32)
    probs = np.zeros((81, 9), dtype=np.float64)
    for cell in givens:
        kind = "hard" if cell in misread else "clean"
        x, p = pools[f"{kind}_x_{puzzle[cell]}"], pools[f"{kind}_p_{puzzle[cell]}"]
        j = rng.integers(len(x))
        images[cell], probs[cell] = x[j], p[j]
    return Instance(images, probs, list(puzzle), list(solution), misread)


def render_mosaic(cells):
    """Paste 81 cells into one image with grid lines."""
    mosaic = np.ones((SIZE, SIZE), dtype=np.float32)
    for cell in range(81):
        r, c = divmod(cell, 9)
        top, left = LINE + r * (CELL + LINE), LINE + c * (CELL + LINE)
        mosaic[top:top + CELL, left:left + CELL] = cells[cell]
    return mosaic


def split_mosaic(mosaic):
    """Cut a mosaic back into its 81 cells using the known layout."""
    cells = np.zeros((81, CELL, CELL), dtype=np.float32)
    for cell in range(81):
        r, c = divmod(cell, 9)
        top, left = LINE + r * (CELL + LINE), LINE + c * (CELL + LINE)
        cells[cell] = mosaic[top:top + CELL, left:left + CELL]
    return cells


def is_blank(cells):
    """Brightness rule: mean pixel value below the threshold means blank."""
    return cells.mean(axis=(1, 2)) < BLANK_THRESHOLD
