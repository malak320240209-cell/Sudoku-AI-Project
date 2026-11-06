"""Connects the three parts: NN reads -> AC-3 finds conflict -> search corrects -> CSP confirms."""
from dataclasses import dataclass, field

import numpy as np

from .config import MAX_CORRECTIONS, TOP_ALTERNATIVES, UNCERTAIN_CELLS
from .csp import ac3_consistent, cell_name, conflict_pairs, count_solutions, solve
from .mosaic import is_blank, split_mosaic
from .search import CorrectionProblem, SearchResult, search

GOALS = {
    "solvable": lambda grid: solve(grid).solved,
    "unique": lambda grid: count_solutions(grid) == 1,
}
MIN_PROBABILITY = 1e-12


@dataclass
class Reading:
    grid: list            # the network's digit per cell, 0 for blank
    probs: np.ndarray     # (81, 9) calibrated probabilities

    @property
    def confidence(self):
        return self.probs.max(axis=1)

    def least_confident(self, n=5):
        filled = [i for i in range(81) if self.grid[i]]
        return sorted(filled, key=lambda i: self.confidence[i])[:n]


def reading_from_probs(probs, blank_mask):
    grid = [0 if blank_mask[i] else int(probs[i].argmax()) + 1 for i in range(81)]
    return Reading(grid, probs)


def read_mosaic(mosaic, reader):
    """The image path: cut the cells, find blanks by brightness, let the network read the rest."""
    cells = split_mosaic(mosaic)
    blank = is_blank(cells)
    probs = np.zeros((81, 9))
    if not blank.all():
        probs[~blank] = reader.probabilities(cells[~blank])
    return reading_from_probs(probs, blank)


def correction_cost(reading, cell, digit):
    """log P(read digit) - log P(new digit), never negative because the read digit is most probable."""
    p = np.clip(reading.probs[cell], MIN_PROBABILITY, 1.0)
    return float(max(0.0, np.log(p[reading.grid[cell] - 1]) - np.log(p[digit - 1])))


def correction_row(reading, cell, digit, cost):
    return {"cell": cell_name(cell), "read_digit": reading.grid[cell], "new_digit": digit,
            "cost": round(cost, 4)}


def build_alternatives(reading, uniform=False, n_cells=UNCERTAIN_CELLS, n_alternatives=TOP_ALTERNATIVES):
    """Uncertain cells are all conflict cells plus the least confident givens.

    Cost = log P(read digit) - log P(new digit), never negative because the read digit is the most
    probable. With uniform=True (NN off) the probabilities are ignored: every change costs 1 and
    every other digit is offered, in cell order.
    """
    grid = reading.grid
    conflict_cells = {c for pair in conflict_pairs(grid) for c in pair}
    if uniform:
        extra = [i for i in range(81) if grid[i] and i not in conflict_cells][:n_cells]
        return {c: [(d, 1.0) for d in range(1, 10) if d != grid[c]]
                for c in sorted(conflict_cells) + extra}
    cells = conflict_cells | set(reading.least_confident(n_cells))
    alternatives = {}
    for cell in sorted(cells):
        p = np.clip(reading.probs[cell], MIN_PROBABILITY, 1.0)
        ranked = [int(j) + 1 for j in np.argsort(-p) if int(j) + 1 != grid[cell]]
        alternatives[cell] = [(d, correction_cost(reading, cell, d)) for d in ranked[:n_alternatives]]
    return alternatives


@dataclass
class Outcome:
    status: str                         # "solved" or "unreadable"
    solution: list | None = None
    corrected_grid: list | None = None
    log: list = field(default_factory=list)     # correction log rows
    search: SearchResult = field(default_factory=SearchResult)
    conflict_detected: bool = False
    least_confident: list = field(default_factory=list)
    message: str = ""


def solve_reading(reading, strategy="ucs", goal="solvable", uniform=False,
                  max_corrections=MAX_CORRECTIONS, keep_trace=False):
    """Run the whole correction loop on a reading."""
    conflict = not ac3_consistent(reading.grid)
    problem = CorrectionProblem(reading.grid, build_alternatives(reading, uniform), GOALS[goal],
                                max_corrections)
    result = search(problem, strategy, keep_trace=keep_trace)
    if not result.found:
        cells = reading.least_confident()
        names = ", ".join(f"{cell_name(c)} ({reading.confidence[c]:.2f})" for c in cells)
        return Outcome("unreadable", search=result, conflict_detected=conflict, least_confident=cells,
                       message=f"NO SOLUTION: puzzle is unreadable within {max_corrections} "
                               f"corrections. Least confident cells: {names}")
    corrected = list(reading.grid)
    log = []
    for cell, digit, cost in result.corrections:
        log.append(correction_row(reading, cell, digit, cost))
        corrected[cell] = digit
    return Outcome("solved", solve(corrected).solution, corrected, log, result, conflict,
                   message=f"Solved with {len(log)} correction(s), total cost {result.cost:.3f}")


def solve_randomly(reading, budget, goal, rng, max_corrections=MAX_CORRECTIONS):
    """Baseline: try random sets of corrections to random givens, with no use of probabilities.

    Returns (solution, states tested, corrections). Costs are computed afterwards for the log only.
    """
    filled = [i for i in range(81) if reading.grid[i]]
    tested = 0
    for attempt in range(budget):
        grid = list(reading.grid)
        size = 0 if attempt == 0 else rng.integers(1, max_corrections + 1)  # the reading itself first
        for cell in rng.choice(filled, size, replace=False):
            grid[cell] = int(rng.integers(1, 10))
        if conflict_pairs(grid):
            continue
        tested += 1
        if GOALS[goal](grid):
            changed = [(i, grid[i], correction_cost(reading, i, grid[i]))
                       for i in filled if grid[i] != reading.grid[i]]
            return solve(grid).solution, tested, changed
    return None, tested, []


def as_rows(grid):
    return [grid[i:i + 9] for i in range(0, 81, 9)]
