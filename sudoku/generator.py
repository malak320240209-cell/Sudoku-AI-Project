"""Seeded puzzle generator: full grid by randomised backtracking, then remove cells while unique."""
import numpy as np

from .csp import PEERS, count_solutions


def full_grid(rng):
    """A random complete valid grid, using fewest-candidates-first randomised backtracking."""
    grid = [0] * 81

    def candidates(cell):
        return set(range(1, 10)) - {grid[p] for p in PEERS[cell]}

    def fill():
        blanks = [i for i in range(81) if not grid[i]]
        if not blanks:
            return True
        options = {i: candidates(i) for i in blanks}
        cell = min(blanks, key=lambda i: len(options[i]))
        for v in rng.permutation(sorted(options[cell])):
            grid[cell] = int(v)
            if fill():
                return True
            grid[cell] = 0
        return False

    fill()
    return grid


def make_puzzle(rng, n_givens):
    """Return (puzzle, solution) with exactly n_givens givens and a unique solution."""
    while True:
        solution = full_grid(rng)
        puzzle = list(solution)
        filled = 81
        for cell in rng.permutation(81):
            if filled == n_givens:
                break
            kept = puzzle[cell]
            puzzle[cell] = 0
            if count_solutions(puzzle) == 1:
                filled -= 1
            else:
                puzzle[cell] = kept
        if filled == n_givens:
            return puzzle, solution


def make_puzzles(seed, n_puzzles, n_givens):
    rng = np.random.default_rng(seed)
    return [make_puzzle(rng, n_givens) for _ in range(n_puzzles)]
