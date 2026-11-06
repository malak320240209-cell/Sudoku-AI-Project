"""End-to-end correction tests. Most use synthetic probabilities; the image test loads the included model."""
import numpy as np
import pytest

from sudoku.generator import make_puzzles
from sudoku.integration import (Reading, build_alternatives, read_mosaic, solve_randomly,
                                solve_reading)
from sudoku.mosaic import load_pools, make_instance, render_mosaic
from sudoku.nn import DigitReader


@pytest.fixture(scope="module")
def puzzle_pair():
    return make_puzzles(11, 1, 30)[0]


def reading_with_misreads(puzzle, misreads, confidence=0.55):
    """misreads: {cell: wrong digit}. Other givens are read at 0.999."""
    probs = np.zeros((81, 9))
    grid = list(puzzle)
    for cell, digit in enumerate(puzzle):
        if digit:
            probs[cell] = 0.001 / 8
            probs[cell, digit - 1] = 0.999
    for cell, wrong in misreads.items():
        true = puzzle[cell]
        probs[cell] = 0.001
        probs[cell, wrong - 1] = confidence
        probs[cell, true - 1] = 1 - confidence - 0.007
        grid[cell] = wrong
    return Reading(grid, probs)


def duplicating_digit(puzzle, cell):
    """A wrong digit for cell that duplicates another given in its row, column or box."""
    from sudoku.csp import PEERS
    return next(puzzle[p] for p in PEERS[cell] if puzzle[p] and puzzle[p] != puzzle[cell])


def givens(puzzle, n):
    return [i for i in range(81) if puzzle[i]][:n]


@pytest.mark.parametrize("strategy", ["ucs", "astar"])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_normal_misreads_are_recovered(puzzle_pair, strategy, n):
    puzzle, solution = puzzle_pair
    cells = givens(puzzle, 30)[::10][:n]
    misreads = {c: duplicating_digit(puzzle, c) for c in cells}
    outcome = solve_reading(reading_with_misreads(puzzle, misreads), strategy, goal="unique")
    assert outcome.status == "solved" and outcome.solution == solution


def test_normal_clean_reading_needs_no_correction(puzzle_pair):
    puzzle, solution = puzzle_pair
    outcome = solve_reading(reading_with_misreads(puzzle, {}))
    assert outcome.solution == solution and outcome.log == []


def test_normal_correction_log_fields(puzzle_pair):
    puzzle, _ = puzzle_pair
    cell = givens(puzzle, 1)[0]
    wrong = duplicating_digit(puzzle, cell)
    outcome = solve_reading(reading_with_misreads(puzzle, {cell: wrong}))
    assert outcome.conflict_detected
    row = outcome.log[0]
    assert (row["read_digit"], row["new_digit"]) == (wrong, puzzle[cell]) and row["cost"] >= 0


def test_normal_image_path_matches_pool_probabilities(puzzle_pair):
    """Rendering, cutting, the blank rule and the network together reproduce the stored reading."""
    puzzle, solution = puzzle_pair
    instance = make_instance(puzzle, solution, 3, load_pools(), np.random.default_rng(5))
    reading = read_mosaic(render_mosaic(instance.images), DigitReader())
    assert [d == 0 for d in reading.grid] == [d == 0 for d in puzzle]
    assert sum(a != b for a, b in zip(reading.grid, puzzle)) == 3
    assert np.allclose(reading.probs, instance.probs, atol=1e-5)


def test_edge_costs_are_never_negative(puzzle_pair):
    puzzle, _ = puzzle_pair
    cell = givens(puzzle, 1)[0]
    alternatives = build_alternatives(reading_with_misreads(puzzle, {cell: 1 + puzzle[cell] % 9}))
    assert all(cost >= 0 for options in alternatives.values() for _, cost in options)


def test_edge_tied_probabilities_cost_zero(puzzle_pair):
    puzzle, _ = puzzle_pair
    cell = givens(puzzle, 1)[0]
    reading = reading_with_misreads(puzzle, {cell: 1 + puzzle[cell] % 9}, confidence=0.4965)
    assert min(c for _, c in build_alternatives(reading)[cell]) == pytest.approx(0, abs=1e-6)


def test_edge_uniform_mode_ignores_probabilities(puzzle_pair):
    puzzle, _ = puzzle_pair
    alternatives = build_alternatives(reading_with_misreads(puzzle, {}), uniform=True)
    assert {c for options in alternatives.values() for _, c in options} == {1.0}


def test_edge_random_baseline_returns_counts(puzzle_pair):
    puzzle, solution = puzzle_pair
    found, tested, changes = solve_randomly(reading_with_misreads(puzzle, {}), 50, "solvable",
                                            np.random.default_rng(1))
    assert tested >= 1 and found == solution and changes == []


def test_edge_random_baseline_reports_its_corrections(puzzle_pair):
    puzzle, solution = puzzle_pair
    cell = givens(puzzle, 1)[0]
    reading = reading_with_misreads(puzzle, {cell: duplicating_digit(puzzle, cell)})
    found, _, changes = solve_randomly(reading, 20_000, "solvable", np.random.default_rng(1))
    assert found is not None and changes
    assert all(reading.grid[c] != d and cost >= 0 for c, d, cost in changes)


def test_impossible_duplicate_with_true_digit_not_offered(puzzle_pair, capsys):
    """Two confident equal digits whose true value has almost no probability cannot be fixed."""
    puzzle, _ = puzzle_pair
    cells = givens(puzzle, 8)
    misreads = {c: duplicating_digit(puzzle, c) for c in cells[:5]}
    reading = reading_with_misreads(puzzle, misreads, confidence=0.99)
    outcome = solve_reading(reading, "ucs")
    print(outcome.message)
    assert outcome.status == "unreadable" and outcome.solution is None
    assert "NO SOLUTION" in capsys.readouterr().out
    assert len(outcome.least_confident) == 5


def test_impossible_empty_alternatives_reports_unreadable():
    grid = [5, 5] + [0] * 79
    probs = np.zeros((81, 9))
    probs[0:2, 4] = 1.0
    outcome = solve_reading(Reading(grid, probs), "astar", max_corrections=0)
    assert outcome.status == "unreadable" and "NO SOLUTION" in outcome.message
