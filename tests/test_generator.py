import numpy as np

from sudoku.csp import count_solutions
from sudoku.generator import full_grid, make_puzzles
from sudoku.mosaic import CELL, is_blank, render_mosaic, split_mosaic


def test_same_seed_same_puzzles():
    assert make_puzzles(5, 2, 30) == make_puzzles(5, 2, 30)


def test_different_seed_different_puzzles():
    assert make_puzzles(5, 1, 30) != make_puzzles(6, 1, 30)


def test_puzzles_have_exact_givens_and_one_solution():
    for n in (25, 30, 35):
        for puzzle, solution in make_puzzles(9, 2, n):
            assert sum(1 for v in puzzle if v) == n
            assert count_solutions(puzzle) == 1
            assert all(p in (0, s) for p, s in zip(puzzle, solution))


def test_full_grid_has_no_blanks():
    assert 0 not in full_grid(np.random.default_rng(1))


def test_mosaic_round_trip():
    cells = np.random.default_rng(1).random((81, CELL, CELL)).astype(np.float32)
    assert np.array_equal(split_mosaic(render_mosaic(cells)), cells)


def test_blank_rule_separates_noise_from_ink():
    rng = np.random.default_rng(1)
    cells = rng.uniform(0, 0.01, (2, CELL, CELL)).astype(np.float32)
    cells[1, 4:24, 13:15] = 1.0     # a thin stroke like a handwritten 1
    assert is_blank(cells).tolist() == [True, False]
