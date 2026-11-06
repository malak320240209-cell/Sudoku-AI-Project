"""CSP tests. Grouped as normal, edge and impossible, as the brief requires."""
import pytest

from sudoku.csp import ac3_consistent, conflict_pairs, count_solutions, solve, to_flat

PUZZLE = to_flat([
    [5, 3, 0, 0, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3], [4, 0, 0, 8, 0, 3, 0, 0, 1], [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0], [0, 0, 0, 4, 1, 9, 0, 0, 5], [0, 0, 0, 0, 8, 0, 0, 7, 9]])
CONFIGS = [(False, False), (True, False), (False, True), (True, True)]


def is_valid_solution(grid, puzzle):
    units = [grid[r * 9:r * 9 + 9] for r in range(9)]
    units += [grid[c::9] for c in range(9)]
    units += [[grid[(r + i) * 9 + c + j] for i in range(3) for j in range(3)]
              for r in (0, 3, 6) for c in (0, 3, 6)]
    return all(sorted(u) == list(range(1, 10)) for u in units) and \
        all(p in (0, g) for p, g in zip(puzzle, grid))


@pytest.mark.parametrize("use_fc,use_ac3", CONFIGS)
def test_normal_all_switch_settings_solve(use_fc, use_ac3):
    result = solve(PUZZLE, use_fc=use_fc, use_ac3=use_ac3, use_mrv=False)
    assert is_valid_solution(result.solution, PUZZLE)


def test_normal_configurations_agree():
    answers = {tuple(solve(PUZZLE, use_fc=f, use_ac3=a).solution) for f, a in CONFIGS}
    assert len(answers) == 1


def test_normal_inference_never_needs_more_assignments():
    plain = solve(PUZZLE, use_fc=False, use_ac3=False, use_mrv=False).counters
    fc = solve(PUZZLE, use_fc=True, use_ac3=False, use_mrv=False).counters
    full = solve(PUZZLE, use_fc=True, use_ac3=True, use_mrv=False).counters
    assert plain.assignments >= fc.assignments >= full.assignments


def test_normal_counters_move():
    counters = solve(PUZZLE).counters
    assert counters.assignments > 0 and counters.arc_revisions > 0 and counters.values_pruned > 0


def test_normal_unique_puzzle_counts_one():
    assert count_solutions(PUZZLE) == 1


def test_edge_empty_grid_has_many_solutions():
    assert count_solutions([0] * 81, limit=2) == 2


def test_edge_full_grid_is_its_own_solution():
    full = solve(PUZZLE).solution
    assert solve(full).solution == full


def test_edge_assignment_cap_reports_unknown_not_unsolvable():
    result = solve(PUZZLE, use_fc=False, use_ac3=False, use_mrv=False, max_assignments=1)
    assert result.capped and not result.solved


def test_impossible_equal_givens_in_a_row_detected_by_ac3():
    grid = [0] * 81
    grid[0] = grid[5] = 4
    assert conflict_pairs(grid) == [(0, 5)]
    assert not ac3_consistent(grid)


def test_impossible_no_duplicates_but_a_cell_has_no_value():
    # Row 1 holds 1-8 and column 9 holds a 9, so R1C9 has nothing left, with no duplicate anywhere.
    grid = [1, 2, 3, 4, 5, 6, 7, 8, 0] + [0] * 8 + [9] + [0] * 63
    assert not conflict_pairs(grid)
    for use_fc, use_ac3 in CONFIGS:
        assert not solve(grid, use_fc=use_fc, use_ac3=use_ac3).solved
