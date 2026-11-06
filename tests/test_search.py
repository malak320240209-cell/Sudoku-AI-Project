"""Search tests, including a small example with a hand-computed answer."""
import math

import numpy as np
import pytest

from sudoku.search import CorrectionProblem, search


def small_problem():
    """Cells 0 and 1 share a row and both read 5.

    Options: cell 0 -> 7 costs 1.0, cell 0 -> 3 costs 4.0, cell 1 -> 8 costs 2.0.
    Only the goal grid with cell 1 = 8 is accepted, so the known answer is {cell 1 -> 8}, cost 2.0.
    """
    grid = [5, 5] + [0] * 79
    alternatives = {0: [(7, 1.0), (3, 4.0)], 1: [(8, 2.0)]}
    return CorrectionProblem(grid, alternatives, lambda g: g[1] == 8)


@pytest.mark.parametrize("strategy", ["ucs", "astar", "greedy"])
def test_small_example_known_answer(strategy):
    result = search(small_problem(), strategy)
    assert result.found and result.corrections == ((1, 8, 2.0),)
    assert result.cost == 2.0


def test_ucs_counts_nodes_in_known_order():
    # Expands {} then {0->7}; pops {1->8} third and accepts it. Children of {} = 3, of {0->7} = 1.
    result = search(small_problem(), "ucs")
    assert result.expanded == 2 and result.generated == 5


def test_trace_records_visits_in_cost_order():
    trace = search(small_problem(), "ucs", keep_trace=True).trace
    costs = [row["g"] for row in trace]
    assert costs == sorted(costs) and trace[-1]["goal"]


def random_problem(rng):
    grid = [0] * 81
    grid[:4] = [5, 5, 6, 6]
    alternatives = {c: [(d, float(rng.uniform(0.1, 5))) for d in (1, 2, 3)] for c in range(4)}
    wanted = {int(rng.integers(0, 2)), int(rng.integers(2, 4))}
    goal = lambda g: all(g[c] in (1, 2, 3) for c in wanted)  # noqa: E731
    return CorrectionProblem(grid, alternatives, goal)


def test_astar_matches_ucs_cost_and_expands_no_more():
    rng = np.random.default_rng(1)
    for _ in range(50):
        problem = random_problem(rng)
        ucs, astar = search(problem, "ucs"), search(problem, "astar")
        assert ucs.found and astar.found
        assert math.isclose(ucs.cost, astar.cost)
        assert astar.expanded <= ucs.expanded


def test_heuristic_is_a_lower_bound():
    rng = np.random.default_rng(2)
    for _ in range(50):
        problem = random_problem(rng)
        assert problem.heuristic(frozenset()) <= search(problem, "ucs").cost + 1e-9


def test_heuristic_zero_on_a_conflict_free_state():
    problem = CorrectionProblem([1, 2] + [0] * 79, {0: [(3, 1.0)]}, lambda g: True)
    assert problem.heuristic(frozenset()) == 0


def test_correction_limit_is_respected():
    grid = [1, 1, 0, 2, 2, 0, 3, 3, 0] + [0] * 72
    alternatives = {c: [(9, 1.0)] for c in (0, 3, 6)}
    problem = CorrectionProblem(grid, alternatives, lambda g: True, max_corrections=2)
    assert problem.heuristic(frozenset()) == math.inf
    assert not search(problem, "ucs").found


def test_impossible_no_alternatives_gives_no_result():
    problem = CorrectionProblem([5, 5] + [0] * 79, {}, lambda g: True)
    for strategy in ("ucs", "astar", "greedy"):
        assert not search(problem, strategy).found
