"""Correction search. A state is the set of corrections applied to the first reading.

UCS orders by path cost g, Greedy by the heuristic h, and A* by g + h. Costs depend only on the set
of corrections, so each state has one g and duplicate states can be dropped when generated.
"""
import heapq
import itertools
import math
from dataclasses import dataclass, field

from .config import MAX_CORRECTIONS, MAX_SEARCH_NODES
from .csp import PEERS


@dataclass
class SearchResult:
    found: bool = False
    corrections: tuple = ()      # (cell, new_digit, cost) in cell order
    cost: float = 0.0
    expanded: int = 0
    generated: int = 0
    goal_tests: int = 0          # states that passed the duplicate filter and reached the CSP
    hit_node_limit: bool = False
    trace: list = field(default_factory=list)


class CorrectionProblem:
    """grid: first reading. alternatives: {cell: [(digit, cost), ...]}. goal: grid -> bool."""

    def __init__(self, grid, alternatives, goal, max_corrections=MAX_CORRECTIONS):
        self.grid = list(grid)
        self.alternatives = alternatives
        self.goal = goal
        self.max_corrections = max_corrections
        self.costs = {(c, d): cost for c, opts in alternatives.items() for d, cost in opts}
        self.cheapest = {c: min(cost for _, cost in opts) for c, opts in alternatives.items() if opts}
        filled = [i for i in range(81) if grid[i]]
        self.peer_pairs = [(a, b) for a in filled for b in PEERS[a] if b > a and grid[b]]

    def apply(self, state):
        grid = list(self.grid)
        for cell, digit in state:
            grid[cell] = digit
        return grid

    def conflicts(self, grid):
        return [(a, b) for a, b in self.peer_pairs if grid[a] == grid[b]]

    def heuristic(self, state):
        """Lower bound on the remaining cost.

        Every duplicate pair needs at least one of its two cells changed. One correction changes
        one cell, so disjoint pairs need separate corrections, and their cheapest costs can be
        summed. Infinite when a pair cannot be fixed or the correction budget would be exceeded.
        """
        done = {cell for cell, _ in state}
        used, pairs, total = set(), 0, 0.0
        for a, b in self.conflicts(self.apply(state)):
            options = [self.cheapest[c] for c in (a, b) if c not in done and c in self.cheapest]
            if not options:
                return math.inf
            if a in used or b in used:
                continue
            used.update((a, b))
            pairs += 1
            total += min(options)
        return math.inf if pairs > self.max_corrections - len(state) else total

    def successors(self, state):
        if len(state) >= self.max_corrections:
            return
        done = {cell for cell, _ in state}
        for cell, options in self.alternatives.items():
            if cell not in done:
                for digit, cost in options:
                    yield state | {(cell, digit)}, cost


def _priority(strategy, g, h, tie):
    if strategy == "ucs":
        return (g, tie)
    if strategy == "astar":
        return (g + h, tie)
    if strategy == "greedy":
        return (h, g, tie)
    raise ValueError(f"Unknown strategy {strategy}")


def search(problem, strategy="ucs", max_nodes=MAX_SEARCH_NODES, keep_trace=False):
    result = SearchResult()
    tie = itertools.count()
    informed = strategy != "ucs"
    start = frozenset()
    h0 = problem.heuristic(start) if informed else 0.0
    if h0 == math.inf:
        return result
    frontier = [(_priority(strategy, 0.0, h0, next(tie)), 0.0, start)]
    seen = {start}
    result.generated = 1

    while frontier:
        _, g, state = heapq.heappop(frontier)
        grid = problem.apply(state)
        conflict_free = not problem.conflicts(grid)
        accepted = False
        if conflict_free:
            result.goal_tests += 1
            accepted = problem.goal(grid)
        if keep_trace:
            result.trace.append({"order": len(result.trace) + 1, "state": sorted(state), "g": g,
                                 "conflict_free": conflict_free, "goal": accepted})
        if accepted:
            result.found = True
            result.cost = g
            result.corrections = tuple(sorted((c, d, problem.costs[(c, d)]) for c, d in state))
            return result
        result.expanded += 1
        for child, step in problem.successors(state):
            if child in seen:
                continue
            if result.generated >= max_nodes:
                result.hit_node_limit = True
                return result
            h = problem.heuristic(child) if informed else 0.0
            seen.add(child)
            result.generated += 1
            if h != math.inf:
                child_g = g + step
                heapq.heappush(frontier, (_priority(strategy, child_g, h, next(tie)), child_g, child))
    return result
