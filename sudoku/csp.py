"""Sudoku as a CSP. Backtracking, Forward Checking and AC-3 are independent switches.

Grids are flat lists of 81 ints, with 0 for blank. Domains are 9-bit masks (bit v-1 means value v),
which keeps the many CSP calls made by the search fast.
"""
from collections import deque
from dataclasses import dataclass, field

ALL_VALUES = 0b111111111


def _peers(cell):
    r, c = divmod(cell, 9)
    rows = [(r // 3) * 3 + i for i in range(3)]
    cols = [(c // 3) * 3 + j for j in range(3)]
    same = {r * 9 + j for j in range(9)} | {i * 9 + c for i in range(9)}
    same |= {i * 9 + j for i in rows for j in cols}
    same.discard(cell)
    return tuple(sorted(same))


PEERS = tuple(_peers(i) for i in range(81))
ALL_ARCS = tuple((xi, xj) for xi in range(81) for xj in PEERS[xi])


def to_flat(board):
    return [v for row in board for v in row]


def cell_name(cell):
    return f"R{cell // 9 + 1}C{cell % 9 + 1}"


def conflict_pairs(grid):
    """Pairs of peer cells holding the same digit."""
    return [(a, b) for a in range(81) if grid[a] for b in PEERS[a] if b > a and grid[a] == grid[b]]


@dataclass
class Counters:
    assignments: int = 0
    backtracks: int = 0
    arc_revisions: int = 0
    values_pruned: int = 0


@dataclass
class SolveResult:
    solutions: list = field(default_factory=list)
    counters: Counters = field(default_factory=Counters)
    capped: bool = False            # stopped at the assignment limit, so the answer is unknown

    @property
    def solved(self):
        return bool(self.solutions)

    @property
    def solution(self):
        return self.solutions[0] if self.solutions else None


def initial_domains(grid):
    return [(1 << (v - 1)) if v else ALL_VALUES for v in grid]


def revise(domains, xi, xj, counters):
    """Remove values of xi with no support in xj. Under !=, only a singleton xj removes anything."""
    counters.arc_revisions += 1
    dj = domains[xj]
    if dj & (dj - 1) == 0 and domains[xi] & dj:
        domains[xi] &= ~dj
        counters.values_pruned += 1
        return True
    return False


def ac3(domains, counters=None, arcs=ALL_ARCS):
    """Enforce arc consistency in place. Returns False when a domain is emptied."""
    counters = counters or Counters()
    queue = deque(arcs)
    while queue:
        xi, xj = queue.popleft()
        if revise(domains, xi, xj, counters):
            if not domains[xi]:
                return False
            queue.extend((xk, xi) for xk in PEERS[xi] if xk != xj)
    return True


def ac3_consistent(grid):
    """AC-3 conflict detector used at the start of the pipeline."""
    return ac3(initial_domains(grid))


class _Solver:
    def __init__(self, grid, use_fc, use_ac3, use_mrv, max_assignments, limit):
        self.grid = list(grid)
        self.use_fc, self.use_ac3, self.use_mrv = use_fc, use_ac3, use_mrv
        self.max_assignments, self.limit = max_assignments, limit
        self.result = SolveResult()
        self.domains = initial_domains(grid)
        self.values = list(grid)
        # The same static order for every configuration: blanks with the most filled peers first.
        blanks = [i for i in range(81) if not grid[i]]
        self.order = sorted(blanks, key=lambda i: (-sum(1 for p in PEERS[i] if grid[p]), i))

    def prepare(self):
        """Root step: returns False if the givens are already inconsistent."""
        if self.use_ac3:
            return ac3(self.domains, self.result.counters)
        if self.use_fc:
            return all(self._forward_check(cell, self.domains[cell])
                       for cell in range(81) if self.grid[cell])
        return not conflict_pairs(self.grid)

    def _forward_check(self, cell, bit):
        counters = self.result.counters
        for p in PEERS[cell]:
            if self.domains[p] & bit:
                self.domains[p] &= ~bit
                counters.values_pruned += 1
                if not self.domains[p]:
                    return False
        return True

    def _select(self, depth):
        if self.use_mrv:
            remaining = [i for i in self.order if not self.values[i]]
            return min(remaining, key=lambda i: self.domains[i].bit_count())
        return self.order[depth]

    def run(self, depth=0):
        """Returns True when the search should stop."""
        if depth == len(self.order):
            self.result.solutions.append(list(self.values))
            return len(self.result.solutions) >= self.limit
        counters = self.result.counters
        cell = self._select(depth)
        domain = self.domains[cell]
        for v in range(1, 10):
            bit = 1 << (v - 1)
            if not domain & bit:
                continue
            if not (self.use_fc or self.use_ac3) and any(self.values[p] == v for p in PEERS[cell]):
                continue
            if counters.assignments >= self.max_assignments:
                self.result.capped = True
                return True
            counters.assignments += 1
            saved = self.domains[:]
            self.values[cell] = v
            self.domains[cell] = bit
            ok = True
            if self.use_fc:
                ok = self._forward_check(cell, bit)
            if ok and self.use_ac3:
                ok = ac3(self.domains, counters, [(p, cell) for p in PEERS[cell]])
            if ok and self.run(depth + 1):
                return True
            self.values[cell] = 0
            self.domains = saved
            counters.backtracks += 1
        return False


def solve(grid, use_fc=True, use_ac3=True, use_mrv=True, max_assignments=2_000_000, limit=1):
    """Solve a flat grid. use_fc=use_ac3=False is plain backtracking.

    Experiment 2 sets use_mrv=False so only the inference differs between configurations.
    """
    solver = _Solver(grid, use_fc, use_ac3, use_mrv, max_assignments, limit)
    if solver.prepare():
        solver.run()
    return solver.result


def count_solutions(grid, limit=2):
    """Number of solutions, stopping at limit."""
    return len(solve(grid, limit=limit).solutions)
