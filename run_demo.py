"""Integrated demo: image -> NN reading -> AC-3 -> search -> CSP -> solution, with plots.

Usage:
  python run_demo.py                      two misreads, A* search, unique goal
  python run_demo.py --misreads 3 --strategy ucs --seed 12
  python run_demo.py --impossible         shows the "no solution" report
Plots are saved to results/demo/.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sudoku.config import RESULTS, TEAM_SEED  # noqa: E402
from sudoku.csp import ac3_consistent, cell_name, conflict_pairs, solve  # noqa: E402
from sudoku.eval import BLUE, GREY, ORANGE, PALETTE, TEAL, write_csv  # noqa: E402
from sudoku.generator import make_puzzles  # noqa: E402
from sudoku.integration import read_mosaic, solve_reading  # noqa: E402
from sudoku.mosaic import load_pools, make_instance, render_mosaic  # noqa: E402
from sudoku.nn import DigitReader  # noqa: E402

CSP_CONFIGS = {"BT": (False, False), "BT+FC": (True, False), "BT+FC+AC-3": (True, True)}


def show_grid(grid, marks=()):
    for r in range(9):
        if r % 3 == 0 and r:
            print("  ------+-------+------")
        row = []
        for c in range(9):
            cell = r * 9 + c
            text = "." if not grid[cell] else str(grid[cell])
            row.append(text + ("*" if cell in marks else " "))
            if c in (2, 5):
                row.append("|")
        print("  " + " ".join(row).replace("  ", " "))


def heading(text):
    print(f"\n=== {text} ===")


def draw_digits(ax, grid, title, highlight=(), originals=None):
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    for i in range(10):
        width = 1.6 if i % 3 == 0 else 0.4
        ax.axhline(i, color="black", lw=width)
        ax.axvline(i, color="black", lw=width)
    for cell, value in enumerate(grid):
        if value:
            r, c = divmod(cell, 9)
            if cell in highlight:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color=ORANGE, alpha=0.35))
            colour = "black" if originals is None or originals[cell] else BLUE
            ax.text(c + 0.5, r + 0.55, value, ha="center", va="center", fontsize=9, color=colour)


def plot_pipeline(mosaic, reading, outcome, instance, path):
    conflicts = {c for pair in conflict_pairs(reading.grid) for c in pair}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(mosaic, cmap="gray_r")
    axes[0].set_title("1. Input image", fontsize=10)
    axes[0].axis("off")
    draw_digits(axes[1], reading.grid, "2. NN reading (duplicates shaded)", conflicts)
    confidence = np.where(np.array(reading.grid) > 0, reading.confidence, np.nan).reshape(9, 9)
    image = axes[2].imshow(confidence, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("3. NN confidence per given", fontsize=10)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(image, ax=axes[2], shrink=0.8)
    if outcome.solution:
        changed = [i for i in range(81) if reading.grid[i] != outcome.corrected_grid[i]]
        draw_digits(axes[3], outcome.solution, "4. CSP solution (corrected cells shaded)", changed,
                    instance.puzzle)
    else:
        axes[3].text(0.5, 0.5, "Unreadable", ha="center", va="center", fontsize=14)
        axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_trace(trace, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    orders = [t["order"] for t in trace]
    colours = [TEAL if t["goal"] else (BLUE if t["conflict_free"] else GREY) for t in trace]
    ax.scatter(orders, [t["g"] for t in trace], c=colours, s=18)
    for colour, label in ((GREY, "Has duplicates, CSP skipped"), (BLUE, "CSP rejected"),
                          (TEAL, "CSP accepted")):
        ax.scatter([], [], c=colour, s=18, label=label)
    ax.set_xlabel("Order visited by the search")
    ax.set_ylabel("Cost so far")
    ax.set_title("Search visits cheap correction sets, and the CSP confirms the first good one")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_csp(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    names = list(rows)
    for ax, key, title in zip(axes, ("assignments", "backtracks"), ("Assignments", "Backtracks")):
        values = [max(rows[n][key], 0.5) for n in names]
        ax.bar(names, values, color=PALETTE[:3])
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        for i, n in enumerate(names):
            ax.text(i, values[i], f"{rows[n][key]:,}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Same puzzle, same variable order, three CSP settings", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seed", type=int, default=TEAM_SEED)
    parser.add_argument("--misreads", type=int, default=2)
    parser.add_argument("--givens", type=int, default=30)
    parser.add_argument("--strategy", choices=["ucs", "greedy", "astar"], default="astar")
    parser.add_argument("--goal", choices=["solvable", "unique"], default="unique")
    parser.add_argument("--impossible", action="store_true", help="force an unfixable reading")
    args = parser.parse_args()
    out = RESULTS / "demo"
    out.mkdir(parents=True, exist_ok=True)

    misreads = 6 if args.impossible else args.misreads
    pools = load_pools()
    rng = np.random.default_rng(args.seed)
    puzzle, solution = make_puzzles(args.seed, 1, args.givens)[0]
    instance = make_instance(puzzle, solution, misreads, pools, rng)

    heading("1. Input: a generated puzzle rendered from MNIST digits")
    mosaic = render_mosaic(instance.images)
    print(f"Mosaic {mosaic.shape}, {args.givens} givens, seed {args.seed}. "
          f"Injected misreads: {[cell_name(c) for c in instance.misread_cells]}")

    heading("2. Neural network reads the image")
    reading = read_mosaic(mosaic, DigitReader())
    conflicts = {c for pair in conflict_pairs(reading.grid) for c in pair}
    show_grid(reading.grid, conflicts)
    print("(* marks a duplicate digit)")
    for c in reading.least_confident(3):
        p = reading.probs[c]
        top = np.argsort(-p)[:2]
        print(f"  {cell_name(c)}: {top[0] + 1} ({p[top[0]]:.0%}) vs {top[1] + 1} ({p[top[1]]:.0%})")

    heading("3. AC-3 checks the reading")
    print("Consistent." if ac3_consistent(reading.grid) else
          f"CONFLICT: a domain was emptied. Duplicate pairs: "
          f"{[(cell_name(a), cell_name(b)) for a, b in conflict_pairs(reading.grid)]}")

    heading(f"4. {args.strategy.upper()} search over corrections (goal: {args.goal}), NN ON")
    outcome = solve_reading(reading, args.strategy, args.goal, keep_trace=True)
    result = outcome.search
    print(f"Nodes expanded {result.expanded}, generated {result.generated}, "
          f"CSP calls {result.goal_tests}")
    for step in result.trace[:8] + result.trace[-2:] if len(result.trace) > 10 else result.trace:
        tried = [f"{cell_name(c)}={d}" for c, d in step["state"]] or ["(first reading)"]
        verdict = "ACCEPTED" if step["goal"] else ("CSP rejected" if step["conflict_free"]
                                                   else "has duplicates")
        print(f"  #{step['order']:<4} cost {step['g']:6.3f}  {', '.join(tried):<34} {verdict}")

    heading("5. Correction log")
    if outcome.status == "solved":
        for row in outcome.log:
            print(f"  {row['cell']}: read {row['read_digit']} -> {row['new_digit']} (cost {row['cost']})")
        if not outcome.log:
            print("  No corrections needed.")
    print(outcome.message)
    if outcome.log:
        write_csv(outcome.log, out / "correction_log.csv")

    if outcome.solution:
        heading("6. CSP solution")
        show_grid(outcome.solution, {i for i in range(81) if reading.grid[i] != outcome.corrected_grid[i]})
        print("Matches the true solution:", outcome.solution == solution)

    heading("7. CSP pruning on this puzzle (no MRV, so only inference differs)")
    target = outcome.corrected_grid or puzzle
    rows = {}
    for name, (use_fc, use_ac3) in CSP_CONFIGS.items():
        c = solve(target, use_fc=use_fc, use_ac3=use_ac3, use_mrv=False, max_assignments=500_000).counters
        rows[name] = {"assignments": c.assignments, "backtracks": c.backtracks}
        print(f"  {name:<11} assignments {c.assignments:>7,}  backtracks {c.backtracks:>7,}  "
              f"values pruned {c.values_pruned:>6,}")

    heading("8. System with the NN off (uniform costs, probabilities ignored)")
    off = solve_reading(reading, "ucs", args.goal, uniform=True)
    off_ok = off.solution == solution
    print(f"NN OFF: {off.status}, nodes expanded {off.search.expanded:,}, "
          f"CSP calls {off.search.goal_tests:,}, matches truth: {off_ok}")
    print(f"NN ON : {outcome.status}, nodes expanded {result.expanded:,}, "
          f"CSP calls {result.goal_tests:,}, matches truth: {outcome.solution == solution}")

    plot_pipeline(mosaic, reading, outcome, instance, out / "pipeline.png")
    if result.trace:
        plot_trace(result.trace, out / "search_trace.png")
    plot_csp(rows, out / "csp_pruning.png")
    print("\nPlots saved to results/demo")


if __name__ == "__main__":
    main()
