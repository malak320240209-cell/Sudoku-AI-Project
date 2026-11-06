"""Run the three experiments across five seeds. Tables and plots go to results/.

Usage: python run_experiments.py [--quick]
"""
import argparse
import json
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch

from sudoku.config import MAX_SEARCH_NODES, RESULTS, TEAM_SEED
from sudoku.csp import ac3_consistent, solve
from sudoku.eval import line_plot, markdown_table, recovered, summarise, write_csv
from sudoku.generator import make_puzzles
from sudoku.integration import correction_row, read_mosaic, solve_randomly, solve_reading
from sudoku.mosaic import load_pools, make_instance, render_mosaic
from sudoku.nn import DigitReader

SEEDS = [TEAM_SEED + i for i in range(5)]
GIVENS_DEFAULT = 30
CSP_CONFIGS = {"BT": (False, False), "BT+FC": (True, False), "BT+FC+AC-3": (True, True)}
BT_CAP = 500_000

POOLS = None
READER = None


def _init():
    global POOLS, READER
    torch.set_num_threads(1)  # one process per core already
    POOLS = load_pools()
    READER = DigitReader()


def _reading(instance):
    """Read the puzzle from its image, exactly as the demo does."""
    return read_mosaic(render_mosaic(instance.images), READER)


def log_rows(base, instance, reading, solution, corrections):
    """One row per correction on an answered puzzle. A puzzle needing none gets a single 'none' row."""
    base = dict(base, recovered=recovered(solution, instance.solution))
    if not corrections:
        return [dict(base, cell="none", read_digit="", new_digit="", cost=0.0, true_digit="")]
    return [dict(base, **correction_row(reading, cell, digit, cost), true_digit=instance.puzzle[cell])
            for cell, digit, cost in corrections]


def exp1_task(args):
    """UCS vs Greedy vs A*, under the 'solvable' and 'unique' goals, at one misread level and seed."""
    seed, k, n = args
    rows = []
    rng = np.random.default_rng([seed, 1, k])
    puzzles = make_puzzles(seed * 10 + k, n, GIVENS_DEFAULT)
    instances = [make_instance(p, s, k, POOLS, rng) for p, s in puzzles]
    readings = [_reading(instance) for instance in instances]
    log = []
    for strategy in ("ucs", "greedy", "astar"):
        for goal in ("solvable", "unique"):
            stats = dict(expanded=[], generated=[], goal_tests=[], recovered=[], answered=[],
                         seconds=[], cost=[])
            for index, (instance, reading) in enumerate(zip(instances, readings)):
                start = time.perf_counter()
                outcome = solve_reading(reading, strategy, goal)
                stats["seconds"].append(time.perf_counter() - start)
                stats["expanded"].append(outcome.search.expanded)
                stats["generated"].append(outcome.search.generated)
                stats["goal_tests"].append(outcome.search.goal_tests)
                stats["recovered"].append(recovered(outcome.solution, instance.solution))
                stats["answered"].append(outcome.status == "solved")
                if outcome.status == "solved":
                    stats["cost"].append(outcome.search.cost)
                    base = {"experiment": 1, "method": strategy, "goal": goal, "seed": seed,
                            "misreads": k, "puzzle": index}
                    log += log_rows(base, instance, reading, outcome.solution, outcome.search.corrections)
            row = {k2: float(np.mean(v)) if v else 0.0 for k2, v in stats.items()}
            row["recovery_pct"] = 100 * row.pop("recovered")
            row["answered_pct"] = 100 * row.pop("answered")
            row["wrong_answer_pct"] = row["answered_pct"] - row["recovery_pct"]
            row["milliseconds"] = 1000 * row.pop("seconds")
            rows.append({"seed": seed, "misreads": k, "strategy": strategy, "goal": goal, **row})
    return rows, log


def exp2_task(args):
    """BT vs BT+FC vs BT+FC+AC-3 on correctly read puzzles with 25, 30 or 35 givens."""
    seed, givens, n = args
    puzzles = make_puzzles(seed * 100 + givens, n, givens)
    rows = []
    for name, (use_fc, use_ac3) in CSP_CONFIGS.items():
        assignments, backtracks, pruned, seconds, capped = [], [], [], [], 0
        for puzzle, solution in puzzles:
            start = time.perf_counter()
            result = solve(puzzle, use_fc=use_fc, use_ac3=use_ac3, use_mrv=False,
                           max_assignments=BT_CAP)
            seconds.append(time.perf_counter() - start)
            assignments.append(result.counters.assignments)
            backtracks.append(result.counters.backtracks)
            pruned.append(result.counters.values_pruned)
            capped += result.capped
            assert result.capped or result.solution == solution
        rows.append({"seed": seed, "givens": givens, "config": name,
                     "assignments": float(np.mean(assignments)),
                     "backtracks": float(np.mean(backtracks)),
                     "values_pruned": float(np.mean(pruned)),
                     "milliseconds": 1000 * float(np.mean(seconds)),
                     "capped_pct": 100 * capped / n})
    return rows


def exp3_task(args):
    """NN-guided (A*, unique goal) vs no correction vs random correction with the same budget."""
    seed, k, n = args
    rng = np.random.default_rng([seed, 3, k])
    puzzles = make_puzzles(seed * 1000 + k, n, GIVENS_DEFAULT)
    instances = [make_instance(p, s, k, POOLS, rng) for p, s in puzzles]
    tallies = {name: {"recovered": 0, "answered": 0} for name in ("NN-guided", "No correction", "Random")}
    log = []
    for index, instance in enumerate(instances):
        reading = _reading(instance)
        guided = solve_reading(reading, "astar", "unique")
        random_solution, _, random_changes = solve_randomly(reading, max(guided.search.generated, 1),
                                                            "unique", rng)
        answers = {
            "NN-guided": ("unique", guided.solution, guided.search.corrections),
            "No correction": ("none", solve(reading.grid).solution, []),
            "Random": ("unique", random_solution, random_changes),
        }
        for name, (goal, solution, corrections) in answers.items():
            tallies[name]["recovered"] += recovered(solution, instance.solution)
            tallies[name]["answered"] += solution is not None
            if solution is not None:
                base = {"experiment": 3, "method": name, "goal": goal, "seed": seed, "misreads": k,
                        "puzzle": index}
                log += log_rows(base, instance, reading, solution, corrections)
    rows = [{"seed": seed, "misreads": k, "method": name,
             "recovery_pct": 100 * t["recovered"] / n, "answered_pct": 100 * t["answered"] / n,
             "wrong_answer_pct": 100 * (t["answered"] - t["recovered"]) / n}
            for name, t in tallies.items()]
    return rows, log


def misread_profile(n=200):
    """How often an injected misread is visible as a conflict, and whether its digit can be reached."""
    rng = np.random.default_rng(TEAM_SEED)
    visible = 0
    puzzles = make_puzzles(TEAM_SEED, n, GIVENS_DEFAULT)
    for puzzle, solution in puzzles:
        instance = make_instance(puzzle, solution, 1, POOLS, rng)
        visible += not ac3_consistent(_reading(instance).grid)
    pool_images = [POOLS[f"{kind}_x_{d}"] for kind in ("clean", "hard") for d in range(1, 10)]
    dimmest = min(float(images.mean(axis=(1, 2)).min()) for images in pool_images)
    true_rank = []
    for digit in range(1, 10):
        probs = POOLS[f"hard_p_{digit}"]
        for p in probs:
            true_rank.append(int((-p).argsort().tolist().index(digit - 1)))
    ranks = np.array(true_rank)
    return {"one_misread_visible_to_ac3_pct": 100 * visible / n,
            "dimmest_pool_image_brightness": dimmest,
            "hard_digits": len(ranks),
            "true_digit_second_pct": float(100 * (ranks == 1).mean()),
            "true_digit_top4_pct": float(100 * (ranks <= 3).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="10 puzzles per setting, for a check run")
    args = parser.parse_args()
    n1 = 10 if args.quick else 100
    n2 = 4 if args.quick else 30
    out = RESULTS / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with Pool(max(cpu_count() - 2, 1), initializer=_init) as pool:
        outputs1 = pool.map(exp1_task, [(s, k, n1) for s in SEEDS for k in range(5)], chunksize=1)
        rows1 = [row for rows, _ in outputs1 for row in rows]
        print(f"Experiment 1 done at {time.time() - started:.0f}s", flush=True)
        rows2 = sum(pool.map(exp2_task, [(s, g, n2) for s in SEEDS for g in (25, 30, 35)], chunksize=1), [])
        print(f"Experiment 2 done at {time.time() - started:.0f}s", flush=True)
        outputs3 = pool.map(exp3_task, [(s, k, n1) for s in SEEDS for k in range(5)], chunksize=1)
        rows3 = [row for rows, _ in outputs3 for row in rows]
        print(f"Experiment 3 done at {time.time() - started:.0f}s", flush=True)

    correction_log = [row for _, log in outputs1 + outputs3 for row in log]
    _init()
    profile = misread_profile(50 if args.quick else 200)

    metrics1 = ["recovery_pct", "wrong_answer_pct", "answered_pct", "expanded", "generated",
                "goal_tests", "milliseconds", "cost"]
    table1 = summarise(rows1, ["misreads", "strategy", "goal"], metrics1)
    metrics2 = ["assignments", "backtracks", "values_pruned", "milliseconds", "capped_pct"]
    table2 = summarise(rows2, ["givens", "config"], metrics2)
    metrics3 = ["recovery_pct", "wrong_answer_pct", "answered_pct"]
    table3 = summarise(rows3, ["misreads", "method"], metrics3)

    write_csv(rows1, out / "exp1_runs.csv")
    write_csv(rows2, out / "exp2_runs.csv")
    write_csv(rows3, out / "exp3_runs.csv")
    write_csv(correction_log, out / "correction_log.csv")
    write_csv(table1, out / "exp1_summary.csv")
    write_csv(table2, out / "exp2_summary.csv")
    write_csv(table3, out / "exp3_summary.csv")

    unique1 = [dict(r, name=r["strategy"].upper()) for r in table1 if r["goal"] == "unique"]
    line_plot(unique1, "misreads", "name", "recovery_pct", out / "exp1_recovery.png",
              "Recovery falls past the 3-correction limit", "Injected misreads",
              "Exact recovery (%)")
    line_plot(unique1, "misreads", "name", "expanded", out / "exp1_nodes.png",
              "A* expands fewer nodes than UCS", "Injected misreads", "Nodes expanded (mean)",
              log_scale=True)
    goals = [dict(r, name=f"A*, {r['goal']} goal") for r in table1 if r["strategy"] == "astar"]
    line_plot(goals, "misreads", "name", "wrong_answer_pct", out / "exp1_wrong_answers.png",
              "Wrong answers by goal type", "Injected misreads", "Wrong answers (%)")
    line_plot(table2, "givens", "config", "assignments", out / "exp2_assignments.png",
              "Inference cuts the assignments needed", "Givens", "Assignments (mean)",
              log_scale=True)
    line_plot(table2, "givens", "config", "backtracks", out / "exp2_backtracks.png",
              "Backtracks by CSP configuration", "Givens", "Backtracks (mean)", log_scale=True)
    line_plot(table3, "misreads", "method", "recovery_pct", out / "exp3_recovery.png",
              "NN-guided correction against the baselines", "Injected misreads",
              "Exact recovery (%)")

    report = ["# Experiment results", f"Seeds: {SEEDS}. Puzzles per setting per seed: {n1} "
              f"(Experiments 1 and 3), {n2} (Experiment 2). Values are mean ± standard deviation "
              f"across seeds. Plain BT cap: {BT_CAP:,} assignments. Search node limit: "
              f"{MAX_SEARCH_NODES:,}.", "",
              "## Experiment 1: UCS vs Greedy vs A*",
              markdown_table(table1, ["misreads", "strategy", "goal"], metrics1), "",
              "## Experiment 2: BT vs BT+FC vs BT+FC+AC-3",
              markdown_table(table2, ["givens", "config"], metrics2), "",
              "## Experiment 3: NN-guided vs no correction vs random",
              markdown_table(table3, ["misreads", "method"], metrics3), "",
              "## Misread profile", json.dumps(profile, indent=2)]
    (out / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    (out / "misread_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"Finished in {time.time() - started:.0f}s. See {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
