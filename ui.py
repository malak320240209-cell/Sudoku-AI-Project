"""Local test UI for the Sudoku reader and solver.

Usage: python -m streamlit run ui.py
It runs the real pipeline and never writes to results/, models/ or report/.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_demo import CSP_CONFIGS, draw_digits  # noqa: E402
from sudoku.config import MAX_CORRECTIONS, RESULTS, TEAM_SEED  # noqa: E402
from sudoku.csp import ac3_consistent, cell_name, conflict_pairs, solve  # noqa: E402
from sudoku.generator import make_puzzles  # noqa: E402
from sudoku.integration import Reading, read_mosaic, solve_reading  # noqa: E402
from sudoku.mosaic import load_pools, make_instance, render_mosaic  # noqa: E402
from sudoku.nn import DigitReader  # noqa: E402

EXPERIMENTS = RESULTS / "experiments"
STRATEGIES = {"A*": "astar", "UCS": "ucs", "Greedy": "greedy"}


@st.cache_resource
def get_pools():
    return load_pools()


@st.cache_resource
def get_reader():
    return DigitReader()


@st.cache_data
def build_puzzle(seed, givens, misreads):
    """Same steps as run_demo.py, so a seed gives the same puzzle in both."""
    rng = np.random.default_rng(seed)
    puzzle, solution = make_puzzles(seed, 1, givens)[0]
    instance = make_instance(puzzle, solution, misreads, get_pools(), rng)
    mosaic = render_mosaic(instance.images)
    return instance, mosaic, read_mosaic(mosaic, get_reader())


@st.cache_data
def run_search(grid, probs, strategy, goal, uniform=False):
    return solve_reading(Reading(list(grid), probs), strategy, goal, uniform=uniform, keep_trace=not uniform)


@st.cache_data
def csp_table(grid):
    rows = []
    for name, (use_fc, use_ac3) in CSP_CONFIGS.items():
        c = solve(list(grid), use_fc=use_fc, use_ac3=use_ac3, use_mrv=False, max_assignments=500_000).counters
        rows.append({"Setting": name, "Assignments": c.assignments, "Backtracks": c.backtracks,
                     "Values pruned": c.values_pruned})
    return pd.DataFrame(rows)


@st.cache_data
def load_table(path):
    return pd.read_csv(path)


@st.cache_data
def load_log():
    log = pd.read_csv(EXPERIMENTS / "correction_log.csv", dtype=str, keep_default_na=False)
    log["cost"] = log["cost"].astype(float)
    return log


def grid_figure(grid, title, highlight=(), originals=None):
    fig = Figure(figsize=(4, 4.2))
    draw_digits(fig.subplots(), grid, title, highlight, originals)
    fig.tight_layout()
    return fig


def mosaic_figure(mosaic):
    fig = Figure(figsize=(4, 4.2))
    ax = fig.subplots()
    ax.imshow(mosaic, cmap="gray_r")
    ax.set_title("Input image", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def trace_table(trace):
    rows = []
    for step in trace:
        verdict = "ACCEPTED" if step["goal"] else ("CSP rejected" if step["conflict_free"] else "Has duplicates")
        tried = ", ".join(f"{cell_name(c)}={d}" for c, d in step["state"]) or "(first reading)"
        rows.append({"Order": step["order"], "Cost so far": round(step["g"], 3), "Corrections tried": tried,
                     "Result": verdict})
    return pd.DataFrame(rows)


def show_outcome(reading, instance, strategy, goal):
    """Everything after the reading: AC-3, search, correction log, solution, CSP pruning, NN off."""
    truth = instance.solution
    if ac3_consistent(reading.grid):
        st.success("AC-3: no conflict found in this reading.")
    else:
        pairs = ", ".join(f"{cell_name(a)} and {cell_name(b)}" for a, b in conflict_pairs(reading.grid))
        st.warning(f"AC-3: conflict found. Duplicate pairs: {pairs or 'none visible, a domain emptied'}.")

    outcome = run_search(tuple(reading.grid), reading.probs, strategy, goal)
    exact = outcome.solution == truth
    cols = st.columns(5)
    cols[0].metric("Status", outcome.status.capitalize())
    cols[1].metric("Matches true solution", "Yes" if exact else "No")
    cols[2].metric("Corrections", len(outcome.log))
    cols[3].metric("Nodes expanded", f"{outcome.search.expanded:,}")
    cols[4].metric("CSP calls", f"{outcome.search.goal_tests:,}")

    if outcome.status != "solved":
        st.error(outcome.message)
    elif exact:
        st.success(f"{outcome.message}. The answer is exactly right.")
    else:
        st.error(f"{outcome.message}. This is a valid grid, but it is not the true solution.")

    left, right = st.columns(2)
    with left:
        st.subheader("Correction log")
        if outcome.log:
            rows = []
            for row, (cell, digit, _) in zip(outcome.log, outcome.search.corrections):
                rows.append({"Cell": row["cell"], "Read digit": row["read_digit"], "New digit": digit,
                             "True digit": instance.puzzle[cell], "Cost": row["cost"],
                             "Right fix": "Yes" if digit == instance.puzzle[cell] else "No"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.write("No corrections made.")
    with right:
        if outcome.solution:
            wrong = [i for i in range(81) if outcome.solution[i] != truth[i]]
            title = f"Answer ({len(wrong)} wrong cells shaded)" if wrong else "Answer (blue digits were filled in)"
            st.pyplot(grid_figure(outcome.solution, title, wrong, instance.puzzle), width="stretch")

    with st.expander(f"Search trace ({len(outcome.search.trace):,} states visited)"):
        st.dataframe(trace_table(outcome.search.trace), hide_index=True, width="stretch")

    with st.expander("CSP pruning: BT vs BT+FC vs BT+FC+AC-3"):
        st.caption("Same puzzle and variable order, so only the inference differs. Plain BT stops at 500,000 "
                   "assignments.")
        target = outcome.corrected_grid or instance.puzzle
        st.dataframe(csp_table(tuple(target)), hide_index=True, width="stretch")

    with st.expander("System with the NN on vs off"):
        st.caption("NN off ignores the probabilities: every change costs 1, and UCS tries them in cell order.")
        off = run_search(tuple(reading.grid), reading.probs, "ucs", goal, uniform=True)
        rows = [("NN on", outcome), ("NN off", off)]
        st.dataframe(pd.DataFrame([{"System": name, "Status": o.status, "Nodes expanded": o.search.expanded,
                                    "CSP calls": o.search.goal_tests, "Matches truth": o.solution == truth}
                                   for name, o in rows]), hide_index=True, width="stretch")


def solve_tab(instance, mosaic, reading, strategy, goal):
    conflicts = {c for pair in conflict_pairs(reading.grid) for c in pair}
    left, right = st.columns(2)
    left.pyplot(mosaic_figure(mosaic), width="stretch")
    right.pyplot(grid_figure(reading.grid, "NN reading (duplicates shaded)", conflicts), width="stretch")
    if instance.misread_cells:
        injected = ", ".join(f"{cell_name(c)} read as {reading.grid[c]}, true digit {instance.puzzle[c]}"
                             for c in instance.misread_cells)
        st.info(f"Injected misreads (hidden from the system): {injected}")
    else:
        st.info("No misreads were injected.")
    show_outcome(reading, instance, strategy, goal)


def edit_tab(instance, reading, strategy, goal, puzzle_key):
    st.write("This grid holds the network's reading. Change any digits, using 0 for blank, then press Solve to see "
             "whether the system recovers the true solution.")
    st.caption(f"Your digit is treated as what the network read, but the network's probabilities stay the same. "
               f"The system can make at most {MAX_CORRECTIONS} corrections.")
    table = pd.DataFrame([reading.grid[i:i + 9] for i in range(0, 81, 9)],
                         index=[f"R{i}" for i in range(1, 10)], columns=[f"C{i}" for i in range(1, 10)])
    config = {c: st.column_config.NumberColumn(c, min_value=0, max_value=9, step=1, width="small")
              for c in table.columns}
    with st.form(f"edit_{puzzle_key}"):
        edited = st.data_editor(table, column_config=config, width="content")
        submitted = st.form_submit_button("Solve my version", type="primary")
    if submitted:
        st.session_state["edited"] = (puzzle_key, edited.fillna(0).astype(int).to_numpy().ravel().tolist())
    saved = st.session_state.get("edited")
    if not saved or saved[0] != puzzle_key:
        return
    grid = saved[1]
    changed = [i for i in range(81) if grid[i] != reading.grid[i]]
    if changed:
        st.write("You changed: " + ", ".join(f"{cell_name(c)} from {reading.grid[c]} to {grid[c]}" for c in changed))
    else:
        st.write("You changed nothing, so this is the network's reading.")
    show_outcome(Reading(grid, reading.probs), instance, strategy, goal)


def metrics_section():
    metrics = json.loads((RESULTS / "nn_metrics.json").read_text())
    names = {"cnn_uncalibrated": "CNN, uncalibrated", "cnn_calibrated": "CNN, calibrated",
             "logistic_regression": "Logistic regression"}
    st.dataframe(pd.DataFrame([{"Model": label, "Accuracy": f"{metrics[key]['accuracy']:.2%}",
                                "ECE": round(metrics[key]["ece"], 4), "NLL": round(metrics[key]["nll"], 4)}
                               for key, label in names.items()]), hide_index=True, width="stretch")
    cols = st.columns(3)
    for col, name in zip(cols, ("confusion_matrix.png", "confusion_matrix_logreg.png", "reliability.png")):
        col.image(str(RESULTS / name), width="stretch")


def experiment_section():
    plots = {1: ("exp1_recovery.png", "exp1_nodes.png", "exp1_wrong_answers.png"),
             2: ("exp2_assignments.png", "exp2_backtracks.png"), 3: ("exp3_recovery.png",)}
    titles = {1: "1: UCS vs Greedy vs A*", 2: "2: BT vs BT+FC vs BT+FC+AC-3", 3: "3: NN-guided vs none vs random"}
    number = st.radio("Experiment", list(titles), format_func=titles.get, horizontal=True)
    table = load_table(EXPERIMENTS / f"exp{number}_summary.csv")
    st.dataframe(table.round(2), hide_index=True, width="stretch")
    for col, name in zip(st.columns(3), plots[number]):
        col.image(str(EXPERIMENTS / name), width="stretch")


def log_section():
    log = load_log()
    cols = st.columns(4)
    experiment = cols[0].selectbox("Experiment", ["All", "1", "3"])
    shown = log if experiment == "All" else log[log["experiment"] == experiment]
    method = cols[1].selectbox("Method", ["All"] + sorted(shown["method"].unique()))
    misreads = cols[2].selectbox("Misreads", ["All"] + sorted(log["misreads"].unique()))
    recovered = cols[3].selectbox("Recovered", ["All", "True", "False"])
    for column, value in (("method", method), ("misreads", misreads), ("recovered", recovered)):
        if value != "All":
            shown = shown[shown[column] == value]
    puzzles = shown.drop_duplicates(["experiment", "method", "goal", "seed", "misreads", "puzzle"])
    exact = int((puzzles["recovered"] == "True").sum())
    st.write(f"{len(puzzles):,} answered puzzles, {exact:,} exactly right, {len(shown):,} rows.")
    st.dataframe(shown, hide_index=True, width="stretch")


def tests_tab():
    st.write("Runs the full test suite: `python -m pytest -q`.")
    if st.button("Run tests", type="primary"):
        with st.spinner("Running tests..."):
            st.session_state["tests"] = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    result = st.session_state.get("tests")
    if result:
        lines = result.stdout.strip().splitlines()
        (st.success if result.returncode == 0 else st.error)(lines[-1] if lines else "No output")
        st.code(result.stdout + result.stderr)


def randomise_seed():
    st.session_state["seed"] = int(np.random.default_rng().integers(1, 1_000_000))


def main():
    st.set_page_config(page_title="P7 Sudoku tester", layout="wide")
    st.title("Handwritten Sudoku Reader and Solver")

    st.session_state.setdefault("seed", TEAM_SEED)
    with st.sidebar:
        st.header("Puzzle settings")
        seed = st.number_input("Seed", min_value=0, step=1, key="seed")
        st.button("Random seed", on_click=randomise_seed)
        givens = st.slider("Givens", 25, 35, 30)
        misreads = st.slider("Injected misreads", 0, 6, 2)
        strategy = STRATEGIES[st.selectbox("Search", list(STRATEGIES))]
        goal = st.selectbox("Goal", ["unique", "solvable"])
        st.caption(f"At most {MAX_CORRECTIONS} corrections, so 4 or more misreads should fail.")

    solve_page, edit_page, results_page, tests_page = st.tabs(
        ["Solve a puzzle", "Make your own misread", "Results", "Run tests"])
    instance, mosaic, reading = build_puzzle(int(seed), givens, misreads)
    with solve_page:
        solve_tab(instance, mosaic, reading, strategy, goal)
    with edit_page:
        edit_tab(instance, reading, strategy, goal, f"{seed}_{givens}_{misreads}")
    with results_page:
        st.subheader("Digit classifier")
        metrics_section()
        st.subheader("Experiments")
        experiment_section()
        st.subheader("Correction log")
        log_section()
    with tests_page:
        tests_tab()


main()
