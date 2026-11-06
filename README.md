# P7: Handwritten Sudoku Reader and Solver

A neural network reads a Sudoku made of MNIST digits. When a misread digit breaks the puzzle, a search
finds the cheapest corrections and a CSP solver confirms them.

```
image -> NN reads digits -> AC-3 finds a conflict -> search tries cheapest fixes -> CSP tests each -> solution
```

## Setup (Windows, Python 3.12)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate`. The CPU build of torch is enough. If the first
`pip install` is skipped, `requirements.txt` needs the same PyTorch index, because `+cpu` is not on PyPI.

On Windows, keep the project in a short path such as `D:\P7_Sudoku`. The torch install contains very long file
names and fails in deeply nested folders unless Windows long paths are enabled.

## Run

Run these in order from the project root.

| Step | Command | Time | Output |
|---|---|---|---|
| Prepare data | `python make_data.py` | under 1 min | `data/processed/` |
| Train and evaluate networks | `python train.py` | about 5 min on CPU | `models/`, `results/nn_metrics.json`, plots |
| Integrated demo | `python run_demo.py` | seconds | console, `results/demo/` |
| Unit tests | `python -m pytest -q` | seconds | pass/fail |
| All three experiments | `python run_experiments.py` | about 5 min on 16 cores | `results/experiments/` |

A trained model and the digit pools are included in `models/`, so `python train.py` can be skipped.
`python make_data.py` is only needed for training or for rebuilding pools.

`python run_experiments.py --quick` runs a small version for checking that everything works. It overwrites
`results/experiments/`, so run the full version again afterwards.

`report/Report.docx` and `report/Presentation.pptx` are edited directly in Word and PowerPoint.
`report/Report.pdf` is exported from the Word file.

## Demo options

```powershell
python run_demo.py --misreads 3 --strategy ucs --seed 12
python run_demo.py --goal solvable
python run_demo.py --impossible
```

The demo prints the reading, the AC-3 result, the search trace, the correction log, CSP pruning with and
without FC and AC-3, and the system with the NN on and off.

## Test UI (optional)

A local browser page for trying puzzles, making your own misreads, viewing results and running the tests.

```powershell
pip install -r requirements-ui.txt
python -m streamlit run ui.py
```

It runs the real pipeline and never writes to `results/`, `models/` or `report/`.

## Git log

Export the log for submission from the project root:

```powershell
git log --date=iso --stat --output=git_log.txt
```

## Project layout

| Path | Purpose |
|---|---|
| `sudoku/config.py` | Seeds and limits. Set `TEAM_SEED` here |
| `sudoku/data.py`, `make_data.py` | Data download and preprocessing |
| `sudoku/nn.py`, `sudoku/calibration.py`, `train.py` | Classifier, temperature scaling, calibration error |
| `sudoku/search.py` | UCS, Greedy, A* with node counters |
| `sudoku/csp.py` | Backtracking, Forward Checking, AC-3 switches with counters |
| `sudoku/generator.py`, `sudoku/mosaic.py` | Puzzle and image generators, misread injection |
| `sudoku/integration.py` | Connects the three parts |
| `sudoku/eval.py`, `run_experiments.py` | Tables, plots, experiments |
| `tests/` | Normal, edge and impossible tests |
| `DATA_DICTIONARY.md` | Every data field, real or generated |

## Reproducibility

All randomness derives from `TEAM_SEED`, `SPLIT_SEED` and `TRAIN_SEED` in `sudoku/config.py`. Changing `TEAM_SEED`
gives new puzzles; rerun `python train.py` and `python run_experiments.py` afterwards.

A local copy of MNIST is included in `data/raw/`, so `make_data.py` needs no internet. If a file is missing, it
downloads it again. The original MNIST page returned 404 when this was built, so it falls back to a mirror. Every
file's MD5 checksum is verified.
