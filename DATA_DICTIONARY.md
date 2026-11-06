# Data dictionary

## Summary of real and generated data

| Item | Real or generated | Source or method |
|---|---|---|
| Digit images and labels | Real | MNIST handwritten digits |
| Sudoku puzzles and solutions | Generated | Randomised backtracking, then cell removal |
| 9x9 image mosaics | Generated | MNIST images pasted into a grid |
| Injected hard digits | Real images, chosen by rule | Test images the network misreads |
| Perturbed hard digits | Generated | Rotated and occluded test images, used only to top up small pools |
| Blank cells | Generated | Faint uniform noise from 0 to 0.01 |

This is not photographed real-world Sudoku.

## MNIST files (real)

Source: http://yann.lecun.com/exdb/mnist/ (404 at build time), retrieved from a mirror listed in `sudoku/data.py`.
A local copy of the four files is included in `data/raw/`, and MD5 checksums in `sudoku/data.py` match for all of them.

| File | Contents |
|---|---|
| `train-images-idx3-ubyte.gz` | 60,000 images, 28x28, grey 0-255 |
| `train-labels-idx1-ubyte.gz` | 60,000 labels, 0-9 |
| `t10k-images-idx3-ubyte.gz` | 10,000 images, written by different people from the training set |
| `t10k-labels-idx1-ubyte.gz` | 10,000 labels, 0-9 |

## `data/processed/mnist_1to9.npz`

Made by `make_data.py` with `SPLIT_SEED = 42`: shuffle, split 50,000 train and 10,000 validation, then remove
digit 0 from all splits. The split comes first so the brief's 50,000/10,000 split is respected. Counts are therefore
below 50,000 and 10,000 after removal, and the exact numbers are in `results/data_summary.json`.

| Column | Type | Real or generated | Meaning |
|---|---|---|---|
| `x_train`, `x_val`, `x_test` | uint8 (n, 28, 28) | Real | Pixels. Scaled to 0-1 by `load_processed()` |
| `y_train`, `y_val`, `y_test` | uint8 (n) | Real | Digit 1-9. Network class index = digit - 1 |

The test set is used only for final evaluation and as the source of digit images for puzzles. Training, model
selection and the temperature all use train and validation data only.

## `models/pools.npz`

Made by `train.py` with `TEAM_SEED`. For each digit `d` from 1 to 9:

| Column | Real or generated | Meaning |
|---|---|---|
| `clean_x_d` | Real | Test images of `d` the network reads as `d` |
| `clean_p_d` | Generated (model output) | Their calibrated probabilities |
| `hard_x_d` | Real, plus perturbed copies if needed | Images of `d` the network reads as another digit with confidence below 0.90 |
| `hard_p_d` | Generated (model output) | Their calibrated probabilities |

**Injection rule.** In a puzzle with k misreads, k givens are chosen uniformly at random and take a random `hard_x`
image of their true digit. All other givens take a random `clean_x` image, so the misread count is exactly k.

**Top-up rule.** If a digit has fewer than 15 natural hard images, copies of clean images are rotated by 20 to 35
degrees either way and given a 7x7 blanked patch. Copies the network misreads with confidence below 0.90 are added
until the pool reaches 15. The natural and perturbed counts per digit are in `results/nn_metrics.json`.

## Generated puzzle fields (`Instance` in `sudoku/mosaic.py`)

| Field | Type | Real or generated | Meaning |
|---|---|---|---|
| `puzzle` | 81 ints | Generated | True givens, 0 for blank |
| `solution` | 81 ints | Generated | The unique solution, used as ground truth |
| `images` | (81, 28, 28) float | Real digits, generated blanks | The image for each cell |
| `probs` | (81, 9) float | Generated (model output) | Calibrated digit probabilities stored in the pools. Experiments and the demo recompute them from the image |
| `misread_cells` | list of ints | Generated | Cells that received a hard image |

**Puzzle method.** A complete grid is built by randomised backtracking. Cells are removed in random order, and a
removal is kept only if the puzzle still has exactly one solution, until the target number of givens is reached.
If the target cannot be reached, a new grid is started.

**Mosaic method.** Cells are pasted into a 9x9 grid of 28x28 cells separated by 2-pixel lines, and cut back out using
the known layout. A cell is blank when its mean brightness is below 0.02. The network never sees blank cells.

## Correction log

`results/experiments/correction_log.csv` has rows for every answered puzzle in every method of Experiments 1 and 3.
Experiment 2 is not included because it makes no corrections. `results/demo/correction_log.csv` has `cell`,
`read_digit`, `new_digit` and `cost` for the latest demo run. All fields are generated.

| Field | Meaning |
|---|---|
| `experiment` | 1 or 3 |
| `method` | `ucs`, `greedy` or `astar` in Experiment 1; `NN-guided`, `No correction` or `Random` in Experiment 3 |
| `goal` | `solvable`, `unique`, or `none` for no correction |
| `seed`, `misreads`, `puzzle` | Which puzzle this row belongs to |
| `recovered` | True if the answer matches the true solution exactly |
| `cell` | Cell name, for example R3C7, or `none` if the puzzle needed no correction |
| `read_digit` | The digit the network read |
| `new_digit` | The digit chosen by the method |
| `cost` | log P(read digit) - log P(new digit), never negative. For Random it is computed afterwards and not used |
| `true_digit` | The correct digit for that cell |
