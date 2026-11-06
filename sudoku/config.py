"""Shared constants. Change TEAM_SEED once per team: every generated puzzle derives from it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Each team must use its own generator seed (project brief). Set this to your team's number.
TEAM_SEED = 7007

SPLIT_SEED = 42            # train/validation split
TRAIN_SEED = 42            # network weights
N_TRAIN, N_VAL = 50_000, 10_000

MAX_CORRECTIONS = 3        # brief: at most 3 corrections
UNCERTAIN_CELLS = 8        # least-confident givens offered to the search (conflict cells are always added)
TOP_ALTERNATIVES = 3       # alternatives offered per uncertain cell
MAX_SEARCH_NODES = 50_000  # safety limit on generated states

BLANK_THRESHOLD = 0.02     # a cell with mean brightness below this is blank
HARD_CONFIDENCE = 0.90     # a hard digit is misread with confidence below this
MIN_HARD_PER_DIGIT = 15    # below this, a digit's hard pool is topped up with perturbed copies
