"""
config.py - shared settings for Navya's parts (P2 features, P4 boosting).

Big data files live one folder ABOVE this repo (in "ML Project/"),
so they are never committed to git.
"""
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR.parent                      # .../ML Project/

CLEANED_CSV = DATA_DIR / "cleaned_flights.csv"  # made by P1's cleanup.py
FEATURES_CSV = DATA_DIR / "features_v1.csv"     # frozen feature table (P2)

OUT_DIR = REPO_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
RESULT_DIR = OUT_DIR / "results"
MODEL_DIR = OUT_DIR / "models"          # git-ignored
PRED_DIR = OUT_DIR / "predictions"      # git-ignored
for d in (FIG_DIR, TABLE_DIR, RESULT_DIR, MODEL_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Task definition -------------------------------------------------
LATE_THRESHOLD = 15      # LATE = 1 if DEP_DELAY >= 15 min (as in the brief)
CHECKPOINT_MIN = 120     # checkpoint T = 2 hours before scheduled departure
WINDOW_MIN = 120         # "recent activity" window: the 2 hours before T

# ---- Time split (dates are inclusive) --------------------------------
DEV_START = "2024-01-01"
DEV_END = "2024-01-23"   # development data = Jan 1 - Jan 23
GAP_DAY = "2024-01-24"   # 1-day gap, not used anywhere
TEST_START = "2024-01-25"  # final test = Jan 25 - Jan 31 (used once)

# Forward-chaining validation folds inside DEV (1-day gap before each val block)
#   (train_end, val_start, val_end)
FOLDS = [
    ("2024-01-11", "2024-01-13", "2024-01-15"),
    ("2024-01-15", "2024-01-17", "2024-01-19"),
    ("2024-01-19", "2024-01-21", "2024-01-23"),
]
SEEDS = [0, 1, 2]

# US federal holidays in / near the data window
HOLIDAYS = ["2023-12-25", "2024-01-01", "2024-01-15", "2024-02-19"]

# Plot colours (validated categorical palette, light mode)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#257de0", "#f47230", "#00b894", "#f2a600"
C_TEXT, C_MUTED, C_GRID = "#191919", "#666666", "#e6e6e6"
MODEL_COLORS = {"XGBoost": C_BLUE, "LightGBM": C_ORANGE, "CatBoost": C_AQUA}


def style_plots():
    """Consistent, quiet matplotlib style for all report figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C_MUTED, "axes.labelcolor": C_TEXT,
        "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9,
        "legend.frameon": False, "lines.linewidth": 2,
    })
    return plt
