"""Central path and constant configuration.

Every module resolves its inputs and outputs through this file, so the
repository can be relocated or cloned anywhere without editing individual
scripts. Paths are derived from the location of this file:

    <repo>/src/citd_ml/paths.py  ->  PROJECT_ROOT = <repo>
"""

from __future__ import annotations

from pathlib import Path

# <repo>/src/citd_ml/paths.py -> parents[0]=src/citd_ml, parents[1]=src, parents[2]=<repo>
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

SRC_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------- data files
RAW_M1_CSV = RAW_DATA_DIR / "BTCUSD_m1_2018_to_now.csv"
DATASET_CSV = PROCESSED_DATA_DIR / "dataset_catboost.csv"
TRADELIST_CSV = PROCESSED_DATA_DIR / "tradelist_pyramid_local.csv"
LABELS_CSV = PROCESSED_DATA_DIR / "triple_barrier_labels.csv"

# ------------------------------------------------------------------- outputs
CATBOOST_TRAINING_DIR = OUTPUTS_DIR / "catboost_training"
BACKTEST_DIR = OUTPUTS_DIR / "backtest"
VERIFICATION_DIR = OUTPUTS_DIR / "verification"

HOLDOUT_DIR = OUTPUTS_DIR / "holdout"
HOLDOUT_STAGE1_DIR = HOLDOUT_DIR / "stage1"
HOLDOUT_STAGE2_DIR = HOLDOUT_DIR / "stage2"
HOLDOUT_STAGE3_DIR = HOLDOUT_DIR / "stage3"
HOLDOUT_STAGE4_DIR = HOLDOUT_DIR / "stage4"

# ------------------------------------------------------- frozen research config
# These values are fixed by docs/BAN_GIAO_task_holdout.md. Do not change them.
HOLDOUT_START = "2025-02-08 15:30:00"  # bar M15 number 199,968
EMBARGO_START_BAR = 199_918
VERTICAL_BARS = 50
