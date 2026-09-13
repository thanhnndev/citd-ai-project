# CITD ML — BTCUSD Pyramid Strategy + CatBoost Signal Filter

A machine-learning study on **when a trading strategy should be trusted**. The
CatBoost model is a *signal filter*, not a trading decision maker: the pyramid
strategy decides which orders to open, and the model scores each order — low
scores are skipped.

The core research question is **label leakage**. Triple-barrier labels are
determined by looking into the future, so orders that open close together share
overlapping labels. The strategy deliberately opens a base order plus three
pyramid legs (four orders per `origin_bar` family), so siblings are almost
guaranteed to share the same outcome. A naive random split puts siblings on
both sides of train/test and inflates the score. This project compares four
splits of increasing strictness and then evaluates the model on a **sealed
holdout period** that nobody touched during development.

> Vietnamese guide: [`README_VI.md`](README_VI.md). The original handover task
> specifications are in [`docs/`](docs/) (Vietnamese).

---

## Results

Frozen holdout boundary: **`2025-02-08 15:30:00`** (M15 bar 199,968).

### Table 1 — Classification, chunk 1 removed on all branches

| Method | ROC-AUC | F1 @ 0.5 |
|---|---:|---:|
| Cách 1 — Random K-Fold | 0.8595 | 0.6525 |
| Cách 1b — Grouped K-Fold | 0.7475 | 0.5105 |
| Cách 2 — Walk-forward | 0.5867 | 0.3267 |
| Cách 3 — WF + Purge/Embargo | 0.5948 | 0.3304 |
| **Holdout** | **0.6046** | **0.4022** |

The gap between Random K-Fold (0.86) and the leak-aware splits (~0.59) is the
leakage effect this project exists to demonstrate.

### Table 2 — Financial metrics, top 50% kept

| | Trades | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---:|---:|---:|---:|---:|
| Baseline (chunk 2–5) | 20,007 | +3,358.3 | 236.1 | 1.294 | 31.7 |
| Cách 1 — Random K-Fold | 10,004 | +6,998.3 | 75.7 | 2.563 | 44.9 |
| Cách 1b — Grouped K-Fold | 10,004 | +4,724.3 | 99.5 | 1.942 | 39.2 |
| Cách 2 — Walk-forward | 10,004 | +2,207.3 | 205.3 | 1.423 | 34.9 |
| Cách 3 — WF + Purge/Embargo | 10,004 | +2,253.7 | 154.1 | 1.435 | 35.2 |
| **Baseline holdout** | 5,028 | +245.93 | 305.32 | 1.0992 | 35.28 |
| **Holdout, top 50%** | 2,514 | −36.55 | 263.25 | 0.9718 | 34.81 |

On the sealed holdout the model still does **not** beat the unfiltered baseline:
keeping the top 50% leaves 2,514 trades with **−36.55 R** and profit factor
0.972, versus **+245.93 R** and profit factor 1.099 for the full 5,028-trade
baseline. The full 20–80% sweep is in
[`outputs/holdout/stage4/`](outputs/holdout/stage4/), and the complete report is
[`docs/BAO_CAO_KET_QUA_HOLDOUT.md`](docs/BAO_CAO_KET_QUA_HOLDOUT.md).

---

## Repository layout

```
citd-ml-project/
├── README.md                  ← this file (English)
├── README_VI.md               ← Vietnamese guide
├── LICENSE                    MIT
├── pyproject.toml             package metadata + pinned dependencies (uv)
├── requirements.txt           same pins for plain pip
├── .gitignore .gitattributes
│
├── data/
│   ├── raw/                   raw BTCUSD M1 CSV (NOT committed, see below)
│   └── processed/             committed pipeline inputs
│       ├── dataset_catboost.csv          25,008 rows × 31 cols
│       ├── tradelist_pyramid_local.csv   30,040 baseline orders
│       └── triple_barrier_labels.csv     triple-barrier labels
│
├── src/citd_ml/               installable Python package
│   ├── paths.py               single source of truth for all paths/constants
│   ├── strategy/              pyramid entry/exit state machine
│   ├── labeling/              triple-barrier labelling
│   ├── features/              dataset build + validation
│   ├── training/              data prep, 4 splits, CatBoost training
│   ├── backtest/              baseline + score-gated backtest
│   └── verification/          byte-level reproducibility check
│
├── scripts/                   command-line entry points
│   ├── build_dataset.py       build processed dataset
│   ├── verify_dataset.py      validate the dataset
│   ├── train_models.py        train the 4 branches, write OOF tables
│   ├── run_backtest.py        baseline + 20–80% sweep
│   ├── verify_pipeline.py     run twice, assert byte-identical output
│   ├── holdout_stage1_split.py
│   ├── holdout_stage2_train.py
│   ├── holdout_stage3_backtest.py
│   └── holdout_stage4_report.py
│
├── outputs/                   committed results/evidence
│   ├── catboost_training/     4 OOF tables + fold metrics
│   ├── backtest/              scored universe, sweeps, top-50 trades
│   ├── verification/          reproducibility manifest
│   └── holdout/
│       ├── stage1/            regenerated + holdout datasets, validation
│       ├── stage2/            final models (.cbm), predictions, reports
│       ├── stage3/            scored holdout, top-20…80% trades, summary
│       └── stage4/            tables, decisions, HTML equity curves
│
├── docs/                      handover tasks + final report (Vietnamese)
└── tests/                     lightweight layout/constant tests
```

---

## Environment setup

**Python 3.12 is required** (the verified run used Python 3.12.14). Choose
either `uv` (recommended) or a plain `.venv`.

### Option A — `uv` (recommended)

`uv` creates and manages `.venv/` for you, and installs the exact pinned
versions from `pyproject.toml`.

```bash
# 1. install uv once (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. create .venv + install dependencies and the package in editable mode
uv sync --extra dev

# 3. run anything inside the project environment
uv run python scripts/train_models.py
uv run pytest
```

`uv sync --extra dev` reads `pyproject.toml`, creates `.venv/`, installs
`pytest` from the `dev` extra, and uses the committed `uv.lock` so every clone
gets identical versions.

To add a new dependency:

```bash
uv add <package>          # updates pyproject.toml + uv.lock
```

### Option B — plain `venv` + `pip`

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                 # installs the citd_ml package
python scripts/train_models.py
```

> The `.venv/` folder is ignored by git. Never commit it. On Windows the
> activation command is `.venv\Scripts\activate`.

### Verify the environment

```bash
uv run python -c "import catboost, sklearn, pandas, numpy, plotly; print('ok')"
uv run pytest
```

---

## Data setup

The raw market file `BTCUSD_m1_2018_to_now.csv` is **~209 MB**, which exceeds
GitHub's 100 MB per-file limit, so it is **not committed**. See
[`data/raw/README.md`](data/raw/README.md) for the exact format and how to
restore it.

Place it at:

```
data/raw/BTCUSD_m1_2018_to_now.csv
```

Everything under `data/processed/` and `outputs/` **is** committed, so you can
inspect every result and re-run training/backtesting without the raw file. The
raw file is only needed to rebuild the dataset from scratch.

---

## Running the pipeline

All commands assume you are at the repository root and use `uv run` (swap in
`python` if you activated a `.venv` manually).

> **Frozen step-4 artifacts.** `outputs/catboost_training/`, `outputs/backtest/`
> and `outputs/verification/` are the handed-over step-4 results, kept
> byte-identical to the handover. Steps 3–5 below re-run the step-4 pipeline and
> **overwrite** them; CatBoost training is hardware-sensitive, so a re-run on a
> different host will not reproduce the frozen numbers. The sealed-holdout
> workflow in the next section is the step-5 deliverable and never touches those
> folders.

### 1. Build the processed dataset

```bash
uv run python scripts/build_dataset.py
```

Replays the strategy over M1 history before the holdout, labels every order
with the triple-barrier rule, computes 23 features, and writes
`data/processed/dataset_catboost.csv` (25,008 rows).

### 2. Validate the dataset

```bash
uv run python scripts/verify_dataset.py
```

Checks row/column counts, feature ranges, label distribution, chunk edges and
that the holdout is still sealed. A non-zero exit code means **stop**.

### 3. Train the four branches

```bash
uv run python scripts/train_models.py
```

Trains Random K-Fold, Grouped K-Fold, Walk-forward and Purged Walk-forward and
writes OOF probability tables + fold metrics to `outputs/catboost_training/`.

### 4. Backtest and sweep

```bash
uv run python scripts/run_backtest.py
```

Runs the fixed-universe baseline, gates it with each OOF score table at
20–80%, and writes reports to `outputs/backtest/`.

### 5. Reproducibility check

```bash
uv run python scripts/verify_pipeline.py
```

Runs training and backtest twice and asserts the results are identical to the
committed CSVs, byte for byte. Writes
`outputs/verification/reproducibility.json`.

---

## Sealed holdout workflow

The holdout is opened once, in four stages. Run them in order.

```bash
# Stage 1 — regenerate full history and split out the holdout
uv run python scripts/build_dataset.py --holdout 2100-01-01 \
    --output outputs/holdout/stage1/dataset_catboost_full_regenerated.csv
uv run python scripts/holdout_stage1_split.py

# Stage 2 — train the final model twice and verify it is reproducible
uv run python scripts/holdout_stage2_train.py --run-id 1
uv run python scripts/holdout_stage2_train.py --run-id 2
uv run python scripts/holdout_stage2_train.py --verify

# Stage 3 — score the sealed holdout and backtest fixed-universe retention
uv run python scripts/holdout_stage3_backtest.py

# Stage 4 — build the summary tables and equity-curve HTML charts
uv run python scripts/holdout_stage4_report.py
```

Stage 1 also proves that the regenerated pre-holdout dataset contains all
25,008 frozen keys and matches all 575,184 feature cells exactly. Stage 2
verifies purge/embargo remove 0 rows and that two independent training runs
produce identical predictions.

---

## Frozen constants

These values are fixed by the handover spec and must not be changed
(`src/citd_ml/paths.py`).

| Constant | Value |
|---|---|
| Holdout boundary | `2025-02-08 15:30:00` (M15 bar 199,968) |
| Dataset | 25,008 rows, 6,252 families × 4 legs, label-1 rate 26.2% |
| Features / metadata | 23 `FEATURES`, 7 `META` |
| Chunk edges | `[0, 5001, 10003, 15004, 20006, 25008]` |
| CatBoost params | iterations 1000, lr 0.05, depth 6, l2_leaf_reg 3.0, `auto_class_weights=Balanced`, `eval_metric=AUC`, seed 42, `thread_count=1` |
| Verified environment | Python 3.12.14 · catboost 1.2.10 · scikit-learn 1.9.0 · pandas 3.0.5 · numpy 2.5.2 |

> `thread_count=1` is an **added technical setting**, not one of the original
> model hyperparameters: it removes CPU thread count as one known source of
> numerical variation. The automated check proves exact repeatability for two
> runs on the same machine; it does not prove byte-identical models on every
> machine. The handed-over branch
> artifacts in `outputs/catboost_training/` and `outputs/backtest/` were produced
> without it and are preserved as-is (see Changelog).

---

## Notes and conventions

- **Never pass `META` columns to the model.** `leg` and `entry_vs_base_R`
  describe the pyramid scaffolding, not the market, and would leak family
  structure.
- **The candidate-trade universe is fixed.** Filtering a trade out must not
  change later signals; the backtest keeps a shadow strategy for exactly this
  reason.
- **Do not edit committed datasets or `outputs/`** when experimenting. Write new
  results to a new folder so the frozen artifacts stay valid.
- **Reproducibility:** `random_seed=42` and `thread_count=1` are both pinned.
  On the artifact-producing machine, two independent training runs yield
  identical prediction arrays; the two `.cbm` files have different SHA-256
  hashes because serialized metadata differs. A Windows teammate reported
  predictions within `1e-15`, but no cross-machine artifact or controlled
  thread-count experiment is committed, so this report does not attribute all
  host variation to multithreading. The handed-over step-4
  artifacts (`outputs/catboost_training/`, `outputs/backtest/`) are kept
  byte-identical and were produced on the original machine without
  `thread_count`, so re-running steps 3–4 will not reproduce those files on a
  different host.

## Documentation

| File | Contents |
|---|---|
| [`docs/BAO_CAO_KET_QUA_HOLDOUT.md`](docs/BAO_CAO_KET_QUA_HOLDOUT.md) | Final holdout report (numbers, charts, verification) |
| [`docs/BAN_GIAO_task_holdout.md`](docs/BAN_GIAO_task_holdout.md) | Original holdout task spec |
| [`docs/BAN_GIAO_task_train_catboost.md`](docs/BAN_GIAO_task_train_catboost.md) | Original training task spec |
| [`docs/quy_trinh_lam_viec.md`](docs/quy_trinh_lam_viec.md) | Project working process |
| [`docs/handover_README.md`](docs/handover_README.md) | Original handover README |

## Changelog

### 2026-09-13 — Finalize holdout handoff and evidence scope

- **Incorporated final report review:** documented the holdout re-run history
  and technical reason, clarified that Table 1 reports fold means after chunk 1
  is removed, explicitly marked full-pipeline byte reproducibility as unproven,
  and added a purpose description for every generated artifact.
- **Made report publication fail closed:** Stage 4 now validates the underlying
  Stage 1–3 evidence flags and expected counts before writing any report output.
  Negative tests confirm that failed feature matching, prediction equality, or
  baseline replay prevents a PASS report from being generated.
- **Changed equity curves to step lines:** all seven Plotly traces now remain
  flat between close times and jump only when closed-trade R is recorded; trace
  counts, data scope, CSV endpoints, and reported results remain unchanged.
- **Finalized the submission report:**
  `docs/BAO_CAO_KET_QUA_HOLDOUT.md` is now generated from the committed Stage
  1–3 JSON/CSV artifacts. It includes the final training configuration,
  classification and financial tables, the 20–80% holdout sweep, chart links,
  verification status, and an automatically refreshed output inventory.
- **Scoped reproducibility claims to the available evidence:** two independent
  runs on the artifact-producing machine have 25,008 byte-identical
  predictions (`max abs difference = 0.0`), while the serialized `.cbm` hashes
  differ. The Windows teammate result (predictions within `1e-15`) is recorded
  as supporting evidence, not proof that thread count is the sole cause of all
  cross-machine variation.
- **Made both Plotly reports deterministic:** fixed HTML `div_id` values prevent
  random UUID-only diffs when Stage 4 is run again.
- **Added consistency tests** for the reproducibility evidence, rounded holdout
  summary, and deterministic chart IDs. The final suite passes 5/5 tests.
- **Corrected fresh-clone setup** to use `uv sync --extra dev`, ensuring
  `pytest` is installed before the documented test command is run.
- **Revalidated the complete holdout workflow:** 25,008/25,008 frozen keys and
  575,184/575,184 feature cells match; purge and embargo remove 0 rows; both
  training runs have identical predictions; baseline replay matches 5,028/5,028
  holdout trades; Stage 3 metrics and Stage 4 reports regenerate unchanged.

### 2026-09-12 — Restore handed-over branch artifacts, fix chart 1, document deviation

- **Restored** `data/processed/dataset_catboost.csv`,
  `outputs/catboost_training/`, `outputs/backtest/` and `outputs/verification/`
  to the handover originals (they had been regenerated on Linux), so Table 1/2
  and chart 1 use the frozen branch reference values.
- **Fixed** chart 1 (`equity-curve-chunk2-5-top50.html`): it now builds from the
  restored handed-over `backtest_scored_universe.csv`, so its 4 branch endpoints
  match Table 2 exactly (+6,998.3 / +4,724.3 / +2,207.3 / +2,253.7 R).
- **Corrected** the Stage-4 report's file table (real script names and sizes) and
  synced `outputs/holdout/stage4/implementation_decisions.md` with the report.
- **Documented** the `thread_count=1` deviation from the frozen hyperparameter
  list as a technical (non-model) setting.

### 2026-09-12 — Pin `thread_count=1` (reduce one source of numerical drift)

- **Initial observation:** CatBoost was configured with `random_seed=42` but no
  `thread_count`, so execution thread count depended on the host. This is a
  plausible source of floating-point ordering variation. Earlier runs did
  differ, but the repository contains no controlled experiment changing only
  `thread_count`, so it is not claimed as the sole root cause.
- **Fix:** pinned `thread_count=1` in
  `src/citd_ml/training/train_catboost.py` and
  `scripts/holdout_stage2_train.py`.
- **Re-ran** the full pipeline and holdout stages. Holdout ROC-AUC/F1 =
  **0.6046 / 0.4022**; holdout top-50% = **−36.55 R** (profit factor 0.972)
  versus baseline **+245.93 R** (profit factor 1.099).

### 2026-09-12 — uv environment + full pipeline re-run

- **Environment:** installed and pinned the project with `uv sync`
  (Python 3.12.14, catboost 1.2.10, scikit-learn 1.9.0, pandas 3.0.5,
  numpy 2.5.2, plotly 7.0.0) and committed `uv.lock`.
- **Full pipeline re-run:** `build_dataset → verify_dataset → train_models →
  run_backtest → verify_pipeline` all pass, plus holdout stages 1–4. The
  regenerated step-4 artifacts were later restored to the handover originals
  (see the entry above); only `outputs/holdout/` keeps the new run.
- **Holdout re-run:** the holdout stages were re-run; final numbers are in the
  entry above (the classification/financial values were refreshed again when
  `thread_count` was pinned).
- **Fixes:**
  - `scripts/holdout_stage4_report.py` now fills the Table 1 *Holdout* row from
    `outputs/holdout/stage3/stage3_report.json` instead of a hard-coded value
    that could silently go stale.
  - `scripts/holdout_stage1_split.py` replaces the pandas-deprecated
    `set_index(..., verify_integrity=True)` with an explicit duplicate-key check.
  - Fixed dead chart links in `docs/BAO_CAO_KET_QUA_HOLDOUT.md`
    (`../work/stage4_results/...` → `../outputs/holdout/stage4/...`).
- **Removed:** `docs/legacy/` (the archived pre-refactor stage-3 variant).
  The canonical implementation has lived in `scripts/holdout_stage3_backtest.py`
  since the refactor and produced all committed stage-3 artefacts, so the
  legacy copy was redundant.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
