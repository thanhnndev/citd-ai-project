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
holdout period**. Three historical result versions can be retrieved from Git;
technical reruns and later thread-count experiments are disclosed separately.
This does not establish the total number of executions or prove that no
development decision was influenced by viewing the holdout. See
[`outputs/verification/holdout_run_history.md`](outputs/verification/holdout_run_history.md).

> Vietnamese guide: [`README_VI.md`](README_VI.md). The original handover task
> specifications are in [`docs/`](docs/) (Vietnamese).

---

## Results

Frozen holdout boundary: **`2025-02-08 15:30:00`** (M15 bar 199,968).

The main blocks below are the **canonical `thread_count=1` rerun** committed in
[`outputs/step4_thread1/`](outputs/step4_thread1/); all eight compared CSVs are
byte-identical to the `thread_count=1` artifacts of commit `7748828`. The
handed-over branch values (produced before `thread_count` was pinned) are kept
as a clearly labelled reference; the holdout rows are unchanged.

### Table 1 — Classification, chunk 1 removed on all branches

| Method | ROC-AUC | F1 @ 0.5 |
|---|---:|---:|
| Cách 1 — Random K-Fold | 0.8582 | 0.6494 |
| Cách 1b — Grouped K-Fold | 0.7454 | 0.5077 |
| Cách 2 — Walk-forward | 0.5875 | 0.3288 |
| Cách 3 — WF + Purge/Embargo | 0.5895 | 0.3247 |
| **Holdout** | **0.6046** | **0.4022** |

Reference block — handed-over artifacts (no `thread_count` pinned), with the
canonical-minus-reference delta:

| Method | Reference ROC-AUC | Reference F1 | ΔROC-AUC | ΔF1 |
|---|---:|---:|---:|---:|
| Cách 1 — Random K-Fold | 0.8595 | 0.6525 | −0.0013 | −0.0031 |
| Cách 1b — Grouped K-Fold | 0.7475 | 0.5105 | −0.0021 | −0.0028 |
| Cách 2 — Walk-forward | 0.5867 | 0.3267 | +0.0007 | +0.0020 |
| Cách 3 — WF + Purge/Embargo | 0.5948 | 0.3304 | −0.0052 | −0.0057 |

### Table 2 — Financial metrics, top 50% kept

Canonical `thread_count=1` main block:

| | Trades | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---:|---:|---:|---:|---:|
| Baseline (chunk 2–5) | 20,007 | +3,358.25 | 236.13 | 1.2940 | 31.71 |
| Cách 1 — Random K-Fold | 10,004 | +6,937.53 | 73.76 | 2.5437 | 44.65 |
| Cách 1b — Grouped K-Fold | 10,004 | +4,752.04 | 99.71 | 1.9500 | 39.26 |
| Cách 2 — Walk-forward | 10,004 | +2,148.31 | 211.29 | 1.4117 | 34.76 |
| Cách 3 — WF + Purge/Embargo | 10,004 | +2,119.48 | 142.36 | 1.4074 | 34.99 |

Reference block — handed-over artifacts:

| | Trades | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---:|---:|---:|---:|---:|
| Baseline (chunk 2–5) | 20,007 | +3,358.25 | 236.13 | 1.2940 | 31.71 |
| Cách 1 — Random K-Fold | 10,004 | +6,998.30 | 75.71 | 2.5627 | 44.88 |
| Cách 1b — Grouped K-Fold | 10,004 | +4,724.34 | 99.50 | 1.9418 | 39.24 |
| Cách 2 — Walk-forward | 10,004 | +2,207.25 | 205.31 | 1.4226 | 34.86 |
| Cách 3 — WF + Purge/Embargo | 10,004 | +2,253.72 | 154.11 | 1.4350 | 35.17 |

Canonical top-50 net R differs from the handed-over reference by
−60.77 / +27.70 / −58.94 / −134.24 R for the four branches in order (per-file
comparison in
[`outputs/step4_thread1/comparison_report.json`](outputs/step4_thread1/comparison_report.json)).

Holdout block (unchanged):

| | Trades | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---:|---:|---:|---:|---:|
| **Baseline holdout** | 5,028 | +245.93 | 305.32 | 1.0992 | 35.28 |
| **Holdout, top 50%** | 2,514 | −36.55 | 263.25 | 0.9718 | 34.81 |

The full 20–80% sweep is in
[`outputs/holdout/stage4/`](outputs/holdout/stage4/), and the complete report is
[`docs/BAO_CAO_KET_QUA_HOLDOUT.md`](docs/BAO_CAO_KET_QUA_HOLDOUT.md).

### Thread-count sensitivity and holdout run history

A controlled experiment varying only `thread_count`
([`outputs/thread_count_sensitivity/`](outputs/thread_count_sensitivity/)) shows
that on this machine two `thread_count=1` runs are bit-identical
(`max|Δp| = 0`), while `thread_count=2` / `-1` change **every** prediction
(train `max|Δp|` 0.2832 / 0.2999; holdout 0.3469 / 0.3352; holdout Pearson
0.9331 / 0.9409). Holdout top-50 net R: **−36.55 R** (`tc=1`) vs **−12.26 R**
(`tc=2`) vs **+30.79 R** (default). The effect is in training, not inference:
re-scoring the same fitted model at prediction `thread_count` 1 / 2 / −1 gives
identical probabilities. CatBoost documents `thread_count` as a speed setting
that does not affect results; the measurement on this machine confirms that at
prediction time but contradicts it at training time. Scope: one machine, one
CatBoost build, one dataset, one seed.

Three historical holdout result versions are retrieved from Git in
[`outputs/verification/holdout_run_history.json`](outputs/verification/holdout_run_history.json)
([Markdown](outputs/verification/holdout_run_history.md)): `dc25cd3` Windows
0.60502 / 0.40171, top-50 −23.83 R; `5f46e41` Linux multi-thread
0.60232 / 0.40647, +30.79 R; `7748828` Linux `thread_count=1`
0.60455 / 0.40221, −36.55 R. The full run1-vs-run2 chain check
([`outputs/holdout/repro/stage5_repro_report.json`](outputs/holdout/repro/stage5_repro_report.json))
is **PASS** with all deltas 0.0 and all four charts byte-identical.

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
│   ├── thread_count_sensitivity.py   controlled thread_count experiment
│   ├── holdout_run_history.py        recover all 3 holdout openings from git
│   ├── holdout_stage1_split.py
│   ├── holdout_stage2_train.py
│   ├── holdout_stage3_backtest.py
│   ├── holdout_stage4_report.py
│   └── holdout_stage5_repro_check.py run1-vs-run2 chain comparison
│
├── outputs/                   committed results/evidence
│   ├── catboost_training/     handed-over 4 OOF tables + fold metrics (frozen reference)
│   ├── backtest/              handed-over scored universe, sweeps, top-50 trades
│   ├── step4_thread1/         canonical thread_count=1 four-split rerun + comparison_report.json
│   ├── thread_count_sensitivity/  controlled experiment JSON + summary CSV
│   ├── verification/          handover manifest + holdout_run_history.json/.md
│   └── holdout/
│       ├── stage1/            regenerated + holdout datasets, validation
│       ├── stage2/            final models (.cbm), predictions, reports
│       ├── stage3/            scored holdout, top-20…80% trades, summary
│       ├── stage4/            tables, decisions, HTML equity curves
│       └── repro/             stage5 run1-vs-run2 report + independent run2 tree
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

> **Frozen step-4 artifacts.** `outputs/catboost_training/`,
> `outputs/backtest/` and `outputs/verification/reproducibility.json` are the
> handed-over step-4 results, kept byte-identical to the handover. They are
> **reference-only**: the canonical comparison tree is
> [`outputs/step4_thread1/`](outputs/step4_thread1/) (`thread_count=1`), and
> CatBoost training is hardware-sensitive, so the frozen handover files are
> intentionally not byte-reproducible on Linux. Steps 3–5 below therefore write
> to the canonical tree via `--output-dir` / `--oof-dir`; running them without
> those flags **overwrites** the frozen folders. The sealed-holdout workflow in
> the next section is the step-5 deliverable and never touches those folders.

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
uv run python scripts/train_models.py \
    --output-dir outputs/step4_thread1/catboost_training
```

Trains Random K-Fold, Grouped K-Fold, Walk-forward and Purged Walk-forward and
writes OOF probability tables + fold metrics to the directory given by
`--output-dir` (default: `outputs/catboost_training/`).

### 4. Backtest and sweep

```bash
uv run python scripts/run_backtest.py \
    --output-dir outputs/step4_thread1/backtest \
    --oof-dir outputs/step4_thread1/catboost_training
```

Runs the fixed-universe baseline, gates it with each OOF score table at
20–80%, and writes reports to `--output-dir` (default: `outputs/backtest/`);
`--oof-dir` selects which OOF tables to gate with.

### 5. Reproducibility check (canonical tree)

```bash
uv run python scripts/verify_pipeline.py \
    --train-dir outputs/step4_thread1/catboost_training \
    --backtest-dir outputs/step4_thread1/backtest \
    --manifest outputs/step4_thread1/verification/reproducibility.json
```

Runs training and backtest twice and asserts the results are identical to the
saved canonical CSVs, byte for byte (~10 minutes). `--train-dir` /
`--backtest-dir` choose the tree to check and `--manifest` where the evidence
JSON is written. The frozen `outputs/verification/reproducibility.json` is the
**handover** manifest and is intentionally left untouched; the canonical
manifest lives in the `outputs/step4_thread1/verification/` tree. Running
`verify_pipeline.py` without arguments targets the frozen handover dirs and
overwrites `outputs/verification/reproducibility.json`; do not use it in the
canonical flow.

---

## Sealed holdout workflow

> Holdout evaluation is closed as of 2026-09-18. Do not run further training, scoring, backtesting, verification, or experiments on holdout. Commands below document the historical workflow only.

The workflow runs the holdout in four stages, then compares the two runs with a
fifth check. The three historical result versions are recorded in
[`outputs/verification/holdout_run_history.md`](outputs/verification/holdout_run_history.md).
The ledger is not an exhaustive execution log. Rechecks and the thread-count
experiment are additional holdout accesses. Run the stages in order.

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

# Independent run 2, written to the repro tree (does not touch stage3/stage4)
uv run python scripts/holdout_stage3_backtest.py --run-id 2 \
    --out-dir outputs/holdout/repro/run2/stage3
uv run python scripts/holdout_stage4_report.py \
    --stage3-dir outputs/holdout/repro/run2/stage3 \
    --out-dir outputs/holdout/repro/run2/stage4 \
    --report outputs/holdout/repro/run2/BAO_CAO_holdout_run2.md

# Stage 5 — compare run 1 and run 2 end to end (Stage 3–4)
uv run python scripts/holdout_stage5_repro_check.py
```

Stage 3 accepts `--run-id` and `--out-dir`; Stage 4 accepts `--stage3-dir`,
`--out-dir` and `--report` (plus `--canonical-dir`, `--sensitivity-json`,
`--run-history-json` and `--stage5-json` overrides). The run-2 tree is kept
under [`outputs/holdout/repro/run2/`](outputs/holdout/repro/run2/), and Stage 5
compares the two runs end to end: it reports **PASS** with all deltas 0.0 and
all four charts byte-identical
([`stage5_repro_report.json`](outputs/holdout/repro/stage5_repro_report.json)).

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
| Verified environment | Python 3.12.14 · catboost 1.2.10 · scikit-learn 1.9.0 · pandas 3.0.5 · numpy 2.5.2 · plotly 7.0.0 |

> `thread_count=1` is an **added technical setting**, not one of the original
> model hyperparameters. It was pinned on 2026-09-12 based on an initial
> technical hypothesis; the controlled experiment was added on 2026-09-15. It
> measures on this machine that two same-configuration `thread_count=1` runs are
> bit-identical (`max|Δp| = 0`), while moving the **training** `thread_count` to
> `2` or `-1` changes every prediction (train `max|Δp|` 0.2832 / 0.2999;
> holdout 0.3469 / 0.3352) and the top-50 holdout net R (−36.55 / −12.26 /
> +30.79 R). The effect is in training, not inference. This confirms the
> CatBoost documentation statement at prediction time but contradicts it at
> training time; scope is one machine, one CatBoost build, one dataset and one
> seed, so it does not attribute all cross-machine variation to thread count.
> The handed-over branch artifacts in `outputs/catboost_training/` and
> `outputs/backtest/` were produced without it and are preserved as-is (see
> Changelog).

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
  hashes because serialized metadata differs. The commit-`7748828` run was
  reproduced byte-for-byte by the canonical rerun (eight compared CSVs), and
  the frozen `verify_pipeline` handover manifest is left untouched while the
  canonical manifest lives in `outputs/step4_thread1/verification/`. The
  committed controlled experiment shows `thread_count` **does** change
  training results on this machine (train `max|Δp|` 0.2832 / 0.2999; holdout
  0.3469 / 0.3352; top-50 holdout net R −36.55 vs −12.26 vs +30.79 R for
  `tc=1` / `tc=2` / default), while repeated inference on a fitted model is
  unchanged. The pre-agreed top 50% remains fixed; the
  holdout sweep is not used to select a new retention rate. Cross-machine
  equality is **not established**: the Windows teammate reported predictions
  within `1e-15` (not byte-identical) with no committed artifact. Same-machine
  full-chain run1-vs-run2 (Stage 3–4) is PASS with all deltas 0.0 and
  byte-identical charts. The handed-over step-4 artifacts
  (`outputs/catboost_training/`, `outputs/backtest/`) are kept byte-identical
  and were produced on the original machine without `thread_count`, so
  the Linux rerun did not reproduce those original files byte for byte.
  This observation is not a guarantee of failure on every other host.

## Documentation

| File | Contents |
|---|---|
| [`docs/BAO_CAO_KET_QUA_HOLDOUT.md`](docs/BAO_CAO_KET_QUA_HOLDOUT.md) | Final holdout report (numbers, charts, verification) |
| [`docs/BAN_GIAO_task_holdout.md`](docs/BAN_GIAO_task_holdout.md) | Original holdout task spec |
| [`docs/BAN_GIAO_task_train_catboost.md`](docs/BAN_GIAO_task_train_catboost.md) | Original training task spec |
| [`docs/quy_trinh_lam_viec.md`](docs/quy_trinh_lam_viec.md) | Project working process |
| [`docs/handover_README.md`](docs/handover_README.md) | Original handover README |

## Changelog

### 2026-09-15 — Final report for Nhi: evidence scope and requirements check

- Updated the main results report and its generator: three Git-retrievable historical result versions, the 12 September hypothesis versus the 15 September controlled experiment, fixed top 50%, and the boundary-trade replay scope.
- Clarified the scope of the recorded holdout history. The feedback checklist was subsequently removed on 2026-09-18.
- Regenerated both report copies and the run-history ledger; numerical results and model settings remain unchanged. Cross-machine equality remains unestablished.
- The main deliverable for Nhi is [the results report](docs/BAO_CAO_KET_QUA_HOLDOUT.md).

### 2026-09-15 — Review feedback round: canonical thread-count rerun, controlled experiment, run history, full-chain verification

- **Canonical `thread_count=1` four-split rerun** (`54eed41`,
  `outputs/step4_thread1/`): `scripts/train_models.py` and
  `scripts/run_backtest.py` gained `--output-dir` / `--oof-dir` so the four
  splits can be rerun without touching the frozen handover trees. Canonical
  chunk-2–5 metrics: 0.8582/0.6494, 0.7454/0.5077, 0.5875/0.3288,
  0.5895/0.3247 (handover: 0.8595/0.6525, 0.7475/0.5105, 0.5867/0.3267,
  0.5948/0.3304); canonical top-50 net R +6,937.53 / +4,752.04 / +2,148.31 /
  +2,119.48 (baseline 20,007 trades, +3,358.25 R) vs handover +6,998.30 /
  +4,724.34 / +2,207.25 / +2,253.72. All eight compared CSVs are byte-identical
  to commit `7748828`; the handover comparison is in `comparison_report.json`.
- **Controlled `thread_count` experiment** (`b278617`,
  `outputs/thread_count_sensitivity/`): varying only `thread_count`, two
  `thread_count=1` runs are bit-identical (`max|Δp| = 0`), while
  `thread_count=2` / `-1` change every prediction (train `max|Δp|` 0.2832 /
  0.2999; holdout 0.3469 / 0.3352; Pearson 0.9331 / 0.9409). Holdout top-50
  net R: −36.55 (`tc=1`) / −12.26 (`tc=2`) / +30.79 (default). The effect is in
  training, not inference. CatBoost documents `thread_count` as a speed setting;
  that claim is confirmed at prediction time but contradicted at training time
  on this machine. Scope: one machine, one CatBoost build, one dataset, one
  seed.
- **Sealed-holdout run history recovered** (`a3ff712`,
  `outputs/verification/holdout_run_history.json` + `.md`): the three historical result versions
  reconstructed from git with full 20–80% sweeps and pairwise deltas —
  `dc25cd3` Windows 0.60502/0.40171, top-50 −23.83 R; `5f46e41` Linux
  multi-thread 0.60232/0.40647, +30.79 R; `7748828` Linux `thread_count=1`
  0.60455/0.40221, −36.55 R. It also records teammate Bùi Quốc Thịnh's Windows
  verification (predictions within `1e-15`, not byte-identical, no metrics
  recorded) as non-recomputable external evidence, and re-verifies all 12 known
  values to `1e-12`.
- **Full run1-vs-run2 chain verification** (`8003c82`,
  `outputs/holdout/repro/stage5_repro_report.json`): Stage 2/3/4 scripts gained
  `--run-id` / `--out-dir` / `--stage3-dir` / `--report`; run 2 Stage 3–4 was
  regenerated under `outputs/holdout/repro/run2/` and stage 5 compares the whole
  chain: **PASS**, all deltas 0.0, all four charts byte-identical. Stage 2
  reports now record OS (`platform`) and plotly version.
- **Canonical `verify_pipeline` manifest** (`8003c82`): `verify_pipeline.py`
  gained `--train-dir` / `--backtest-dir` / `--manifest`; the canonical run
  (training ×2, backtest ×2) is byte-identical to `outputs/step4_thread1/` and
  writes `outputs/step4_thread1/verification/reproducibility.json`. The frozen
  `outputs/verification/reproducibility.json` was left untouched.
- **Results report regenerated from artifacts** (`ee24520`,
  `docs/BAO_CAO_KET_QUA_HOLDOUT.md`): Table 1/2/3 main blocks now use the
  canonical tree with the handed-over values as a labelled reference and a
  delta table; Table 4 keeps the holdout 20–80% sweep; new run-history and
  thread-count sections; PNG charts embedded next to the HTML links; the four
  boundary orders documented (25,008 + 4 + 5,028 = 30,040) and the holdout
  period recorded as 2025-02-09 → 2026-08-21; verification table refreshed.
- **Refreshed the evidence invalidated by the source edits:** the run-history
  ledger was regenerated at HEAD `ee24520`; the canonical `verify_pipeline`
  command above was re-run **PASS** (two trainings + two backtests, canonical
  manifest refreshed); `holdout_stage5_repro_check.py` re-run **PASS**; the
  test suite is **13 passed**.
- **Best-practice fixes from the source audit**:
  `allow_writing_files=False` now lives in the single-source `MODEL_PARAMS` used
  by Stage 2 and the thread-count experiment (no `catboost_info/` is created);
  `matplotlib==3.11.2` is declared/pinned and its version recorded in the
  Stage 2/Stage 5 evidence; Stage 4 PNGs use `metadata={"Software": None}` so
  their bytes no longer depend on the matplotlib version, and Plotly uses the
  documented `include_plotlyjs=True`. All evidence (Stage 2–5 and the canonical
  `verify_pipeline` manifest) was regenerated with unchanged numbers and
  byte-identical predictions; the audit doc now has a Resolution section.

### 2026-09-13 — Finalize holdout handoff and evidence scope

- **Incorporated final report review:** documented the holdout re-run history
  and technical reason, clarified that Table 1 reports fold means after chunk 1
  is removed, and marked full-pipeline byte reproducibility as unproven **at
  the time** — superseded on 2026-09-15: the same-machine run1-vs-run2 chain is
  now PASS, while cross-machine equality remains limited. Also added a purpose
  description for every generated artifact.
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
  summary, and deterministic chart IDs. The final suite passed 5/5 tests at the
  time (13 tests as of 2026-09-15).
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
  and chart 1 use the frozen branch reference values (superseded on 2026-09-15:
  the Table 1/2 main blocks now use the canonical `thread_count=1` tree; the
  frozen branch values became the labelled reference block).
- **Fixed** chart 1 (`equity-curve-chunk2-5-top50.html`): it now builds from the
  restored handed-over `backtest_scored_universe.csv`, so its 4 branch endpoints
  match Table 2 exactly (+6,998.3 / +4,724.3 / +2,207.3 / +2,253.7 R)
  (superseded on 2026-09-15: chart 1 is rebuilt from the canonical
  `thread_count=1` scored universe
  `outputs/step4_thread1/backtest/backtest_scored_universe.csv`, endpoints
  +6,937.53 / +4,752.04 / +2,148.31 / +2,119.48 R).
- **Corrected** the Stage-4 report's file table (real script names and sizes) and
  synced `outputs/holdout/stage4/implementation_decisions.md` with the report.
- **Documented** the `thread_count=1` deviation from the frozen hyperparameter
  list as a technical (non-model) setting.

### 2026-09-12 — Pin `thread_count=1` (reduce one source of numerical drift)

- **Initial observation:** CatBoost was configured with `random_seed=42` but no
  `thread_count`, so execution thread count depended on the host. This is a
  plausible source of floating-point ordering variation. Earlier runs did
  differ, but at the time the repository contained no controlled experiment
  changing only `thread_count`, so it was not claimed as the sole root cause
  (superseded on 2026-09-15: the controlled experiment is now committed in
  `outputs/thread_count_sensitivity/` and shows changing `thread_count` changes
  training results on this machine).
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
