# Raw market data

This folder holds the **raw BTCUSD M1 CSV** used by the whole pipeline.

## Why the file is missing from the repository

`BTCUSD_m1_2018_to_now.csv` is about **209 MB**. GitHub rejects any file
larger than 100 MB, so the raw file is intentionally ignored by `.gitignore`.

## How to restore it

1. Obtain `BTCUSD_m1_2018_to_now.csv` from the project owner / original data
   provider.
2. Place the file here with exactly this name:

   ```
   data/raw/BTCUSD_m1_2018_to_now.csv
   ```

3. The pipeline is ready. You can verify it is read correctly with:

   ```bash
   uv run python scripts/build_dataset.py --holdout 2100-01-01 \
       --output outputs/holdout/stage1/dataset_catboost_regenerated_check.csv
   ```

## Expected format

MetaTrader-style M1 export, comma separated, with a header row:

| Column | Example | Notes |
|---|---|---|
| `Date` | `2018.01.04` | format `%Y.%m.%d` |
| `Time` | `00:00:00` | format `%H:%M:%S` |
| `Open` | `15103.8` | numeric |
| `High` | `15120.0` | numeric |
| `Low` | `15090.0` | numeric |
| `Close` | `15110.0` | numeric |
| `Volume` | `12` | numeric |

Coverage: **2018-01 → 2026-08**, including the sealed holdout period
(`entry_time >= 2025-02-08 15:30:00`).

## Committed derivatives

Because the raw file is not committed, the pipeline outputs that are needed to
inspect results are committed instead under `data/processed/` and `outputs/`
(see the root `README.md`).
