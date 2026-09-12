# Legacy scripts

Archived implementations kept for reference. They are **not** part of the
current pipeline.

| File | Note |
|---|---|
| `score_and_backtest_holdout.py` | Earlier holdout stage-3 variant that reused functions from `backtest_pyramid_local.py`. Superseded by `scripts/holdout_stage3_backtest.py`, which replays the strategy standalone and produced the results in `outputs/holdout/stage3/`. |

Paths inside these files point at the pre-refactor `Codex/` working tree and
will not run as-is.
