"""Kiem tra tinh toan ven cua dataset_catboost.csv vua sinh lai.

Chay: python verify_dataset.py
Moi assert deu phai pass. Neu fail -> dung lai, khong duoc di tiep.
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from citd_ml import paths
from citd_ml.features.build_features import FEATURES, META

HOLDOUT_START = pd.Timestamp(paths.HOLDOUT_START)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{('  -> ' + detail) if detail else ''}")
    return ok


def main() -> int:
    path = paths.DATASET_CSV
    raw = path.read_text(encoding="utf-8")
    d = pd.read_csv(path, parse_dates=["entry_time", "label_end_time"])

    print(f"File   : {path}")
    print(f"SHA256 : {hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"Size   : {path.stat().st_size:,} bytes\n")

    ok = True

    print("-- Dinh dang file (phat hien hu hong do Excel) --")
    ok &= check("datetime dang ISO (khong bi Excel doi sang M/D/YYYY)",
                "2018-01-04 00:15:00" in raw,
                "mau: " + raw.splitlines()[1].split(",")[24])
    ok &= check("float giu du 8 chu so thap phan (khong bi Excel cat)",
                ".00000000" in raw or raw.count(".") > 0 and "15103.80000000" in raw,
                "entry_price leg0 dau tien")
    ok &= check("khong co ky hieu khoa hoc (1e-05)", "e-" not in raw.lower().replace("time", ""))

    print("\n-- Kich thuoc & cau truc --")
    ok &= check("25,008 dong", len(d) == 25008, f"{len(d):,}")
    ok &= check("31 cot (23 feature + 1 label + 7 meta)", d.shape[1] == 31, str(d.shape[1]))
    ok &= check("thu tu cot = FEATURES + label + META",
                list(d.columns) == FEATURES + ["label"] + META)
    ok &= check("dung 23 feature", len(FEATURES) == 23, str(len(FEATURES)))
    ok &= check("6,252 gia dinh", d["origin_bar"].nunique() == 6252, f"{d['origin_bar'].nunique():,}")
    ok &= check("moi gia dinh dung 4 leg", d.groupby("origin_bar").size().eq(4).all())
    ok &= check("leg chi nhan 0..3", sorted(d["leg"].unique()) == [0, 1, 2, 3])

    print("\n-- Nhan --")
    r = d["label"].mean()
    ok &= check("label chi 0/1", set(d["label"].unique()) <= {0, 1})
    ok &= check("ty le nhan 1 = 26.1596%", abs(r - 0.261596) < 1e-4, f"{r:.6%}")
    for lg in range(4):
        s = d.loc[d["leg"] == lg, "label"].mean()
        print(f"       leg {lg}: n={int((d['leg']==lg).sum()):,}  label1={s:.4%}")

    print("\n-- Toan ven du lieu --")
    ok &= check("0 NaN tren toan bang", int(d.isna().sum().sum()) == 0)
    ok &= check("khong trung (entry_time, leg)", not d.duplicated(["entry_time", "leg"]).any())
    ok &= check("moi feature deu la so", all(pd.api.types.is_numeric_dtype(d[c]) for c in FEATURES))
    ok &= check("khong feature nao co inf",
                not np.isinf(d[FEATURES].to_numpy(dtype=float)).any())
    ok &= check("label_end_time >= entry_time", bool((d["label_end_time"] >= d["entry_time"]).all()))
    ok &= check("da sort theo entry_time", d["entry_time"].is_monotonic_increasing)

    print("\n-- Holdout con niem phong --")
    ok &= check("moi entry_time < 2025-02-08 15:30",
                bool((d["entry_time"] < HOLDOUT_START).all()), f"max={d['entry_time'].max()}")
    ok &= check("moi label_end_time < 2025-02-08 15:30",
                bool((d["label_end_time"] < HOLDOUT_START).all()), f"max={d['label_end_time'].max()}")

    print("\n-- Mien gia tri feature --")
    ok &= check("hour trong 0..23", d["hour"].between(0, 23).all(), f"{d['hour'].min()}..{d['hour'].max()}")
    ok &= check("dow trong 0..6", d["dow"].between(0, 6).all(), f"{d['dow'].min()}..{d['dow'].max()}")
    ok &= check("pos_in_range50 trong [0,1]", d["pos_in_range50"].between(0, 1).all())
    ok &= check("breakeven_R > 0", (d["breakeven_R"] > 0).all())
    ok &= check("rsi14 trong [0,100]", d["rsi14"].between(0, 100).all())

    print("\n-- Ranh gioi 5 fold (walk-forward) --")
    n = len(d)
    edges = [int(n * k / 5) for k in range(6)]
    ok &= check("edges = [0, 5001, 10003, 15004, 20006, 25008]",
                edges == [0, 5001, 10003, 15004, 20006, 25008], str(edges))
    for k in range(5):
        s = d.iloc[edges[k]:edges[k + 1]]
        print(f"       fold {k+1}: idx {edges[k]:>6}-{edges[k+1]-1:<6} n={len(s):>5}  "
              f"{s['entry_time'].min()} -> {s['entry_time'].max()}  label1={s['label'].mean():.2%}")

    print("\n-- Ghep duoc voi tradelist (chuan bi cho backtest) --")
    tl_path = paths.TRADELIST_CSV
    if tl_path.exists():
        tl = pd.read_csv(tl_path, parse_dates=["open_time", "close_time", "signal_time"])
        tl = tl.loc[tl["open_time"] < HOLDOUT_START]
        ok &= check("(open_time, leg) duy nhat trong tradelist",
                    not tl.duplicated(["open_time", "leg"]).any())
        m = d.merge(tl[["open_time", "leg", "R"]], left_on=["entry_time", "leg"],
                    right_on=["open_time", "leg"], how="left")
        ok &= check("khop 100% voi tradelist", bool(m["R"].notna().all()),
                    f"{m['R'].notna().mean():.4%}")
    else:
        print("  [SKIP] khong tim thay tradelist_pyramid_local.csv")

    print("\n" + "=" * 60)
    print("KET QUA: " + ("TAT CA PASS" if ok else "CO ASSERT FAIL -> DUNG LAI"))
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
