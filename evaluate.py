#!/usr/bin/env python3
"""
evaluate.py - reproduce each model's headline metrics from its shipped predictions.

Every model dir contains test_preds.npz with arrays {t, y, p_model, p_valve} over
the same held-out pub test split (374,237 game-minutes).  This recomputes
Accuracy / AUC / Brier / ECE straight from those arrays -- no raw data, no model
load required -- so you can verify the numbers behind the charts.

It also prints the Valve baseline (p_valve) once, for reference.

Usage
-----
  python evaluate.py            # all model dirs next to this script
  python evaluate.py herowr     # one model
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MODELS = ["herowr", "max482", "micro19", "min3"]


def auc(y, p):
    # Mann-Whitney U, no sklearn dependency
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = float(y.sum()); n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p, y, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    n = len(p)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else \
            (p >= bins[i]) & (p <= bins[i + 1])
        k = int(m.sum())
        if k:
            out += k / n * abs(p[m].mean() - y[m].mean())
    return float(out)


def metrics(y, p):
    acc = float(np.mean((p >= 0.5) == (y >= 0.5)))
    brier = float(np.mean((p - y) ** 2))
    return acc, auc(y, p), brier, ece(p, y)


def report(name, y, p):
    a, u, b, e = metrics(y, p)
    print(f"  {name:9s}  acc={a:.4f}  auc={u:.4f}  brier={b:.4f}  ece={e:.4f}  (n={len(y):,})")


def main(argv):
    wanted = argv[1:] or MODELS
    valve_done = False
    for name in wanted:
        npz = HERE / name / "test_preds.npz"
        if not npz.exists():
            print(f"  {name}: missing test_preds.npz - skipped")
            continue
        d = np.load(npz)
        y = d["y"].astype(np.float64)
        if not valve_done and "p_valve" in d:
            report("Valve", y, d["p_valve"].astype(np.float64))
            valve_done = True
        report(name, y, d["p_model"].astype(np.float64))


if __name__ == "__main__":
    main(sys.argv)
