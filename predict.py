#!/usr/bin/env python3
"""
predict.py - score a Dota 2 win-probability model on one or more feature rows.

Each model is a LightGBM booster (model.txt) plus a model_meta.json listing the
exact feature names, in order.  This script loads a model, lines your input
columns up to that model's feature_names (by name), and prints the predicted
Radiant win probability per row.

Usage
-----
  # score the bundled example row with the headline model
  python predict.py herowr example_input.csv

  # any model dir works (herowr / max482 / micro19 / min3)
  python predict.py min3 example_input.csv

Input formats accepted
----------------------
  * a 2-column "feature,value" CSV (the example_input.csv shape) -> 1 row
  * a wide CSV with a header row of feature names, one game-minute per row

Output: one probability in [0, 1] per row (Radiant win probability).

Feature spec: see README.md.  Slot order is 0-4 Radiant, 5-9 Dire.  Net worth
is raw gold; *_diff features are Radiant-minus-Dire; towers_dead_*/rax_dead_*
are cumulative counts.  Any feature you omit defaults to 0.0 (fine for the few
hero-prior columns the 482-feature model uses; required structural features
should always be supplied).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np


def load_model(model_dir: Path):
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    meta = json.loads((model_dir / "model_meta.json").read_text())
    return booster, meta["feature_names"], meta.get("best_iteration")


def read_rows(path: Path):
    """Return (feature_names_or_None, list_of_dicts)."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f"empty input: {path}")
    header = rows[0]
    # "feature,value" tall format
    if len(header) == 2 and header[0].lower() == "feature" and header[1].lower() == "value":
        d = {r[0]: float(r[1]) for r in rows[1:] if len(r) == 2 and r[1] != ""}
        return [d]
    # wide format: header = feature names
    out = []
    for r in rows[1:]:
        out.append({h: (float(v) if v != "" else 0.0) for h, v in zip(header, r)})
    return out


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: python predict.py <model_dir> <input.csv>")
    model_dir = Path(argv[1])
    if not (model_dir / "model.txt").exists():
        # allow running from anywhere: resolve relative to this file
        model_dir = Path(__file__).parent / argv[1]
    booster, feat_names, best_iter = load_model(model_dir)
    rows = read_rows(Path(argv[2]))

    X = np.zeros((len(rows), len(feat_names)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, name in enumerate(feat_names):
            X[i, j] = row.get(name, 0.0)

    p = booster.predict(X, num_iteration=best_iter)
    for i, prob in enumerate(np.atleast_1d(p)):
        print(f"row {i}: radiant_win_prob = {prob:.4f}")


if __name__ == "__main__":
    main(sys.argv)
