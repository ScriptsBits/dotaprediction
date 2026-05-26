# Dota 2 Win-Probability Models

LightGBM models that predict **Radiant win probability** from a single
game-minute snapshot. These are the models behind the win-probability write-up.
Everything here is self-contained — you can verify the published metrics and
score your own game-minutes without the raw replay data.

## Models

| dir | features | what it is |
|-----|----------|------------|
| `herowr/`  | 482 | **Headline "Our Model"** — net worth + XP + tempo + towers/rax + buyback + hero win-rate priors + rank. The one in the hero-compare chart and the LightGBM scorecard. |
| `max482/`  | 482 | The "Max" model from the pro-generalization test (`player_lgbm_herowr_rank`). |
| `micro19/` | 19  | Net worth (per slot + team) and towers/rax only. The ablation endpoint that keeps almost all of the signal. |
| `min3/`    | 3   | `{nw_d, nw_diff, rax_dead_D}` — the irreducible core. Still beats a coin flip by a mile. |

`herowr` and `max482` are two 482-feature experiments that converged — they give
identical predictions on this pub test split, so `evaluate.py` reports the same
metrics for both. They're kept separate because `max482` is the booster used in
the pro-generalization comparison.

`micro19` and `min3` were retrained from the cached pub training matrix with the
ablation recipe (see `../../Win Probability Analysis/make_model_bundle.py`); the
other two are the boosters as trained.

Each model dir contains:
- `model.txt` — the LightGBM booster. Reload: `lgb.Booster(model_file="model.txt")`
- `model_meta.json` — `feature_names` (exact order), params, `best_iteration`
- `test_preds.npz` — `{t, y, p_model, p_valve}` over the shared 374,237-row pub test split

## Quick start

```bash
pip install lightgbm numpy scikit-learn      # scikit-learn only used by the trainer

# 1) reproduce the headline metrics straight from shipped predictions
python evaluate.py
#   -> Acc / AUC / Brier / ECE per model, plus the Valve baseline

# 2) score one feature row with any model
python predict.py herowr example_input.csv
python predict.py min3    example_input.csv
```

`example_input.csv` is one real pub-test game-minute (482 features, canonical
order). `example_input.meta.json` gives that row's true outcome and Valve's
prediction so you can sanity-check the output.

## Scoring your own game-minute

`predict.py` aligns your input columns to a model's `feature_names` **by name**,
so you only need to supply the features that model uses (any missing column
defaults to `0.0`). Two input shapes are accepted:
- tall `feature,value` CSV (like `example_input.csv`) → one row
- wide CSV: header row of feature names, one game-minute per row

### Feature spec (conventions)
- **Slots**: `*_s0..s9`; slots 0–4 = Radiant, 5–9 = Dire.
- **Net worth** `nw_*`: raw gold. `nw_r`/`nw_d` = team totals, `nw_diff = nw_r - nw_d`.
- **XP** `xp_*`, **last hits** `dlh*`: same slot/team layout (used by larger models).
- **Tempo** `d60_*`/`d300_*` (or `dxp60_*`, `dlh60_*`): change over the last 60 s / 300 s.
- **Buildings**: `towers_dead_R`/`towers_dead_D` (0–11), `rax_dead_R`/`rax_dead_D` (0–6),
  cumulative. `towers_diff = towers_dead_D - towers_dead_R` (>0 = Radiant ahead).
- **Buyback** `bb_*`: 1 if available, else 0.
- **Time** `t`: game time in seconds (negative = pre-horn).

The full per-model order is in each `model_meta.json`. For the smaller models the
list is short — `min3` needs only `nw_d`, `nw_diff`, `rax_dead_D`.

## CSV files

### In this bundle
- **`example_input.csv`** — one real pub-test game-minute as `feature,value` rows
  (482 features, canonical order). Demo input for `predict.py`. Its true outcome
  and Valve's prediction are in `example_input.meta.json`.
- **`herowr/feature_importance.csv`** — gain-based importance for the headline
  model. Columns: `feature`, `gain` (total split-gain attributed to the feature),
  `split` (number of times it was used to split). Ranked high → low.

### Chart data (parent `blog_stills/` folder)
One CSV per still image, holding the exact numbers plotted (so the figures are
reproducible). Accuracy/win-rate values are fractions 0–1 unless a column name
ends in `_pct` (already ×100). `_pp` columns are percentage-point gaps.

- **`hero_accuracy.csv`** — Valve accuracy per hero, best→worst. Columns
  `rank, hero, valve_acc, win_rate, valve_mean_p, delta_pp, n_games`;
  `delta_pp` = (win_rate − valve_mean_p) in percentage points.
- **`hero_compare.csv`** — Valve vs our model per hero. Columns
  `rank, hero, valve_acc, model_acc, edge_pp`; `edge_pp` = (model_acc − valve_acc) in pp.
- **`hero_overunder.csv`** — 5 biggest over- then 5 biggest under-performers vs our model.
  Columns `hero, win_rate, model_mean_p, gap_pp, n_games`; `gap_pp` = (win_rate − model_mean_p) in pp.
- **`lgbm_scorecard.csv`** — LightGBM vs Valve headline metrics (full-game window).
  Columns `metric, lgbm, valve, delta, better`; `delta` = lgbm − valve, `better` ∈ {`lgbm`, `valve`}.
- **`winprob_scorecard.csv`** — Valve metrics on pub games. `metric, value`
  (`n_samples`, then `acc, auc, brier, ece`).
- **`pro_scorecard.csv`** — Valve metrics on pro games. `metric, value`
  (`games`, `n_samples`, then `acc, auc, brier, ece`).
- **`pro_skill_by_time.csv`** — Valve skill per game minute, pro games.
  `minute, accuracy_pct, auc_pct`.
- **`pro_error_by_time.csv`** — Valve error per game minute, pro games.
  `minute, brier, logloss` (raw, lower = better).
- **`skill_by_progress.csv`** — Valve skill by % through the game, pro vs pub.
  `pct, pro_accuracy_pct, pro_auc_pct, pub_accuracy_pct, pub_auc_pct`.

## Notes
- **Valve baseline**: `p_valve` in each npz is Valve's own in-client win probability
  on the same rows — that's the comparison line in the charts.
- **Split**: a single match-level train/test split is shared across all models, so
  rows line up across the four `test_preds.npz` files.
- **Pro generalization**: on out-of-domain pro games the tiny `micro19`/`min3`
  models actually *beat* `max482` (pub hero priors + missing rank hurt the big model),
  and all of them beat Valve on AUC. `test_preds.npz` here is the **pub** test split;
  rerun against pro data to reproduce that result.
