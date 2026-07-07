# Pro win-probability model

The other model dirs (`herowr/`, `max482/`, `micro19/`, `min3/`) are trained on
**public matchmaking** replays. This one is trained directly on **professional**
matches — role-indexed net worth plus tower/rax state, sourced from OpenDota's
pro match JSON rather than parsed replays.

Deployed model: **9 features**, LightGBM, Platt-calibrated.

```
nw_r_p2, nw_r_p3, nw_d_p2, nw_d_p3, nw_r, nw_diff,
towers_dead_R, towers_dead_D, rax_diff
```

`_p2`/`_p3` = Mid / Offlane Core role net worth (assigned per game from
OpenDota's `lane_role` + laning-phase net worth ranking — see
`model_meta.json` for the exact feature list and Platt params). Carry and
support net worth add no unique signal once team totals are present; only
Mid and Offlane Core survive per-role.

## Why a separate pro model

Pub-trained models don't transfer cleanly to pro games — different pacing,
different hero-winrate priors, no comparable rank signal. This model is
trained and evaluated entirely on pro data (963 train / 159 test games,
split by date so test is strictly later than train, no leakage).

## Results

Held-out pro test set (159 games / 6,512 rows), see `pro_model_comparison.csv`:

| model | features | AUC | ACC | Brier | ECE |
|---|---|---|---|---|---|
| **This model — no_rank Full(19)** | 19 | **0.827** | 0.734 | 0.172 | 0.049 |
| This model — Min(3) | 3 | 0.836 | 0.737 | 0.171 | 0.073 |
| with_rank Full(25) | 25 | 0.818 | 0.737 | 0.183 | 0.090 |
| Valve (disjoint reference set) | — | 0.767 | 0.682 | 0.194 | 0.025 |

The Valve row above comes from a disjoint 495-game set (OpenDota JSON carries
no `valve_p`), so the ECE comparison isn't apples-to-apples. A matched
comparison on the 117 games present in both sources (`matched_holdout_compare.csv`)
closes that gap and still wins on every metric, including calibration:

| model | AUC | ACC | Brier | ECE |
|---|---|---|---|---|
| **This model (Platt)** | **0.815** | **0.738** | **0.176** | **0.042** |
| Valve | 0.779 | 0.693 | 0.192 | 0.065 |

Adding team priors (most-recent Elo64/Glicko2) hurts every metric overall
(`with_rank` rows above) — full-game training dilutes the early-game rating
signal those features carry. See the project notes for a per-minute breakdown;
ratings only help in minutes 1-10.

## Loading

```python
import lightgbm as lgb, json

booster = lgb.Booster(model_file="pro/model.txt")
meta = json.loads(open("pro/model_meta.json").read())
p_raw = booster.predict(X)  # X columns = meta["feature_names"], in order

# apply Platt calibration (recommended - improves ECE, monotone so AUC unchanged)
import numpy as np
a, b = meta["platt"]["a"], meta["platt"]["b"]
logit = np.log(p_raw / (1 - p_raw))
p_calibrated = 1 / (1 + np.exp(-(a * logit + b)))
```
