# Forecast inputs on a different scale from training (v2.54.0)

**Status:** Shipped in v2.54.0, in the same release as
[`2026-10-absent-covariate-at-forecast.md`](2026-10-absent-covariate-at-forecast.md),
whose review found both defects. Both are inference-only: training frames,
row counts and sliding windows are unchanged, so benchmarks and rankings are
unchanged and there is no model-cache schema bump.

The pinning tests:
- `ml-forecast-lab/tests/unit/test_covariate_transform_scale.py`: real
  `lightgbm` / `nlinear` models on synthetic signed covariates, with a stubbed
  `forecast` attribute;
- `ml-forecast-lab/tests/unit/test_rolling_std_parity.py`: both recursive
  builders, against pandas' rolling std on the grid.

```bash
cd ml-forecast-lab/
python -m pytest tests/unit/test_covariate_transform_scale.py tests/unit/test_rolling_std_parity.py -v
```

---

## TL;DR

| Defect (live up to v2.52.7) | Observed | Fix |
| --- | --- | --- |
| **A.** `cov_cfg.transform` applied to covariate history in `_fetch_and_preprocess`, but the four future-value paths (`_run_production_inference` / `_compute_cached_forecast` × tree / neural) applied `scale` only | A future-role covariate with `transform: log` was trained on [1.32, 2.46] and fed raw forecasts in [-2.00, 5.15]. The neural past block was in log space and the future block raw. Tree MAE at the training instant: 0.172 as-is vs 0.074 fixed | `_future_covariate_on_model_scale`: `scale`, then the recorded transform |
| **B.** The `log` / `box_cox` shift is chosen from the window minimum, and each forecast tick re-fetches its own window | Signed series: the shift moved 5.733 → 2.996 when a cold snap left the window, so x=0 moved 0.65 in a channel 1.14 wide. A positive series with rare exact zeros flips the `log` shift 0 ↔ 1, moving the latest value a median 0.98 log units (~5 training sd) | `covariate_transforms`: each covariate's `{transform, shift, min}` recorded at retrain, stored with the model, applied to history and future values |
| **C.** Both recursive row builders computed `y_rolling_std_*` with `np.std` (`ddof=0`); `build_features` uses pandas' rolling std (`ddof=1`) | Every tree forecast row was fed `sqrt((w-1)/w)` of the trained statistic: 8.7% low for the 6-row window, 4.3% for 12, 1% for 48 | `np.std(window, ddof=1)`, and 0.0 for a single-value window |

## A and B: the transform

### Where it was applied

- `preprocessing.apply_transform` picks the shift from the series:
  - `log`: 0 when min > 0, 1 when min == 0, |min| + 1 when min < 0;
  - `shifted_log` / `box_cox`: 1 when min ≥ 0, otherwise |min| + 1;
  - `sqrt`: clip at 0, no shift.

  The shift lived only in `series.attrs` and was never persisted.
- `_fetch_and_preprocess` applies `scale`, then the transform, to the
  resampled window `[now - days_history, now]`. Every consumer goes through
  it.
- Benchmark, tuning and covariate analysis use one window's transformed
  history throughout, and the neural future block in training comes from
  `_collect_train_future_covariates`. They never call `fetch_future`, so
  they were self-consistent. Fixing A and B leaves them bit-identical.
- `fetch_future` callers for covariates are the four future-value paths only.
  `_capture_external_forecast` also calls it, but for target forecasts, with
  no scale or transform.
- The target's `log_transform` uses a fixed shift of 1 and cannot drift.

### How often B bites

A one-year synthetic 30-minute series, `days_history=14`, a daily retrain,
comparing each tick's shift with the one at the last retrain:

| Series | transform | Ticks with a different shift | Latest-value offset (transformed) |
| --- | --- | --- | --- |
| Outdoor temperature (signed) | `log` | 7.5% | median 0.14 (0.28 training sd), p95 0.68, max 8.0 |
| Signed grid power | `log` | 6.6% | median 0.046, p95 0.30 |
| kW load, 4–12 one-hour zeros a year | `log` | 1.4–1.9% | median 0.98 (~5 sd), p95 12–13 sd |
| PV with zeros | `log`, `box_cox` | 0% | none (shift always 1) |
| Humidity, always > 0 | `log` / `box_cox` | 0% | none (shift always 0 / 1) |

The max of 8.0 comes from the window minimum crossing zero, which flips the
`log` shift between 0 and |min| + 1. NLinear barely reacted (≤ 0.04), as it
leans on the target channel; LightGBM moved substantially.

### Design

- **Record at retrain, apply at inference.**
  - `_retrain_and_cache` passes a fresh `{}` into `_prepare_training_frame`
    → `_fetch_and_preprocess`, which writes one record per covariate.
  - The record is stored as `covariate_transforms` in the cache entry and
    `cache_meta.json`.
  - A retrain never receives the previous map: pinning an old shift into
    training would change training frames.
- **The map covers every covariate, including untransformed ones**
  (`{"transform": None}`), so a stored map is authoritative.
  - The first prototype keyed only transformed covariates, so `{}` meant
    both "old cache" and "trained untransformed".
  - Adding `transform: log` to the YAML after training, then restarting,
    then fed log values to a model fitted on raw ones at every horizon step.
  - A recorded transform now wins over the YAML until the next retrain, and
    an INFO line says so each tick.
- **Pinned floor.** With a recorded shift `S` and minimum `m`, the pinned map
  is `log(max(x + S, min(1, m + S)))`.
  - It equals the unpinned output on the training series (bit-identical,
    including after a JSON round-trip of the record).
  - It is monotone and stays finite below the trained range. A value below
    `min(1, m)` with `S = 0` is floored instead of going towards −inf.
  - No value that a window-derived shift of 1 or more would accept is moved.
- **Incomplete records.** An all-NaN window gives shift NaN. The record is
  then `{"transform": t}` with no shift: it is re-derived per window and
  never written to JSON as NaN. A future value whose log shift cannot be
  reproduced returns None, and the covariate falls back to carry-forward
  rather than be fed raw.
- **Failed fresh fetch.** The tick resets the map to the cache's record,
  since `cache["combined"]` went through the training transforms. A partial
  write from the failed fetch is discarded.
- **Old caches** (no key) restore it as `{}`.
  - Each tick derives the history shift exactly as v2.52.7 did
    (bit-identical).
  - The future values get that same tick's record, so history and future
    agree within a tick.
  - The exact pin arrives at the next scheduled retrain.

### Verified unchanged

On a copy with the fix, against an untouched tree, with the same persisted
model loaded in both (LightGBM and NLinear, ticks at +0/6/12/24 h):
- **Bit-identical:**
  - no transform configured (grid, frame, rows, windows, X_rows / neural
    window, `y_pred`);
  - the training side in every scenario;
  - history and lagged-only channels with a stable shift (PV with zeros,
    positive humidity, `box_cox` with min ≥ 0, `sqrt`);
  - an old cache's history grid.
- **Intended changes:** the future positions of transformed future-role
  covariates; history from the first tick whose own window would have chosen
  a different shift.

### No schema bump

- Training is unchanged, so a retrain would rebuild the same model.
- Every changed input moves towards what the model was fitted on.
- A global bump would retrain every experiment on every Pi, though
  `transform` is YAML-only and rare, and would make every existing replay
  bundle's forecast stage "not comparable".

The `CLAUDE.md` rule covers this case.

## C: rolling std

- `build_features` computes `y_rolling_std_{w}` as
  `target.shift(1).rolling(w).std()`, which is pandas' sample std (`ddof=1`).
- Both recursive builders used `np.std(window)`, the population std
  (`ddof=0`), on a buffer seeded from the complete grid.
- Benchmarks score tree models with `model.predict(X_test)` on rows built by
  `build_features`, so rankings were never affected.
- Only production forecasts moved, every tree forecast row, by the constant
  factor `sqrt((w-1)/w)`.
- The fix uses `ddof=1`, and 0.0 when the window holds one value: training
  never sees a partial window, and pandas would give NaN.

## Limits that remain

- **`scale` is not pinned.** A `scale` edited after training still rescales
  history and future values at once under the old shift. This is
  pre-existing config drift.
- **Binary covariates are transformed.** `transform` applies to `is_binary`
  covariates although the Binary tooltip says binary covariates skip it.
  Skipping it would change training frames, so it is a separate change.
- **Missing neural future values.** When a neural future-role covariate has
  no forecast, its future block is a silent 0: raw value `1 - S` once
  transformed. Pinning makes this consistent but does not fix it.
- **Replay bundles captured before v2.54.0** report forecast-input
  differences (exit 1) for experiments using a covariate `transform`. This is
  expected.
