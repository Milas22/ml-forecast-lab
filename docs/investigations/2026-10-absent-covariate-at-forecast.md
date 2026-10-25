# Silent zeros at forecast time (v2.53.0)

**Status:** Shipped in v2.53.0. Defects where the recursive tree forecast
fed a model 0.0 for a feature it was trained on with real values, the fixes,
and why no model-cache schema bump.

The pinning tests: `ml-forecast-lab/tests/unit/test_absent_covariate_forecast.py`.
They train real `lightgbm` / `nlinear` models on synthetic history and read
what the model was fed from `diag["X_rows"]`. The end-to-end cases fail on
v2.52.7. Also `tests/unit/test_covariate_cache.py::test_failed_backfill_is_retried_next_cycle`,
whose precondition changed.

```bash
cd ml-forecast-lab/
python -m pytest tests/unit/test_absent_covariate_forecast.py tests/unit/test_covariate_cache.py -v
```

---

## TL;DR

| Defect (live up to v2.52.7) | Observed | Fix |
| --- | --- | --- |
| **A.** A covariate the model was trained on arrived with no column at forecast time | Tree model fed `0.0`, pinned `<col>_missing` zero-filled ("measured"), `row_dict.get(c, 0.0)` kept the nan_to_num backstop silent. Repro: trained on 5–15, fed 0.0; forecast moved 0.31 (2.12 for a gap-free-trained covariate) on a ~21 target | `_reinstate_trained_covariates` puts the column back as all-NaN; `resolve_missingness(empty_fill=…)` fills it with the cached `covariate_fill` (the training-window statistic) and flags it |
| **B.** `CovariateResolver._fetch_raw_history` let an HA error escape, and `fetch_history` then returned an empty series, discarding the SQLite rows it had read | The column vanished (A) on every HA hiccup, in training and benchmarks too | On an HA error with cached rows, serve the cache. The unreturned tail is masked, imputed and flagged like any gap. The backfill horizon is not recorded, so the next cycle retries |
| **C.** Neither recursive row builder set `is_holiday` | Tree experiments with `country` set always forecast holidays as ordinary days (repro: IE, 26 Oct 2026) | Both builders set `is_holiday(ts, exp_cfg.country)` when the feature is fitted |
| **D.** Covariates were picked by name shape (`startswith('y_')`, `endswith('_missing')`, `== 'sun_elevation'`) | `y_*` covariates fed 0.0 with their interactions; `*_missing` covariates' interactions fed 0.0; a lagged covariate column `sun_elevation` had every future row replaced by pvlib sun elevation | `_fitted_source_columns` reads sources off the fitted schema (every source has an `_x_hour_sin` pair); `_pinned_indicator_map` maps the cache's pinned indicators, including `_missing_flag` collisions; `_computed_source_columns` decides which solar columns are pvlib's from the config, and `_future_solar_frame` computes only those |

Both recursive builders now default an unproduced feature to NaN, so any
remaining hole reaches `_nan_to_num_guarded` and warns with its name instead
of becoming a silent 0.0.

## How the column went missing

- An HA error on `get_history` escaped `_fetch_raw_history`.
- `fetch_history` caught it and returned `pd.Series(dtype=float)`.
- `_fetch_and_preprocess` skips an empty series (`if cov_series.empty: continue`)
  and also drops a lagged-only covariate with 0% coverage.
- A covariate removed from `mlfl.yaml` while a cached model still uses it
  reaches the same state.

Training never shows this skew: training, benchmarks, tuning and covariate
analysis fit and score on one frame, so a covariate absent there is simply
not a feature. Only the cached forecast tick
(`_compute_cached_forecast`) pairs an old schema with a new frame.

## What training fed, and therefore what inference must

A missing covariate cell in training is masked and causally imputed
(`causal_impute`: expanding median of strictly prior observations, or
hold-forward for a binary channel), and flagged `<col>_missing` when an
imputed cell survives selection. A model never saw "0.0, flag 0" for a
covariate it was fitted on.

`_covariate_fill_values` stores, per fitted covariate, the median of the
training window's observations (the last one for a binary channel) as
`covariate_fill` in the cache entry and `cache_meta.json`. That is what
`causal_impute` would use at the end of the training window. An absent column
is filled with it and flagged everywhere, matching what the model met on its
own flagged rows.

## Scope decisions

- **Tree path only.** `_compute_cached_forecast` reinstates only when not on
  the sequence path. A neural model's past window would be wholly invented.
  - In the prototype, reinstating into `nlinear` published forecasts that
    moved by up to 109 (0.0 fill) or 908 (median fill) on a ~27–66 scale.
  - The channel-parity guard already refuses that forecast (`channel_mismatch`),
    so the neural path fails safe.
  - Fix B removes its common trigger.
- **Computed columns are decided from the config, not the name.**
  `_computed_source_columns(exp_cfg)` drives three things: no fill is
  stored for these columns, they are never reinstated, and only they get
  pvlib values on future steps.
  - `sun_elevation` belongs to the user only when a configured covariate
    resolves to that column and `include_sun_elevation` is off. When it is
    on, `_fetch_and_preprocess` writes the pvlib column over the sensor's.
  - Otherwise it is pvlib's. This includes a model trained with
    `include_sun_elevation` that is now off: it keeps getting pvlib values
    rather than a 0.0 fill.
  - `clear_sky_ghi` is always pvlib's. `build_features`' night gate keys on
    the name, so an all-NaN reinstated column would zero every lag. A
    sensor under that name is effectively a reserved name; that is
    unchanged.
  - The neural path is left name-based. Training and inference both build
    the future block's `sun_elevation` from pvlib, so they agree with each
    other.
- **Gap-free experiments are unchanged.**
  - Nothing absent returns the same frame object.
  - `empty_fill` stays `None`.
  - The schema-derived source list and pinned map equal the old name filter
    and suffix map for ordinary names.
  - The computed set is both solar columns unless a covariate resolves to
    `sun_elevation`, so `_future_solar_frame` computes what it did before.
  - The adversarial check measured `X_train` bit-identical across 13 healthy
    scenarios. Healthy-tick rows and `y_pred` were bit-identical apart from
    the intended C and D corrections.

## No schema bump

`CACHE_SCHEMA_VERSION` exists to drop models whose training contract no longer
matches inference (v2.37, v2.51.0 changed training-side features and windows).
Here training is unchanged and every changed input moves toward what the old
model was fitted on:
- holiday rows were trained with `is_holiday`=1;
- `y_axis` was trained with its real values;
- flagged rows were trained with median fills.

A retrain would rebuild the same model and cost every user a full retrain on
update. `CLAUDE.md`'s rule now says so explicitly.

`covariate_fill` is optional. Older `cache_meta.json` files restore it as `{}`,
and a reinstated column then gets 0.0 with its flag raised: still logged and
flagged, never worse than before. The next scheduled retrain writes it.

## Limits that remain

- Fix B changes training and benchmark frames when HA errors on a covariate
  while its cache is warm. The covariate is now present, with a flagged stale
  tail, instead of absent (measured: 33 → 37 training columns). This is
  corrective, but those runs' rankings can differ.
- Fix B also makes a covariate failure quieter: a transient error logs one
  WARNING rather than the "No data for covariate" / 0%-coverage warnings.
- A model trained gap-free on a covariate has no pinned indicator, so its
  fill is fed unflagged. This is the same accepted behaviour as a
  present-but-NaN tail today. The reinstatement warning lists flagged and
  unflagged covariates separately.
- **Found in the same review, not fixed here:**
  - `cov_cfg.transform` is applied to covariate history but not to future
    values. The future paths apply only `scale`, so a future-role covariate
    with a `transform` gets future values on a different scale from training.
  - Both recursive builders compute `y_rolling_std_*` with `np.std`
    (`ddof=0`), while `build_features` uses pandas rolling std (`ddof=1`).
    That is about 9% low for the 6-row window at 30-minute data, on every
    tree forecast.
