# Forecast sensors unavailable after a restart (v2.52.7)

**Status:** Shipped in v2.52.7. The mechanisms behind "some forecasts are
unavailable after a restart", the fixes, the design choices and what remains,
behind that release's CHANGELOG entries.

The pinning tests: `ml-forecast-lab/tests/unit/test_restart_availability.py`.

```bash
cd ml-forecast-lab/
python -m pytest tests/unit/test_restart_availability.py -v
```

---

## TL;DR

| Defect (live up to v2.52.6) | Observed | Fix |
| --- | --- | --- |
| **A.** Nothing re-posted the app's sensors. HA keeps no REST-posted state across its own restart, and the app neither detected the restart nor published at startup | Every `sensor.mlfl_*` missing until that experiment's next forecast: 0–30 min after an HA restart, ~25–30 min after a reboot | `_check_ha_states` re-posts the last payloads once HA has dropped `sensor.mlfl_last_run`; a restored model forecasts `STARTUP_FORECAST_DELAY_SECONDS` (2 min) after start |
| **B.** `startup: services` started the app before Core; the Supervisor proxy answers 502 until Core is `RUNNING` | An experiment with no usable saved model retrained at once, failed on the 502, and was not retried for `retrain_every_hours`: no forecasts for up to a day | `startup: application`; an `HAUnavailableError` retrain or forecast failure retries after `HA_RETRY_MINUTES` (2, 5, 15, 60, capped at the cadence) |
| **C.** A retrain replacing a pinned `production_model` deleted the old cache entry first | A failed replacement left the experiment with nothing to forecast with | The old entry is restored if the retrain fails |
| **D.** `_cumulative` (signed predictions) and the bands (before calibration) were omitted | HA showed their last value until a restart removed them; a PV model that dips below zero at night rarely has an unsigned 24 h horizon, so `_cumulative` stayed missing | Posted as `unknown` with a `status` attribute (`signed_predictions`, `no_conformal_bands`); bands under the configured `conformal_coverage` level's ids |
| **E.** `_last_retrain` and `sensor.mlfl_last_run` were only posted by a retrain | Both missing for up to `retrain_every_hours` after any HA restart | The heartbeat is posted with every successful forecast; `_restore_lifecycle_sensor` re-posts `_last_retrain` when HA lacks it |

## Mechanisms

**HA side.** States POSTed to `/api/states` have no entity-registry entry and
no `RestoreEntity`; a Core restart removes them. Supervisor boots `services`
apps before Core and `application` apps after `_start_core()`, which blocks
until Core reports `RUNNING`. Until then `http://supervisor/core/api` answers
502 (`supervisor/api/proxy.py`, `check_api_state`). See
[2026-10-unit-repairs-after-restart.md](2026-10-unit-repairs-after-restart.md)
for the source references.

**App side, before the fix:**
- `main_loop` set every first forecast tick at `now + forecast_every_minutes`.
- The scheduler advanced a tick's next time before running it, so a tick that
  failed (a restored entry has `combined=None`, so a failed history fetch
  returns `no_frame`) waited a full cadence.
- A retrain was scheduled `now + retrain_every_hours` before it was queued, and
  `_retrain_single`'s `except` never rescheduled. The boot retrain of an
  experiment with no usable model (no cache, a schema mismatch, a failed load,
  or the pinned-model case C) therefore failed on the 502, and the experiment
  published nothing until the next scheduled retrain. Experiments drain one at
  a time through the queue, so the earlier ones failed while later ones could
  succeed once Core was up: hence "some".

Two cases were ruled out:
- A missing weather or other future covariate after startup only degrades a
  forecast; `fetch_future` returns NaN.
- An app-only restart leaves the sensors in HA, stale but present.

## The fix

**Typed HA errors.** `api_call` raises `HAUnavailableError` once every
attempt failed to connect, timed out or got 5xx, and `HAResponseError` (with
`status`) on 4xx. Both subclass `RuntimeError` and keep their messages, so
existing handlers and log text are unchanged. `_ha_unavailable` walks the cause
chain, because the history fetch error reaches `_retrain_single` through
several frames.

**Retries.** `_schedule_retry` brings the next run forward by
`HA_RETRY_MINUTES[n]` and never past the normal cadence, so a lasting fault
settles back onto the schedule.
- Retrains retry only on `HAUnavailableError`. A training failure (too little
  data, a backend error) keeps the normal cadence rather than looping on a
  Pi.
- Forecasts retry on `no_frame`, on a failed sensor post, or on an
  `HAUnavailableError` raised out of the tick.

**HA-restart detection.** Every `HA_STATES_CHECK_SECONDS` (60) the main loop
calls `entity_exists(HEARTBEAT_ENTITY)`: one attempt, with no warning when HA
is down.
- The heartbeat is posted with every successful forecast, so once it has been
  seen, a 404 means HA dropped the app's states.
- The last payloads of current production experiments, and the stored
  `_last_<event>` payloads, are then re-posted once. An experiment whose
  forecast is running is skipped, so fresh values are not overwritten.

Design choices:
- **Re-post the last payload rather than run a forecast.** It is instant, needs
  no history fetch from an HA that is still loading, and adds no extra
  issuance to `forecast_log`. The values carry their original `issued_at`, and
  the next tick refreshes them.
- **A sentinel GET rather than a websocket subscription to
  `homeassistant_started`.** It is one cheap request a minute, needs no new
  client, and also catches a lost POST.
- **One re-post per loss.** If it fails, the forecast retries restore the
  sensors; repeating it every minute against a rejecting HA would only add
  errors to the log.
- **A 2-minute startup delay.** With `startup: application` Core is `RUNNING`,
  but stage-2 integrations, including the sources of future covariates, may
  still be loading. Without the delay the first forecast would run on
  NaN-filled covariates and be logged as a real issuance.

**`_last_retrain` restore.** After an experiment's first successful forecast
since start, or since a loss, `_restore_lifecycle_sensor` checks whether HA
has the sensor.
- It re-posts this process's last payload unchanged, or else the cached
  model's `trained_at`.
- Never "now", and never over a sensor HA still has: a changed timestamp would
  fire automations for a retrain that did not happen.

**`unknown` instead of omitted.** `unknown` is non-numeric, so HA's statistics
ignore it and no unit repair fires. A dashboard shows "Unknown" with a reason
instead of "Entity not available". `_interval` stays omitted for
non-cumulative sources: it would only duplicate `_forecast`.

## Limits that remain

- `_last_benchmark` of a production experiment is only posted by the lab
  benchmark path, so an HA restart still removes it until the next benchmark.
- Entity ids the app no longer produces (an experiment switched to lab,
  renamed, or given a new `conformal_coverage`) are never retired. They
  disappear for good at the next HA restart.
- **Not availability, found in the same review.** A covariate column absent at
  forecast time reaches the tree path as 0.0 with its pinned `_missing`
  indicator zero-filled (`preprocessing.py` `resolve_missingness`,
  `row_dict.get(c, 0.0)`), contrary to the masking invariant. It changes what
  a model is fed, so its fix is a separate minor release.
- After a restart, a production experiment with no saved benchmark result
  shows "Pending" on its card: `_restore_cached_models` runs before the web app
  exists, so the UI mirror never runs.
