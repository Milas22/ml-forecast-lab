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
| **E.** `sensor.mlfl_last_run` was only posted by a retrain; `_last_benchmark` only by the lab-mode cycle, never by a Training-tab benchmark | The heartbeat missing for up to `retrain_every_hours` after any HA restart; `_last_benchmark` of a production experiment never came back | The restart check posts the heartbeat, stamped with the last successful forecast or retrain; the Training-tab benchmark posts `_last_benchmark`. The `_last_<event>` sensors are deliberately not re-created (see below) |
| **F.** Nothing retired an entity id the app stopped producing (experiment renamed, switched to lab or removed; `publish_name`/`publish_prefix` or `conformal_coverage` changed) | The old sensors kept their last values until an HA restart dropped them | `_retire_orphaned_sensors` deletes recorded ids no current production experiment produces |
| **G.** `_restore_cached_models` mirrored restored models into the dashboard status before the web app existed, so the mirror never ran | A production experiment with a restored model but no saved benchmark result showed "Pending" and no model | `_sync_restored_models_to_ui` runs after `start_web_server` and fills only what the saved benchmark left unset |

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
- Only this check posts the heartbeat. Successful forecasts and retrains call
  `_note_success()`, and the check posts that time when it changes.
- An earlier draft posted the heartbeat from every successful forecast. A
  review showed that any experiment's tick, or its 2-minute retry, landing
  between HA's return and the next check re-created the sentinel. The check
  then saw it present and never re-posted the other experiments' sensors.
- Once the heartbeat has been seen, a 404 means HA dropped the app's states.
  The last payloads of current production experiments are re-posted once,
  limited to the ids each still produces (`_expected_entities`). An
  experiment whose forecast is running is skipped, so fresh values are not
  overwritten.
- The heartbeat is then re-posted with the last success time, not "now". A
  re-post is not a forecast, and a watchdog on the heartbeat must not read
  it as one. `_ha_states_seen` is set again as soon as that post succeeds,
  so a second HA restart before the next check is still caught.

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

**`_last_retrain` / `_last_benchmark` are not re-created.** An earlier draft
re-posted them after a restart, from this process's last payload or the
cached `trained_at` / saved benchmark timestamp. A review showed why that is
wrong.
- The entity no longer exists in HA, so the POST creates it: `state_changed`
  with `old_state=None`.
- HA's state trigger matches that whenever no `from` is set, and also on `to:`
  or an `attribute:` filter. A "retrain finished" automation would therefore
  fire after every HA restart and reboot, whatever the timestamp.
- A sensor missing until the next real retrain or benchmark is the lesser
  harm. Users with automations on them need no change.

**Retiring orphaned sensors.** `_record_published` maps every id the app
posts (forecast family and lifecycle) to its experiment and mirrors it to
`/data/ml_forecast_lab/published_entities.json`. Each minute's
`_check_ha_states` (when HA answers) calls `_retire_orphaned_sensors`.
- The expected set is computed from configuration, not from the last
  payloads, so an id a renamed experiment still produces under the same
  `publish_name` is never touched, even before that experiment's first
  forecast.
  - It includes the lifecycle ids (`_lifecycle_entities`) of every configured
    experiment, whatever its mode: lab experiments post `_last_benchmark` too.
  - For production experiments it includes the forecast family
    (`_forecast_entities`: bands at `conformal_coverage`, `_interval` for
    cumulative sources) under both the current config and the `exp_cfg`
    cached at the last retrain (`_live_forecast_entities`).
- The cached `exp_cfg` matters because forecasts publish with it. A review
  caught an earlier draft that used the config alone. After a
  `publish_name`, `publish_prefix` or `conformal_coverage` edit it deleted the
  still-posted old ids every tick until the next retrain, and the next tick
  re-created them.
- An id is deleted (`DELETE /api/states/<id>`) only after it has been
  unexpected in two consecutive checks. A config reload caught mid-write
  therefore can't remove live sensors.
- Nothing runs while the config is the stub (`_config_is_stub`), which is used
  both when no file is found and when the first load fails to parse, or while
  the config lists no experiments.
- HA Core has supported `DELETE /api/states/<id>` for years (admin only; the
  Supervisor token is admin). The Supervisor proxy forwards DELETE on
  `/core/api/...` only from Supervisor 2026.02.0 (commit 7ad9a911); older
  versions answer 405, which disables the cleanup with one warning.
- Only ids this app recorded are candidates, so ids posted before v2.52.7 are
  left alone; HA drops those at its next restart anyway.
- A 404 still clears the record. An unreachable HA keeps the id for the next
  check. Any other refusal disables the cleanup with one warning.
- The re-post after an HA restart is filtered to the expected set, so it
  can't resurrect a retired id.

**`unknown` instead of omitted.** `unknown` is non-numeric, so HA's statistics
ignore it and no unit repair fires. A dashboard shows "Unknown" with a reason
instead of "Entity not available". `_interval` stays omitted for
non-cumulative sources: it would only duplicate `_forecast`.

## Limits that remain

- By design, `_last_retrain` and `_last_benchmark` stay missing after an HA
  restart until the next retrain or benchmark (see above).
- **Not availability, found in the same review:** a covariate column absent at
  forecast time reached the tree path as 0.0, with its pinned `_missing`
  indicator zero-filled. This was fixed in v2.54.0; see
  [2026-10-absent-covariate-at-forecast.md](2026-10-absent-covariate-at-forecast.md).
