# "The unit has changed" repairs after a restart (v2.52.5)

**Status:** Shipped in v2.52.5. The mechanism, the Home Assistant behaviour
it trips, the fix and the limits that remain, behind that release's
CHANGELOG entry.

The pinning tests:
- `ml-forecast-lab/tests/unit/test_unit_restart_fallback.py`: inconclusive
  lookups are never cached as `""`; the last resolved unit survives a
  restart; restore-file robustness; the `run()` wiring;
- `tests/unit/test_unit_inheritance.py`: the v2.43.1 / v2.44.0 behaviour
  (explicit units win, inheritance, caching, retrain re-resolve).

```bash
cd ml-forecast-lab/
python -m pytest tests/unit/test_unit_restart_fallback.py tests/unit/test_unit_inheritance.py -v
```

---

## TL;DR

| Defect (live v2.43.1 – v2.52.4) | Observed | Fix |
| --- | --- | --- |
| `_source_unit_last_good` lived only in memory | The v2.44.0 fallback never protected the first resolve after an add-on restart | Mirrored to `/data/ml_forecast_lab/source_units.json`, restored in `run()` before `main_loop` |
| `_resolve_units` cached `""` whenever the lookup came back empty, including a 404 or an unreachable HA | One miss pinned `""` on every forecast-family sensor until the next retrain, up to `retrain_every_hours` (24 h) | Only a *conclusive* lookup is cached. An inconclusive one with nothing known returns `""` for that publish only, and the next publish retries |

Symptom: Home Assistant Repairs shows "The unit of 'X Forecast'
(sensor.mlfl_X_forecast) changed to '' which can't be converted to the
previously stored unit, 'kW'", for `_forecast`, `_cumulative`,
`_upper_80`, `_lower_80` and `_forecast_accuracy`. Those sensors' long-term
statistics have a gap for the same period.

## Mechanism

1. With `units` blank, `_resolve_units` inherits the target's
   `unit_of_measurement`. It runs once per publish, inside
   `_publish_forecast_sensors`.
2. A fresh add-on process starts with an empty `_source_unit_cache` and an
   empty `_source_unit_last_good`.
3. A publish only happens after `_fetch_and_preprocess` has just fetched
   history. The recorder serves history for an entity id that has **no live
   state**: `history/__init__.py` rejects only *invalid* ids.
4. `GET /api/states/<target>` then returns 404 if the target has not been
   re-created. `HAInterface.get_state` swallows the `RuntimeError` and
   returns `default`. The same `default` comes back for a missing
   attribute, a 404 and an unreachable HA.
5. Pre-fix: no last-good unit, so `""` was cached. Nothing re-resolved it
   until `_retrain_and_cache` popped the cache.

Which targets go missing: Home Assistant writes an `unavailable` placeholder
for every enabled **entity-registry** entry at `EVENT_HOMEASSISTANT_START`.
That placeholder carries the registry's `unit_of_measurement`
(`helpers/entity_registry.py` `write_unavailable_state`), so a registry
target resolves its unit even while its integration loads.

States POSTed to `/api/states` have no registry entry and no `RestoreEntity`.
A Core restart removes them until their publisher posts again. Predbat's
`predbat.pv_power` is one: always `kW`, always via REST. So are the add-on's
own `sensor.mlfl_*`.

**Why a host reboot and not a Core-only restart:** a Core-only restart
leaves the add-on running, so its in-memory unit survives. A host reboot
restarts both the add-on (empty memory) and the target's publisher (target
missing for a while). An add-on update or crash-restart that coincides with
the publisher being down does the same.

### Not the cause: `startup: services`

`config.yaml` has `startup: services`, so on host boot Supervisor starts the
add-on *before* Core (`supervisor/core.py`: SYSTEM → SERVICES → `_start_core`
→ APPLICATION). Until Core reports `RUNNING`, the Supervisor proxy answers
502 to every request (`supervisor/api/proxy.py`, `check_api_state`).

That only delays the first publish. The history fetch fails, a restored
cache entry has `combined=None`, and `_compute_cached_forecast` returns
`no_frame` without publishing. `_resolve_units` is never reached while Core
is down. Moving to `startup: application` would not fix this. The target
can still be missing well after Core is `RUNNING`.

## Home Assistant side

Checked against core 2026.10.4.

- **What raises it:** `sensor/recorder.py` `_update_issues` compares each
  **live** numeric sensor state's unit with the statistics metadata unit. A
  `kW` metadata unit maps to `PowerConverter`. `""` is not in its
  `VALID_UNITS`, so the result is `units_changed`. A missing attribute
  (`None`) fails the same way; the dialog renders both as `''`.
- **When it runs:** hourly, in the 5-minute compile for the period starting
  at :50 (about :55:10 UTC). It also runs during the startup catch-up
  compile, and when Developer Tools → Statistics is opened.
- **Clearing:** the same pass deletes statistics issues it did not re-report.
  The issues are non-persistent, so a repair seen after a restart was raised
  after that restart.
- **Data gap:** while mismatched, `_normalize_states` drops the
  non-convertible states. Five-minute periods holding only `''` states get no
  short-term row, so the hourly long-term statistics have a hole. HA logs
  "The unit of sensor.mlfl_… () cannot be converted …" once per entity per
  Core process; `()` is a literal `""`, `(None)` a missing key.
- **"Fix issue" options are wrong for a transient:**
  - **Update the unit…** writes `''` as the metadata unit with no
    conversion. When `kW` returns, the repair fires in reverse.
  - **Delete all old statistic data** cascades away every statistic for the
    entity. The next compile adopts whatever unit the sensor carries at that
    moment.
  - Users should choose **Close**.

## The fix

`_resolve_units` now classifies each lookup before deciding what to publish
and what to cache:

| Lookup result | Published | Cached until retrain | `last_good` |
| --- | --- | --- | --- |
| Unit present (including on an `unavailable` registry placeholder) | the unit | yes | set (persisted when changed) |
| Present, state not `unknown`/`unavailable`, no unit | `""` | `""` | **dropped** (persisted) |
| Inconclusive (404, HA unreachable, `get_state` raising, `unknown`/`unavailable` with no unit), `last_good` known | `last_good` | `last_good` | unchanged |
| Inconclusive, nothing known | `""` | **no**, retried next publish | unchanged |

When the unit lookup returns nothing, a second `get_state(entity)` reads the
state, because only the state separates "genuinely unitless" from "not there
yet". It costs one extra call per miss and none in steady state. Replay is
unaffected: the publish path is outside both replay stages, and the call's
replay key (`attribute=None`, `default=None`) is deterministic anyway.

Design choices:
- **A conclusive unitless read beats the remembered unit.** The pre-fix code
  kept `last_good` over any empty read, because it could not tell
  unavailability from unitlessness. With persistence, that would make a
  deliberately removed unit permanent. Now an inconclusive read never
  changes the unit, and a conclusive one always does.
- **Storage:** `/data/ml_forecast_lab/source_units.json` holds
  `{target_entity: unit}`, written atomically (tmp + replace) and only when
  `last_good` changes.
  - It is keyed by entity, like the in-memory dict. A schema-mismatched
    model cache, which `_restore_cached_models` discards, does not take the
    unit with it. A changed `target_entity` cannot inherit another sensor's
    unit.
  - `_source_units_path` stays `None` unless `run()` sets it. Tests and
    replay shadow apps therefore never touch `/data`.
- **No model-cache schema bump and patch-level:** what models are fed is
  unchanged.

## Diagnosing from the add-on log

Pre-v2.52.5, the sequence was:
1. `Entity not found: <target>` with no `attempt N/3` retry lines before it,
   meaning a 404 rather than a transport error;
2. no `Auto-inherited unit` line;
3. `Publishing forecast for <exp>: … next=<n>` with no unit after the
   number.

From v2.52.5 an unresolved target logs `Could not resolve a unit from
<target> (absent or unavailable in HA); publishing <exp> without one and
retrying next cycle` on each publish until it resolves.

## Limits that remain

- **First-ever publish with no saved unit:** a fresh install, or the first
  restart after updating to v2.52.5 before any resolve has been saved. If the
  target is missing at that publish, the sensors carry `""` for one cycle.
  The repair only fires if an hourly check lands inside that cycle, and it
  clears at the next check.
- **A unit change that coincides with an inconclusive lookup** is adopted at
  the next retrain, as before, because the saved unit is cached until then.
  If the first lookup after a restart is conclusive, the new unit is adopted
  at once.
- **Related, not fixed here:** on a host reboot, an experiment with no
  restored model retrains immediately, before Core is `RUNNING`.
  - The fetch fails on the 502. `main_loop` has already set the next retrain
    `retrain_every_hours` ahead, and the retrain consumer's `except` does not
    reschedule.
  - Such an experiment publishes no forecasts for up to a day after a boot.
    A restored but stale model keeps forecasting in the meantime.
