# Benchmark buttons never ran (v2.54.1)

**Status:** Fixed in v2.54.1 in `ml_forecast_lab/web/app.py`.

## What happened

- `POST /experiment/{name}/run-benchmark` and `POST /api/benchmarks/run-all`
  called `appstate.benchmark_callback(name)` without `await`.
- The callback (`_benchmark_trigger` in `main.py`) is `async`, so the call
  only created a coroutine that was never scheduled. The log shows
  `RuntimeWarning: coroutine '..._benchmark_trigger' was never awaited`.
- The endpoints still answered "Queued".
- `start_benchmark()` sets the running flag before dispatch, and
  `end_benchmark()` sits in the callback's `finally`, which never ran. The
  experiment stayed flagged as running and answered 409 to further runs until
  restart.
- `POST /experiment/{name}/run-pipeline` was unaffected: it goes through
  `training_queue` and awaits the callback.

## Fix and checks

- New helper `_dispatch_benchmark` schedules the callback with
  `asyncio.create_task` and tracks the task in `_pipeline_tasks`, as
  run-pipeline does, so stop-training can cancel it and the task is not
  garbage-collected mid-run. The done-callback removes the entry.
- `tests/smoke/test_benchmark_dispatch.py` (3 tests) checks that both
  endpoints run the callback and leave no un-awaited coroutine. They fail
  without the fix.
- Smoke suite: 106 passed with `-W error::RuntimeWarning`.
