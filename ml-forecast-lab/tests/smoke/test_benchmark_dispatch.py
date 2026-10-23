"""POST /experiment/{name}/run-benchmark and /api/benchmarks/run-all must
actually run the benchmark.

``benchmark_callback`` is an ``async`` function. Both endpoints used to call
it without awaiting or scheduling it, so the coroutine was created and
dropped ("coroutine ... was never awaited"): the response said "queued",
nothing ran, and because ``end_benchmark()`` lives inside the callback the
experiment stayed flagged as running — every later run answered 409 until
the add-on restarted.
"""

import time

import pytest


@pytest.fixture
def recorded_runs(app, seeded_experiment):
    """Install a stand-in callback that records the experiments it ran."""
    runs = []
    appstate = app.state.appstate

    async def fake_benchmark(name):
        try:
            runs.append(name)
        finally:
            appstate.end_benchmark(name)

    appstate.benchmark_callback = fake_benchmark
    return runs


def _wait_for(runs, expected, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and runs != expected:
        time.sleep(0.02)
    return runs


def test_run_benchmark_runs_the_callback(client, seeded_experiment, recorded_runs):
    resp = client.post(f"/experiment/{seeded_experiment}/run-benchmark")
    assert resp.status_code == 202

    assert _wait_for(recorded_runs, [seeded_experiment]) == [seeded_experiment]


def test_run_benchmark_does_not_jam_the_experiment(
    client, seeded_experiment, recorded_runs,
):
    """A finished run must clear the running flag, so a second run is accepted."""
    client.post(f"/experiment/{seeded_experiment}/run-benchmark")
    _wait_for(recorded_runs, [seeded_experiment])

    resp = client.post(f"/experiment/{seeded_experiment}/run-benchmark")
    assert resp.status_code == 202, resp.json()


def test_run_all_runs_the_callback(client, seeded_experiment, recorded_runs):
    resp = client.post("/api/benchmarks/run-all")
    assert resp.status_code == 202
    assert resp.json()["queued"] == [seeded_experiment]

    assert _wait_for(recorded_runs, [seeded_experiment]) == [seeded_experiment]
