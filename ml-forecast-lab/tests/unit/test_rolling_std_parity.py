"""v2.54.0: the recursive tree forecast computes ``y_rolling_std_*`` as training does.

Defect pinned (live up to v2.52.7): both recursive row builders
(``_run_production_inference`` and ``_compute_cached_forecast``) computed the
rolling standard deviation with ``np.std`` (``ddof=0``), while
``build_features`` uses pandas' rolling ``std`` (``ddof=1``). Every tree
forecast row was fed ``sqrt((w - 1) / w)`` of the rolling std the model was
trained on: about 9% low for a 6-row window. Benchmarks score tree models on
features built by ``build_features``, so rankings were never affected.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab import main as main_mod
from ml_forecast_lab.config import AppConfig, ExperimentCfg
from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.main import MLForecastLabApp

pytest.importorskip("lightgbm")

from ml_forecast_lab.models.lightgbm_backend import LightGBMModel  # noqa: E402

INTERVAL = 30
NOW = datetime(2026, 10, 25, 6, 0, tzinfo=timezone.utc)


def _run(coro):
    return asyncio.run(coro)


class _StubHA:
    def __init__(self, rows_by_entity):
        self.rows_by_entity = rows_by_entity

    async def get_history(self, entity_id, start, end, include_attributes=False):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s.tzinfo is None:
            s = s.tz_localize("UTC")
        if e.tzinfo is None:
            e = e.tz_localize("UTC")
        return [
            r for r in self.rows_by_entity.get(entity_id, [])
            if s <= pd.Timestamp(r["last_changed"]) <= e
        ]

    async def get_state(self, entity_id, default=None, attribute=None):
        return default


def _recorder_rows(entity_now, days, cadence_min=15,
                   value=lambda i: i % 37 + 1.0):
    ts = pd.date_range(
        entity_now - timedelta(days=days), entity_now,
        freq=f"{cadence_min}min", tz="UTC",
    )
    return [
        {"last_changed": t.isoformat(), "state": f"{value(i):.4f}"}
        for i, t in enumerate(ts)
    ]


class _Registry:
    def create(self, name):
        return LightGBMModel()


def _make_app(tmp_path, exp_cfg, monkeypatch):
    app = MLForecastLabApp()
    app.history_db = HistoryDB(str(tmp_path / "h.db"))
    app.ha_interface = _StubHA({
        exp_cfg.target_entity: _recorder_rows(NOW, days=11),
    })
    app.config = AppConfig(experiments=[exp_cfg])
    app.covariate_resolver = CovariateResolver(
        app.ha_interface,
        history_db=app.history_db,
        retention_provider=app._retention_days_for_table,
    )
    app.model_registry = _Registry()

    published: list[dict] = []

    async def _publish(**kw):
        published.append(kw)

    async def _noop(*a, **kw):
        return None

    monkeypatch.setattr(app, "_publish_forecast_sensors", _publish)
    monkeypatch.setattr(app, "_publish_lifecycle_sensor", _noop)
    monkeypatch.setattr(app, "_persist_cached_model", lambda *a, **kw: None)
    orig_fetch = MLForecastLabApp._fetch_and_preprocess

    async def _fetch_at_now(self, cfg, now=None, **kw):
        return await orig_fetch(self, cfg, now=NOW, **kw)

    monkeypatch.setattr(MLForecastLabApp, "_fetch_and_preprocess", _fetch_at_now)

    rows: dict[str, list] = {}
    real_guard = main_mod._nan_to_num_guarded

    def _guard(X, label, cols=None, *a, **kw):
        if "recursive forecast row" in label and cols is not None:
            rows.setdefault(label, []).append(
                dict(zip(cols, np.asarray(X, dtype=float).ravel())),
            )
        return real_guard(X, label, cols, *a, **kw)

    monkeypatch.setattr(main_mod, "_nan_to_num_guarded", _guard)
    return app, published, rows


@pytest.mark.parametrize("path, label", [
    ("production", "production recursive forecast row"),
    ("cached", "cached recursive forecast row"),
])
def test_rolling_std_matches_training(tmp_path, monkeypatch, path, label):
    exp = ExperimentCfg(
        name="load", target_entity="sensor.load", days_history=10,
        interval_minutes=INTERVAL, future_periods=24, max_age=36500,
        models_enabled=["lightgbm"], production_model="lightgbm",
    )
    app, published, rows = _make_app(tmp_path, exp, monkeypatch)
    if path == "production":
        _run(app._run_production_inference(exp))
    else:
        _run(app._retrain_and_cache(exp))
    assert published and rows.get(label), f"{path} path produced no rows"

    grid = _run(app._fetch_and_preprocess(exp))["y"]
    last_ts = pd.Timestamp(published[-1]["ds_future"][0]) - timedelta(minutes=INTERVAL)
    first = rows[label][0]
    windows = [int(c.rsplit("_", 1)[1]) for c in first if c.startswith("y_rolling_std_")]
    assert windows
    for w in windows:
        tail = grid.loc[:last_ts].iloc[-w:]
        assert tail.notna().all() and len(tail) == w
        expected = float(tail.std())  # pandas rolling std is ddof=1
        assert first[f"y_rolling_std_{w}"] == pytest.approx(expected, rel=1e-5)
        assert first[f"y_rolling_std_{w}"] != pytest.approx(
            float(np.std(tail.to_numpy())), rel=1e-5,
        ), "fed the population std"


def test_both_row_builders_use_sample_std():
    src = Path(main_mod.__file__).read_text()
    assert src.count("np.std(window, ddof=1)") == 2
    assert "float(np.std(window))" not in src
