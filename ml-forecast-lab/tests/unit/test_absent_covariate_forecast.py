"""v2.54.0: the forecast feeds models what they were trained on, never a silent 0.0.

Defects pinned (all live up to v2.52.7):
- A covariate the model was trained on but that returned no history at
  forecast time (an HA error made ``fetch_history`` return an empty series,
  so ``_fetch_and_preprocess`` added no column) reached the tree model as
  0.0, its pinned ``<col>_missing`` indicator was zero-filled ("measured"),
  and ``row_dict.get(c, 0.0)`` kept the nan_to_num backstop silent.
- The covariate history fetch threw its SQLite cache away on any HA error.
- Neither recursive row builder set ``is_holiday``: every holiday was fed 0.
- Covariates were picked by name shape, so one named ``y_*`` was fed 0.0 and
  the interactions of one named ``*_missing`` were fed 0.0.

Computed solar columns are excluded from the fill by the experiment's config,
not by name: a fetched covariate on ``sensor.sun_elevation`` is filled like
any other.

The neural path is deliberately unchanged: with no observation at all its
past window would be invented, and the channel-parity guard refuses it.
"""
from __future__ import annotations

import asyncio
import dataclasses
import importlib
import inspect
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_forecast_lab import main as M
from ml_forecast_lab.config import AppConfig, CovariateCfg, ExperimentCfg
from ml_forecast_lab.covariates import CovariateResolver
from ml_forecast_lab.db import HistoryDB
from ml_forecast_lab.features import is_holiday
from ml_forecast_lab.main import (
    MLForecastLabApp, _cache_entry_from_meta, _cache_meta,
    _computed_source_columns, _fitted_source_columns, _inference_indicator_map,
    _pinned_indicator_map, _reinstate_trained_covariates,
)
from ml_forecast_lab.models.registry import ModelRegistry
from ml_forecast_lab.preprocessing import resolve_missingness

NOW = datetime(2026, 10, 25, 6, 0, tzinfo=timezone.utc)
# Monday 26 October 2026 is a public holiday in Ireland.
HOLIDAY_NOW = datetime(2026, 10, 26, 3, 0, tzinfo=timezone.utc)


def _temp(i):
    return 10.0 + 5.0 * math.sin(2 * math.pi * i / 96 + 1.0)


def _pump(i):
    return 1.0 if (i // 7) % 3 == 0 else 0.0


def _rows(fn, now, skip=None, days=8, fmt="{:.4f}"):
    ts = pd.date_range(now - timedelta(days=days), now, freq="15min", tz="UTC")
    return [
        {"last_changed": t.isoformat(), "state": fmt.format(fn(i))}
        for i, t in enumerate(ts)
        if not (skip and skip[0] <= t.to_pydatetime() < skip[1])
    ]


class _StubHA:
    def __init__(self, rows_by_entity):
        self.rows = rows_by_entity
        self.fail: set = set()

    async def get_history(self, entity_id, start, end, include_attributes=False):
        if entity_id in self.fail:
            raise RuntimeError("HA 504 Gateway Timeout (simulated)")
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        s = s.tz_localize("UTC") if s.tzinfo is None else s
        e = e.tz_localize("UTC") if e.tzinfo is None else e
        return [r for r in self.rows.get(entity_id, [])
                if s <= pd.Timestamp(r["last_changed"]) <= e]

    async def get_state(self, entity_id, default=None, attribute=None):
        return default


def _train(tmp_path, monkeypatch, *, backend="lightgbm", covs=None, ents=None,
           country=None, gap=True, now=NOW, **exp_kw):
    """Retrain a real production model on synthetic history; return
    ``(app, cache, ha, exp)`` with the covariate SQLite cache warm.

    ``max_age`` is set far out so cache pruning, which runs against the
    wall clock, never reaches the fixed ``now``."""
    gap_span = (now - timedelta(days=4, hours=6), now - timedelta(days=4))
    if covs is None:
        covs = [CovariateCfg(entity="sensor.temp", role="lagged")]
        ents = {"sensor.temp": _rows(_temp, now, skip=gap_span if gap else None)}
    ents = dict(ents)
    ents["sensor.load"] = _rows(lambda i: 2.0 * _temp(i) + (i % 5) * 0.1, now)
    exp = ExperimentCfg(
        name="adv", target_entity="sensor.load", mode="production",
        interval_minutes=30, future_periods=24, days_history=10,
        country=country, covariates=covs, models_enabled=[backend],
        production_model=backend, max_age=36500, **exp_kw,
    )
    monkeypatch.setattr(
        MLForecastLabApp, "_cached_model_dir",
        staticmethod(lambda n: tmp_path / "models" / n),
    )
    orig = MLForecastLabApp._prepare_training_frame

    async def _at_now(self, exp_cfg, now=None, _o=orig, _n=now, **kw):
        return await _o(self, exp_cfg, now=_n, **kw)

    monkeypatch.setattr(MLForecastLabApp, "_prepare_training_frame", _at_now)

    app = MLForecastLabApp()
    app.history_db = HistoryDB(str(tmp_path / "h.db"))
    ha = _StubHA(ents)
    app.ha_interface = ha
    app._site_location = (53.35, -6.26)
    app.config = AppConfig(experiments=[exp], model_overrides={
        "lightgbm": {"n_estimators": 60}, "nlinear": {"epochs": 2}})
    app.covariate_resolver = CovariateResolver(
        ha, history_db=app.history_db,
        retention_provider=app._retention_days_for_table,
    )
    reg = ModelRegistry()
    reg.register("lightgbm", importlib.import_module(
        "ml_forecast_lab.models.lightgbm_backend").LightGBMModel)
    reg.register("nlinear", importlib.import_module(
        "ml_forecast_lab.models.nlinear_backend").NLinearModel)
    app.model_registry = reg

    async def _noop(*a, **k):
        return True

    app._publish_forecast_sensors = _noop
    app._publish_lifecycle_sensor = _noop
    np.random.seed(0)
    try:
        import torch
        torch.manual_seed(0)
    except ImportError:
        pass
    asyncio.run(app._retrain_and_cache(exp))
    return app, app._cached_models[exp.name], ha, exp


def _tick(app, cache, now=NOW):
    return asyncio.run(app._compute_cached_forecast(cache, now=now))


def _col(res, cache, name):
    return np.asarray(res.diag["X_rows"])[:, cache["feature_cols"].index(name)]


def _cold(app, ha):
    app.covariate_resolver = CovariateResolver(ha, history_db=None)


# --- tree: covariate absent at forecast time -----------------------------

class TestAbsentCovariateTree:
    def test_absent_covariate_gets_its_training_fill_and_flag(self, tmp_path, monkeypatch, caplog):
        app, cache, ha, _ = _train(tmp_path, monkeypatch)
        assert "temp_missing" in cache["missing_indicators"]
        fill = cache["covariate_fill"]["temp"]
        assert 5.0 <= fill <= 15.0
        ha.fail = {"sensor.temp"}
        _cold(app, ha)
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            res = _tick(app, cache)
        assert res.status == "ok"
        assert res.diag["reinstated_covariates"] == ["temp"]
        np.testing.assert_allclose(_col(res, cache, "temp"), fill, rtol=1e-6)
        assert np.all(_col(res, cache, "temp_missing") == 1.0)
        np.testing.assert_allclose(
            _col(res, cache, "temp_x_hour_sin"),
            fill * _col(res, cache, "hour_sin"), rtol=1e-5, atol=1e-6,
        )
        assert "missing from this cycle's data" in caplog.text
        assert not any(
            "imputed at 0.0" in r.getMessage() and "temp" in r.getMessage()
            for r in caplog.records
        )

    def test_warm_cache_serves_the_covariate(self, tmp_path, monkeypatch, caplog):
        app, cache, ha, _ = _train(tmp_path, monkeypatch)
        healthy = _tick(app, cache)
        ha.fail = {"sensor.temp"}
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.covariates"):
            res = _tick(app, cache)
        assert res.status == "ok"
        assert "reinstated_covariates" not in res.diag
        assert "Covariate history fetch failed for sensor.temp" in caplog.text
        np.testing.assert_array_equal(
            _col(res, cache, "temp"), _col(healthy, cache, "temp"),
        )

    @pytest.mark.parametrize("gap", [True, False])
    def test_old_cache_without_fill_is_logged_not_silent(self, tmp_path, monkeypatch, caplog, gap):
        app, cache, ha, _ = _train(tmp_path, monkeypatch, gap=gap)
        old = dict(cache)
        old.pop("covariate_fill")
        ha.fail = {"sensor.temp"}
        _cold(app, ha)
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            res = _tick(app, old)
        assert res.status == "ok"
        assert np.all(_col(res, old, "temp") == 0.0)
        zero = next(r.getMessage() for r in caplog.records
                    if "imputed at 0.0" in r.getMessage())
        held = next(r.getMessage() for r in caplog.records
                    if "missing from this cycle's data" in r.getMessage())
        if gap:
            assert np.all(_col(res, old, "temp_missing") == 1.0)
            assert "indicator raised for every row: ['temp']" in zero
            assert "flagged as imputed: ['temp']" in held
        else:
            assert "temp_missing" not in old["feature_cols"]
            assert "no indicator" in zero and "not flagged" in held

    def test_future_role_covariate_keeps_its_forecast(self, tmp_path, monkeypatch, caplog):
        gap_span = (NOW - timedelta(days=4, hours=6), NOW - timedelta(days=4))
        covs = [CovariateCfg(entity="sensor.temp", role="future")]
        ents = {"sensor.temp": _rows(_temp, NOW, skip=gap_span)}
        app, cache, ha, _ = _train(tmp_path, monkeypatch, covs=covs, ents=ents)
        ha.fail = {"sensor.temp"}
        _cold(app, ha)

        async def _forecast(cov, index):
            return pd.Series(np.linspace(6.0, 14.0, len(index)), index=index)

        app.covariate_resolver.fetch_future = _forecast
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            res = _tick(app, cache)
        assert res.status == "ok"
        assert res.diag["reinstated_covariates"] == ["temp"]
        np.testing.assert_allclose(
            _col(res, cache, "temp"), np.linspace(6.0, 14.0, len(res.ds_future)), rtol=1e-5,
        )
        msg = next(r.getMessage() for r in caplog.records
                   if "missing from this cycle's data" in r.getMessage())
        assert "use their forecast across the horizon" in msg and "held at" not in msg

    def test_gap_free_covariate_is_filled_but_not_flagged(self, tmp_path, monkeypatch, caplog):
        app, cache, ha, _ = _train(tmp_path, monkeypatch, gap=False)
        assert "temp_missing" not in cache["feature_cols"]
        ha.fail = {"sensor.temp"}
        _cold(app, ha)
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            res = _tick(app, cache)
        assert res.status == "ok"
        np.testing.assert_allclose(
            _col(res, cache, "temp"), cache["covariate_fill"]["temp"], rtol=1e-6,
        )
        msg = next(r.getMessage() for r in caplog.records
                   if "missing from this cycle's data" in r.getMessage())
        assert "not flagged" in msg and "flagged as imputed" not in msg

    def test_solar_provenance_comes_from_the_model_not_the_live_config(self, tmp_path, monkeypatch):
        """A restart attaches the live config; removing the sensor from
        mlfl.yaml must not turn the trained column into a pvlib one."""
        gap_span = (NOW - timedelta(days=4, hours=6), NOW - timedelta(days=4))
        covs = [CovariateCfg(entity="sensor.sun_elevation", role="lagged"),
                CovariateCfg(entity="sensor.temp", role="lagged")]
        ents = {"sensor.sun_elevation": _rows(_temp, NOW, skip=gap_span),
                "sensor.temp": _rows(lambda i: _temp(i + 9), NOW)}
        app, cache, _, exp = _train(tmp_path, monkeypatch, covs=covs, ents=ents)
        assert cache["computed_sources"] == ["clear_sky_ghi"]
        live = dataclasses.replace(exp, covariates=covs[1:])
        assert "sun_elevation" in _computed_source_columns(live)
        meta = _cache_meta(cache)
        restored = _cache_entry_from_meta(meta, cache["model"], live)
        res = _tick(app, restored)
        assert res.status == "ok"
        assert res.diag["reinstated_covariates"] == ["sun_elevation"]
        np.testing.assert_allclose(
            _col(res, restored, "sun_elevation"),
            cache["covariate_fill"]["sun_elevation"], rtol=1e-6,
        )
        assert np.all(_col(res, restored, "sun_elevation_missing") == 1.0)
        meta.pop("computed_sources")
        assert _cache_entry_from_meta(meta, cache["model"], live)["computed_sources"] is None

    def test_covariate_named_like_a_solar_column_is_filled(self, tmp_path, monkeypatch):
        gap_span = (NOW - timedelta(days=4, hours=6), NOW - timedelta(days=4))
        covs = [CovariateCfg(entity="sensor.sun_elevation", role="lagged")]
        ents = {"sensor.sun_elevation": _rows(_temp, NOW, skip=gap_span)}
        app, cache, ha, _ = _train(tmp_path, monkeypatch, covs=covs, ents=ents)
        fill = cache["covariate_fill"]["sun_elevation"]
        assert 5.0 <= fill <= 15.0
        # Held like any lagged covariate, never swapped for pvlib values.
        healthy = _col(_tick(app, cache), cache, "sun_elevation")
        assert np.unique(healthy).size == 1 and 5.0 <= healthy[0] <= 15.0
        ha.fail = {"sensor.sun_elevation"}
        _cold(app, ha)
        res = _tick(app, cache)
        assert res.status == "ok"
        assert res.diag["reinstated_covariates"] == ["sun_elevation"]
        np.testing.assert_allclose(_col(res, cache, "sun_elevation"), fill, rtol=1e-6)
        assert np.all(_col(res, cache, "sun_elevation_missing") == 1.0)

    def test_healthy_tick_is_unchanged(self, tmp_path, monkeypatch, caplog):
        app, cache, _, _ = _train(tmp_path, monkeypatch)
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            res = _tick(app, cache)
        monkeypatch.setattr(
            M, "_reinstate_trained_covariates", lambda df, fc, computed: (df, []),
        )
        same = _tick(app, cache)
        np.testing.assert_array_equal(res.diag["X_rows"], same.diag["X_rows"])
        np.testing.assert_array_equal(res.y_pred, same.y_pred)
        assert "nan_to_num backstop fired" not in caplog.text


# --- neural: deliberately unchanged --------------------------------------

class TestAbsentCovariateNeural:
    def test_cold_cache_still_refused(self, tmp_path, monkeypatch):
        app, cache, ha, _ = _train(tmp_path, monkeypatch, backend="nlinear")
        ha.fail = {"sensor.temp"}
        _cold(app, ha)
        assert _tick(app, cache).status == "channel_mismatch"

    def test_warm_cache_publishes(self, tmp_path, monkeypatch):
        app, cache, ha, _ = _train(tmp_path, monkeypatch, backend="nlinear")
        ha.fail = {"sensor.temp"}
        assert _tick(app, cache).status == "ok"


# --- persistence ---------------------------------------------------------

def test_fill_round_trips_through_cache_meta(tmp_path, monkeypatch):
    app, cache, _, exp = _train(tmp_path, monkeypatch)
    meta = _cache_meta(cache)
    assert meta["covariate_fill"] == cache["covariate_fill"]
    restored = _cache_entry_from_meta(meta, cache["model"], exp)
    assert restored["covariate_fill"] == cache["covariate_fill"]
    meta.pop("covariate_fill")
    assert _cache_entry_from_meta(meta, cache["model"], exp)["covariate_fill"] == {}


# --- holidays and covariate names ----------------------------------------

def test_holiday_feature_reaches_the_forecast(tmp_path, monkeypatch):
    app, cache, _, _ = _train(
        tmp_path, monkeypatch, country="IE", now=HOLIDAY_NOW,
    )
    assert "is_holiday" in cache["feature_cols"]
    res = _tick(app, cache, now=HOLIDAY_NOW)
    expected = [float(is_holiday(ts, "IE")) for ts in res.ds_future]
    assert any(expected), "the horizon must include the holiday"
    np.testing.assert_array_equal(_col(res, cache, "is_holiday"), expected)


def test_covariate_named_like_a_lag_or_flag_keeps_its_value(tmp_path, monkeypatch):
    covs = [CovariateCfg(entity="sensor.y_axis", role="lagged"),
            CovariateCfg(entity="sensor.pump_missing", role="lagged")]
    ents = {"sensor.y_axis": _rows(_temp, NOW),
            "sensor.pump_missing": _rows(lambda i: _temp(i + 13) * 0.5, NOW)}
    app, cache, _, _ = _train(tmp_path, monkeypatch, covs=covs, ents=ents)
    res = _tick(app, cache)
    assert res.status == "ok"
    for name in ("y_axis", "y_axis_x_hour_sin", "pump_missing_x_hour_sin"):
        assert np.any(_col(res, cache, name) != 0.0), name


# --- schema helpers ------------------------------------------------------

FC = [
    "y_lag_1", "hour_sin",
    "temp", "temp_x_hour_sin", "temp_x_hour_cos", "temp_missing",
    "pump", "pump_x_hour_sin", "pump_x_hour_cos",
    "pump_missing", "pump_missing_x_hour_sin", "pump_missing_x_hour_cos",
    "pump_missing_flag",
    "clear_sky_ghi", "clear_sky_ghi_x_hour_sin", "clear_sky_ghi_x_hour_cos",
]


class TestSchemaHelpers:
    def test_sources_come_from_the_interaction_pairs(self):
        assert _fitted_source_columns(FC) == ["temp", "pump", "pump_missing", "clear_sky_ghi"]

    def test_pinned_map_names_collision_flags(self):
        assert _pinned_indicator_map(["temp_missing", "pump_missing_flag"], FC) == {
            "temp_missing": "temp", "pump_missing_flag": "pump",
        }

    def test_pinned_map_matches_the_suffix_map_for_ordinary_names(self):
        fc = ["y_lag_1", "temp", "temp_x_hour_sin", "temp_x_hour_cos", "temp_missing"]
        assert _pinned_indicator_map(["temp_missing"], fc) == _inference_indicator_map(fc)

    def test_nothing_absent_returns_the_same_frame(self):
        df = pd.DataFrame({"y": [1.0], "temp": [2.0], "pump": [0.0],
                           "pump_missing": [1.0], "clear_sky_ghi": [0.0]})
        out, reinstated = _reinstate_trained_covariates(df, FC, frozenset())
        assert out is df and reinstated == []

    def test_absent_columns_reinstated_in_fitted_order_never_computed_ones(self):
        df = pd.DataFrame({"y": [1.0], "pump_missing": [1.0], "other": [3.0]})
        computed = _computed_source_columns(ExperimentCfg(name="e", target_entity="sensor.t"))
        out, reinstated = _reinstate_trained_covariates(df, FC, computed)
        assert reinstated == ["temp", "pump"]
        assert list(out.columns) == ["y", "temp", "pump", "pump_missing", "other"]
        assert out["temp"].isna().all()
        assert "clear_sky_ghi" not in out.columns

    def test_computed_columns_follow_the_config(self):
        both = {"clear_sky_ghi", "sun_elevation"}
        base = dict(name="e", target_entity="sensor.t")
        user = dict(base, covariates=[CovariateCfg(entity="sensor.sun_elevation")])
        assert _computed_source_columns(ExperimentCfg(**base)) == both
        assert _computed_source_columns(ExperimentCfg(**user)) == {"clear_sky_ghi"}
        assert _computed_source_columns(
            ExperimentCfg(**user, include_sun_elevation=True),
        ) == both

    def test_resolve_missingness_without_fill_is_unchanged(self):
        idx = pd.date_range("2026-10-01", periods=40, freq="30min")
        df = pd.DataFrame({
            "target": np.arange(40.0), "y_lag_1": np.arange(40.0) - 1,
            "temp": np.r_[np.full(20, np.nan), np.arange(20.0)],
        }, index=idx)
        a, ra = resolve_missingness(df, "target", 2)
        b, rb = resolve_missingness(df, "target", 2, empty_fill=None)
        pd.testing.assert_frame_equal(a, b)
        assert ra["indicator_cols"] == rb["indicator_cols"]


# --- source contracts ----------------------------------------------------

def test_row_builders_never_default_a_feature_to_zero():
    src = Path(M.__file__).read_text()
    assert src.count("row_dict.get(c, 0.0)") == 0
    assert src.count("row_dict.get(c, np.nan)") == 2


def test_reinstatement_is_gated_off_the_neural_path():
    src = inspect.getsource(MLForecastLabApp._compute_cached_forecast)
    gate = src.index("if not (is_neural and 'sequence_data' in seq_kwargs):")
    assert gate < src.index("_reinstate_trained_covariates(")
