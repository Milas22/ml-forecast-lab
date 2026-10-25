"""v2.54.0: a covariate's ``transform`` reaches forecasts on its training scale.

Defects pinned (live up to v2.52.7):
- ``_fetch_and_preprocess`` applied ``cov_cfg.transform`` to covariate
  history, but all four future-value paths (``_run_production_inference`` and
  ``_compute_cached_forecast``, tree and neural) applied ``scale`` only, so a
  future-role covariate with a transform was fed raw forecast values at every
  horizon step to a model trained on transformed ones.
- The ``log`` / ``box_cox`` shift is chosen from the window's minimum, and
  every forecast tick re-fetches its own window, so for a series that goes
  negative (or a positive one with a rare exact zero) the same reading
  reached the model at a different offset from tick to tick until the next
  retrain.

The fix records each covariate's transform and shift at retrain
(``covariate_transforms``, every covariate including untransformed ones, so
a stored map is authoritative) and applies it to history and future values.
Benchmarks, tuning and covariate analysis use one window and never fetch
forecasts, so they are unchanged.
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import json
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
from ml_forecast_lab.main import (
    MLForecastLabApp, _cache_entry_from_meta, _cache_meta,
    _covariate_transform_record, _future_covariate_on_model_scale,
)
from ml_forecast_lab.models.registry import ModelRegistry
from ml_forecast_lab.preprocessing import apply_transform

pytest.importorskip("lightgbm")

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
T0 = NOW - timedelta(days=20)
# 9.5-9.75 days before NOW: inside the 10-day training window, out of the
# window of a tick 12 hours later.
SNAP = (NOW - timedelta(days=9, hours=18), NOW - timedelta(days=9, hours=12))


def _h(t):
    return (t - T0).total_seconds() / 3600.0


def outdoor(t):
    v = 2.0 + 4.0 * math.sin(2 * math.pi * (_h(t) - 9) / 24)
    return v - 4.0 if SNAP[0] <= t < SNAP[1] else v


def ground(t):
    v = 1.0 + 0.8 * math.sin(2 * math.pi * _h(t) / 24 + 0.5)
    return v - 3.0 if SNAP[0] <= t < SNAP[1] else v


def load(t):
    return (3.0 - 0.25 * outdoor(t) + 0.6 * ground(t)
            + 0.4 * math.sin(2 * math.pi * _h(t) / 24))


FN = {"sensor.outdoor": outdoor, "sensor.ground": ground}


def _rows(fn):
    ts = pd.date_range(T0, NOW + timedelta(days=3), freq="15min", tz="UTC")
    return [{"last_changed": t.isoformat(), "state": f"{fn(t.to_pydatetime()):.4f}"}
            for t in ts]


class _StubHA:
    def __init__(self):
        self.rows = {"sensor.load": _rows(load), **{e: _rows(f) for e, f in FN.items()}}
        self.clock = NOW

    async def get_history(self, entity_id, start, end, include_attributes=False):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        s = s.tz_localize("UTC") if s.tzinfo is None else s
        e = e.tz_localize("UTC") if e.tzinfo is None else e
        return [r for r in self.rows.get(entity_id, [])
                if s <= pd.Timestamp(r["last_changed"]) <= e]

    async def get_state(self, entity_id, default=None, attribute=None):
        if entity_id in FN and attribute == "forecast":
            ts = pd.date_range(self.clock, self.clock + timedelta(hours=30),
                               freq="30min", tz="UTC")
            return [{"datetime": t.isoformat(),
                     "value": round(FN[entity_id](t.to_pydatetime()), 4)} for t in ts]
        return default


def _train(tmp_path, monkeypatch, covs, backend="lightgbm", ha=None):
    """Retrain a real production model at ``NOW``; return ``(app, cache, ha, exp)``."""
    exp = ExperimentCfg(
        name="tx", target_entity="sensor.load", mode="production",
        interval_minutes=30, future_periods=24, days_history=10,
        covariates=covs, models_enabled=[backend], production_model=backend,
        max_age=36500,
    )
    monkeypatch.setattr(
        MLForecastLabApp, "_cached_model_dir",
        staticmethod(lambda n: tmp_path / "models" / n),
    )
    orig = MLForecastLabApp._prepare_training_frame

    async def _at_now(self, exp_cfg, now=None, _o=orig, **kw):
        return await _o(self, exp_cfg, now=NOW, **kw)

    monkeypatch.setattr(MLForecastLabApp, "_prepare_training_frame", _at_now)

    app = MLForecastLabApp()
    app.history_db = HistoryDB(str(tmp_path / "h.db"))
    ha = ha or _StubHA()
    app.ha_interface = ha
    app.config = AppConfig(experiments=[exp], model_overrides={
        "lightgbm": {"n_estimators": 60}, "nlinear": {"epochs": 2}})
    app.covariate_resolver = CovariateResolver(ha, history_db=None)
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
    app._forecast_with_cached = _noop
    np.random.seed(0)
    try:
        import torch
        torch.manual_seed(0)
    except ImportError:
        pass
    asyncio.run(app._retrain_and_cache(exp))
    return app, app._cached_models[exp.name], ha, exp


def _tick(app, cache, ha, hours=0):
    t = NOW + timedelta(hours=hours)
    ha.clock = t
    return asyncio.run(app._compute_cached_forecast(cache, now=t))


def _col(res, cache, name):
    return np.asarray(res.diag["X_rows"])[:, cache["feature_cols"].index(name)]


def _utc(t):
    t = pd.Timestamp(t)
    return (t.tz_localize("UTC") if t.tzinfo is None else t).to_pydatetime()


def _forecast(ds):
    """The stub's forecast at the horizon steps (``ds_future`` is naive UTC)."""
    return np.array([round(outdoor(_utc(t)), 4) for t in ds])


def _log_pinned(x, rec):
    return np.log(np.maximum(np.asarray(x) + rec["shift"],
                             min(1.0, rec["min"] + rec["shift"])))


# --- apply_transform with a pinned shift ---------------------------------

SERIES = {
    "signed": pd.Series([-4.7, -1.0, 0.0, 2.5, 6.0]),
    "zeros": pd.Series([0.0, 0.0, 1.5, 3.0]),
    "positive_below_1": pd.Series([0.35, 0.5, 0.9]),
    "positive": pd.Series([50.1, 61.0, 88.0]),
}


class TestPinnedTransform:
    @pytest.mark.parametrize("kind", SERIES)
    @pytest.mark.parametrize("transform", ["log", "box_cox"])
    def test_pinned_equals_unpinned_on_the_training_series(self, kind, transform):
        s = SERIES[kind]
        fresh = apply_transform(s, transform)
        rec = json.loads(json.dumps(_covariate_transform_record(fresh, transform)))
        pinned = apply_transform(s, transform, shift=rec["shift"], lower=rec["min"])
        np.testing.assert_array_equal(pinned.to_numpy(), fresh.to_numpy())

    def test_later_window_keeps_the_trained_shift(self):
        rec = _covariate_transform_record(apply_transform(SERIES["signed"], "log"), "log")
        later = pd.Series([-1.0, 0.5, 3.0])
        assert apply_transform(later, "log").attrs["transform_shift"] != rec["shift"]
        out = apply_transform(later, "log", shift=rec["shift"], lower=rec["min"])
        np.testing.assert_allclose(out, np.log(later + rec["shift"]))

    def test_values_below_the_trained_range_stay_finite(self):
        rec = _covariate_transform_record(apply_transform(SERIES["positive_below_1"], "log"), "log")
        assert rec["shift"] == 0.0
        out = apply_transform(pd.Series([0.0, -2.0, np.nan, 0.6]), "log",
                              shift=rec["shift"], lower=rec["min"])
        assert np.isfinite(out.iloc[[0, 1, 3]]).all() and np.isnan(out.iloc[2])
        assert out.iloc[0] == out.iloc[1] == pytest.approx(math.log(0.35))


class TestTransformRecord:
    def test_untransformed_covariates_are_recorded(self):
        assert _covariate_transform_record(SERIES["signed"], None) == {"transform": None}

    def test_sqrt_needs_no_shift(self):
        s = apply_transform(SERIES["zeros"], "sqrt")
        assert _covariate_transform_record(s, "sqrt") == {"transform": "sqrt"}

    def test_all_nan_window_is_never_pinned(self):
        s = apply_transform(pd.Series([np.nan, np.nan]), "log")
        rec = _covariate_transform_record(s, "log")
        assert rec == {"transform": "log"}
        json.dumps(rec, allow_nan=False)

    def test_box_cox_is_recorded_under_its_applied_name(self):
        s = apply_transform(SERIES["zeros"], "box_cox")
        assert _covariate_transform_record(s, "box_cox")["transform"] == "shifted_log"


class TestFutureOnModelScale:
    def _cfg(self, transform=None, scale=None):
        return CovariateCfg(entity="sensor.outdoor", role="future",
                            transform=transform, scale=scale)

    def test_scale_then_pinned_transform(self):
        rec = {"transform": "log", "shift": 5.0, "min": -4.0}
        out = _future_covariate_on_model_scale(
            pd.Series([1.0, 2.0]), self._cfg("log", 2.0), "outdoor", {"outdoor": rec})
        np.testing.assert_allclose(out, np.log(np.array([2.0, 4.0]) + 5.0))

    def test_recorded_untransformed_wins_over_the_yaml(self):
        out = _future_covariate_on_model_scale(
            pd.Series([-1.0, 2.0]), self._cfg("log"), "outdoor",
            {"outdoor": {"transform": None}})
        np.testing.assert_array_equal(out, [-1.0, 2.0])

    def test_unrecorded_sqrt_is_applied(self):
        out = _future_covariate_on_model_scale(
            pd.Series([4.0, -1.0]), self._cfg("sqrt"), "outdoor", {})
        np.testing.assert_array_equal(out, [2.0, 0.0])

    def test_unrecorded_log_is_never_fed_raw(self):
        assert _future_covariate_on_model_scale(
            pd.Series([1.0]), self._cfg("log"), "outdoor", {}) is None

    def test_no_transform_is_plain_scale(self):
        out = _future_covariate_on_model_scale(
            pd.Series([1.0, 2.0]), self._cfg(scale=3.0), "outdoor", {})
        np.testing.assert_array_equal(out, [3.0, 6.0])


# --- tree pipeline -------------------------------------------------------

FUTURE_LOG = [CovariateCfg(entity="sensor.outdoor", role="future", transform="log")]


class TestTreeForecast:
    def test_future_values_get_the_trained_transform(self, tmp_path, monkeypatch):
        app, cache, ha, _ = _train(tmp_path, monkeypatch, FUTURE_LOG)
        rec = cache["covariate_transforms"]["outdoor"]
        assert rec["transform"] == "log" and rec["min"] < 0
        res = _tick(app, cache, ha)
        assert res.status == "ok"
        fed = _col(res, cache, "outdoor")
        np.testing.assert_allclose(fed, _log_pinned(_forecast(res.ds_future), rec), rtol=1e-5)
        # Raw forecast values span about [-2, 6]; on the trained scale they sit
        # inside the trained column's range (to within the resampling of
        # history into 30-minute means).
        trained = cache["combined"]["outdoor"]
        assert trained.min() - 0.1 <= fed.min() and fed.max() <= trained.max() + 0.1

    def test_history_keeps_the_trained_shift(self, tmp_path, monkeypatch):
        covs = [CovariateCfg(entity="sensor.ground", role="lagged", transform="log")]
        app, cache, ha, exp = _train(tmp_path, monkeypatch, covs)
        rec = cache["covariate_transforms"]["ground"]
        t = NOW + timedelta(hours=12)
        derived = {}
        asyncio.run(app._fetch_and_preprocess(exp, now=t, covariate_transforms=derived))
        assert derived["ground"]["shift"] != rec["shift"], "the window's shift must drift"
        res = _tick(app, cache, ha, hours=12)
        assert res.status == "ok"
        raw = copy.deepcopy(exp)
        raw.covariates[0].transform = None
        last_ts = pd.Timestamp(res.ds_future[0]) - timedelta(minutes=30)
        raw_last = asyncio.run(app._fetch_and_preprocess(raw, now=t))["ground"].loc[last_ts]
        assert np.isfinite(raw_last)
        np.testing.assert_allclose(
            _col(res, cache, "ground"), _log_pinned(raw_last, rec), rtol=1e-5,
        )

    def test_old_cache_uses_the_ticks_own_transform(self, tmp_path, monkeypatch):
        app, cache, ha, exp = _train(tmp_path, monkeypatch, FUTURE_LOG)
        meta = _cache_meta(cache)
        meta.pop("covariate_transforms")
        old = _cache_entry_from_meta(meta, cache["model"], exp)
        assert old["covariate_transforms"] == {}
        t = NOW + timedelta(hours=12)
        tick_rec = {}
        asyncio.run(app._fetch_and_preprocess(exp, now=t, covariate_transforms=tick_rec))
        res = _tick(app, old, ha, hours=12)
        assert res.status == "ok"
        np.testing.assert_allclose(
            _col(res, old, "outdoor"),
            _log_pinned(_forecast(res.ds_future), tick_rec["outdoor"]), rtol=1e-5,
        )

    def test_transform_added_to_the_yaml_waits_for_a_retrain(self, tmp_path, monkeypatch, caplog):
        covs = [CovariateCfg(entity="sensor.outdoor", role="future")]
        app, cache, ha, exp = _train(tmp_path, monkeypatch, covs)
        assert cache["covariate_transforms"] == {"outdoor": {"transform": None}}
        edited = copy.deepcopy(exp)
        edited.covariates[0].transform = "log"
        restored = _cache_entry_from_meta(_cache_meta(cache), cache["model"], edited)
        with caplog.at_level(logging.INFO, logger="ml_forecast_lab.main"):
            res = _tick(app, restored, ha)
        assert res.status == "ok"
        np.testing.assert_allclose(_col(res, restored, "outdoor"), _forecast(res.ds_future), rtol=1e-5)
        assert "takes effect at the next retrain" in caplog.text

    def test_untransformed_experiment_is_unchanged_by_the_map(self, tmp_path, monkeypatch):
        covs = [CovariateCfg(entity="sensor.outdoor", role="future")]
        app, cache, ha, exp = _train(tmp_path, monkeypatch, covs)
        meta = _cache_meta(cache)
        with_map = _cache_entry_from_meta(meta, cache["model"], exp)
        meta.pop("covariate_transforms")
        without = _cache_entry_from_meta(meta, cache["model"], exp)
        a, b = _tick(app, with_map, ha), _tick(app, without, ha)
        np.testing.assert_array_equal(a.diag["X_rows"], b.diag["X_rows"])
        np.testing.assert_array_equal(a.y_pred, b.y_pred)

    def test_failed_fetch_falls_back_to_the_trained_transform(self, tmp_path, monkeypatch):
        app, cache, ha, _ = _train(tmp_path, monkeypatch, FUTURE_LOG)
        rec = cache["covariate_transforms"]["outdoor"]

        async def _down(exp_cfg, now=None, covariate_transforms=None):
            if covariate_transforms is not None:
                covariate_transforms["outdoor"] = {"transform": "log", "shift": 100.0, "min": 0.0}
            raise RuntimeError("HA 504 Gateway Timeout (simulated)")

        app._fetch_and_preprocess = _down
        res = _tick(app, cache, ha)
        assert res.status == "ok" and not res.used_fresh_frame
        np.testing.assert_allclose(
            _col(res, cache, "outdoor"), _log_pinned(_forecast(res.ds_future), rec), rtol=1e-5,
        )

    def test_no_numeric_history_drops_the_forecast_and_says_so(self, tmp_path, monkeypatch, caplog):
        """A categorical state leaves the history all-NaN, so the log offset
        cannot be fixed: the forecast is not used, and both the 0%-coverage
        warning and the forecast path say so."""
        cond = "sensor.outdoor_condition"

        class _Categorical(_StubHA):
            def __init__(self):
                super().__init__()
                self.rows[cond] = [dict(r, state="partlycloudy")
                                   for r in self.rows["sensor.outdoor"]]

            async def get_state(self, entity_id, default=None, attribute=None):
                if entity_id == cond:
                    entity_id = "sensor.outdoor"
                return await super().get_state(entity_id, default, attribute)

        covs = FUTURE_LOG + [CovariateCfg(entity=cond, role="future", transform="log")]
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            app, cache, ha, _ = _train(tmp_path, monkeypatch, covs, ha=_Categorical())
        assert cache["covariate_transforms"]["outdoor_condition"] == {"transform": "log"}
        json.dumps(_cache_meta(cache)["covariate_transforms"], allow_nan=False)
        assert "Its forecast values are not used either" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            res = _tick(app, cache, ha)
        assert res.status == "ok"
        assert res.diag["future_covariates"] == ["outdoor"]
        assert "'outdoor_condition': forecast values not used" in caplog.text

    def test_record_round_trips_through_cache_meta(self, tmp_path, monkeypatch):
        app, cache, _, exp = _train(tmp_path, monkeypatch, FUTURE_LOG)
        meta = json.loads(json.dumps(_cache_meta(cache), allow_nan=False))
        restored = _cache_entry_from_meta(meta, cache["model"], exp)
        assert restored["covariate_transforms"] == cache["covariate_transforms"]


# --- neural pipeline -----------------------------------------------------

def test_neural_future_block_gets_the_trained_transform(tmp_path, monkeypatch):
    app, cache, ha, _ = _train(tmp_path, monkeypatch, FUTURE_LOG, backend="nlinear")
    rec = cache["covariate_transforms"]["outdoor"]
    res = _tick(app, cache, ha)
    assert res.status == "ok" and res.diag["path"] == "neural"
    ch = res.diag["channel_names"].index("outdoor")
    future_block = np.asarray(res.diag["window"])[0, -len(res.ds_future):, ch]
    np.testing.assert_allclose(
        future_block, _log_pinned(_forecast(res.ds_future), rec), rtol=1e-5,
    )


# --- source contracts ----------------------------------------------------

def test_no_future_path_applies_scale_alone():
    src = Path(M.__file__).read_text()
    assert "future_series = future_series * cov_cfg.scale" not in src
    assert src.count("_future_covariate_on_model_scale(") == 5  # def + 4 paths


def test_retrain_always_derives_its_own_transforms():
    src = inspect.getsource(MLForecastLabApp._retrain_and_cache)
    assert src.index("cov_transforms: dict = {}") < src.index(
        "self._prepare_training_frame(")
