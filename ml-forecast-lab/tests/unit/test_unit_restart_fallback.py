"""v2.52.5: forecast sensors no longer lose their unit after an add-on restart.

Defect: with ``units`` blank, the published unit is inherited from the
target sensor. The last-known unit lived only in memory, so after an add-on
restart (a host reboot included) the first publish depended entirely on a
live lookup. A REST-published target such as Predbat's ``predbat.pv_power``
is not in HA's entity registry, so an HA restart removes it until its
publisher's next cycle; the lookup 404'd, ``_resolve_units`` cached "" until
the next retrain (up to a day), and HA raised a "The unit has changed"
repair against the stored kW statistics and stopped compiling them.

These tests pin both halves of the fix: an inconclusive lookup (absent,
``unavailable`` without a unit, or raising) is never cached as "", and the
last resolved unit is persisted so a fresh process can fall back to it.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging

import numpy as np
import pandas as pd

from ml_forecast_lab.config import ExperimentCfg
from ml_forecast_lab.main import MLForecastLabApp

ENTITY = "sensor.pv_power"


class _StubHA:
    """Captures set_state payloads; serves a mutable source unit and state.

    ``state=None`` is a target absent from HA (``HAInterface.get_state``
    swallows the 404 and returns ``default``).
    """

    def __init__(self, unit="kW", state="1.0"):
        self.unit = unit
        self.state = state
        self.raise_on_get = False
        self.captured: list = []

    async def set_state(self, entity_id, state, attributes=None):
        self.captured.append((entity_id, state, attributes or {}))
        return True

    async def get_state(self, entity_id, default=None, attribute=None):
        if self.raise_on_get:
            raise RuntimeError("simulated HA failure")
        if self.state is None:
            return default
        if attribute == "unit_of_measurement":
            return self.unit if self.unit is not None else default
        if attribute is None:
            return self.state
        return default


def _make_app(ha, units_path=None):
    app = MLForecastLabApp()
    app.ha_interface = ha
    app.history_db = None
    app._cached_models = {}
    app._source_units_path = units_path
    return app


def _exp():
    return ExperimentCfg(
        name="solar", target_entity=ENTITY, units="",
        interval_minutes=30, future_periods=3,
        publish_prefix="mlfl_", publish_name="solar", mode="production",
    )


def _publish(app):
    app.ha_interface.captured = []
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ds = pd.DatetimeIndex([
        pd.Timestamp("2026-10-25 12:00") + pd.Timedelta(minutes=30 * (i + 1))
        for i in range(3)
    ])
    asyncio.run(app._publish_forecast_sensors(
        exp_cfg=_exp(), y_pred=y, ds_future=ds,
        model_name="nlinear", last_trained_iso="2026-10-25T00:00:00Z",
    ))
    units = [a.get("unit_of_measurement") for _, _, a in app.ha_interface.captured]
    assert units, "expected at least one sensor published"
    return set(units)


def _retrain(app):
    # The invalidation _retrain_and_cache performs before its re-resolve.
    app._source_unit_cache.pop(ENTITY, None)


class TestInconclusiveLookupIsNotCached:
    def test_absent_target_publishes_empty_then_retries(self, caplog):
        ha = _StubHA(unit=None, state=None)
        app = _make_app(ha)
        with caplog.at_level(logging.INFO, logger="ml_forecast_lab.main"):
            assert _publish(app) == {""}
        assert ENTITY not in app._source_unit_cache
        assert "Could not resolve a unit from sensor.pv_power" in caplog.text

        # The publisher re-creates the target: the next publish (no
        # retrain in between) must pick its unit up.
        ha.unit, ha.state = "kW", "0.4"
        assert _publish(app) == {"kW"}
        assert app._source_unit_cache[ENTITY] == "kW"

    def test_unavailable_target_without_unit_is_inconclusive(self):
        ha = _StubHA(unit=None, state="unavailable")
        app = _make_app(ha)
        assert _publish(app) == {""}
        assert ENTITY not in app._source_unit_cache
        ha.unit, ha.state = "kW", "0.4"
        assert _publish(app) == {"kW"}

    def test_unavailable_target_with_unit_resolves(self):
        # HA's registry placeholder for an entity whose integration hasn't
        # loaded yet keeps the registry unit.
        app = _make_app(_StubHA(unit="kW", state="unavailable"))
        assert _publish(app) == {"kW"}

    def test_raising_lookup_is_not_cached(self):
        ha = _StubHA(unit="kW")
        ha.raise_on_get = True
        app = _make_app(ha)
        assert _publish(app) == {""}
        assert ENTITY not in app._source_unit_cache
        ha.raise_on_get = False
        assert _publish(app) == {"kW"}

    def test_present_unitless_target_is_cached_empty(self):
        app = _make_app(_StubHA(unit=None, state="12.5"))
        assert _publish(app) == {""}
        assert app._source_unit_cache[ENTITY] == ""


class TestLastGoodUnitSurvivesRestart:
    def test_resolved_unit_is_written(self, tmp_path):
        path = tmp_path / "source_units.json"
        app = _make_app(_StubHA(unit="kW"), path)
        assert _publish(app) == {"kW"}
        assert json.loads(path.read_text()) == {ENTITY: "kW"}

    def test_fresh_process_falls_back_while_target_absent(self, tmp_path):
        path = tmp_path / "source_units.json"
        assert _publish(_make_app(_StubHA(unit="kW"), path)) == {"kW"}

        # Restart: new process, target not yet re-created in HA.
        app = _make_app(_StubHA(unit=None, state=None), path)
        app._restore_source_units()
        assert _publish(app) == {"kW"}

    def test_fresh_process_without_saved_unit_does_not_pin_empty(self, tmp_path):
        path = tmp_path / "source_units.json"
        ha = _StubHA(unit=None, state=None)
        app = _make_app(ha, path)
        app._restore_source_units()
        assert _publish(app) == {""}
        assert not path.exists()
        ha.unit, ha.state = "kW", "0.4"
        assert _publish(app) == {"kW"}
        assert json.loads(path.read_text()) == {ENTITY: "kW"}

    def test_unit_change_is_adopted_and_written(self, tmp_path):
        path = tmp_path / "source_units.json"
        ha = _StubHA(unit="W")
        app = _make_app(ha, path)
        assert _publish(app) == {"W"}
        _retrain(app)
        ha.unit = "kW"
        assert _publish(app) == {"kW"}
        assert json.loads(path.read_text()) == {ENTITY: "kW"}

    def test_unchanged_unit_is_not_rewritten(self, tmp_path, monkeypatch):
        writes = []
        app = _make_app(_StubHA(unit="kW"), tmp_path / "source_units.json")
        orig = app._persist_source_units
        monkeypatch.setattr(
            app, "_persist_source_units",
            lambda: (writes.append(1), orig())[1],
        )
        _publish(app)
        _retrain(app)
        _publish(app)
        assert len(writes) == 1

    def test_source_that_loses_its_unit_is_forgotten(self, tmp_path):
        path = tmp_path / "source_units.json"
        ha = _StubHA(unit="kW")
        app = _make_app(ha, path)
        assert _publish(app) == {"kW"}

        # Retrain re-resolve finds the source present, available, unitless.
        _retrain(app)
        ha.unit, ha.state = None, "3.2"
        assert _publish(app) == {""}
        assert json.loads(path.read_text()) == {}

        # A later restart with the target absent must not resurrect kW.
        app2 = _make_app(_StubHA(unit=None, state=None), path)
        app2._restore_source_units()
        assert _publish(app2) == {""}


class TestRestoreFile:
    def test_missing_file_is_a_no_op(self, tmp_path):
        app = _make_app(_StubHA(), tmp_path / "absent.json")
        app._restore_source_units()
        assert app._source_unit_last_good == {}

    def test_corrupt_file_is_ignored(self, tmp_path, caplog):
        path = tmp_path / "source_units.json"
        path.write_text("{not json")
        app = _make_app(_StubHA(), path)
        with caplog.at_level(logging.WARNING, logger="ml_forecast_lab.main"):
            app._restore_source_units()
        assert app._source_unit_last_good == {}
        assert "Could not read saved source units" in caplog.text

    def test_invalid_entries_are_skipped(self, tmp_path):
        path = tmp_path / "source_units.json"
        path.write_text(json.dumps({
            ENTITY: "kW", "sensor.blank": "  ", "sensor.num": 5,
        }))
        app = _make_app(_StubHA(), path)
        app._restore_source_units()
        assert app._source_unit_last_good == {ENTITY: "kW"}

    def test_non_dict_payload_is_ignored(self, tmp_path):
        path = tmp_path / "source_units.json"
        path.write_text(json.dumps(["kW"]))
        app = _make_app(_StubHA(), path)
        app._restore_source_units()
        assert app._source_unit_last_good == {}

    def test_unwritable_path_never_breaks_publish(self, tmp_path):
        path = tmp_path / "missing_dir" / "source_units.json"
        app = _make_app(_StubHA(unit="kW"), path)
        assert _publish(app) == {"kW"}


def test_persistence_is_off_unless_run_wires_it():
    # Tests and replay shadow apps construct the app directly; they must
    # never write to /data.
    assert MLForecastLabApp()._source_units_path is None


def test_run_restores_units_before_main_loop():
    src = inspect.getsource(MLForecastLabApp.run)
    path_at = src.index('"/data/ml_forecast_lab/source_units.json"')
    restore_at = src.index("self._restore_source_units()")
    loop_at = src.index("await self.main_loop()")
    assert path_at < restore_at < loop_at
