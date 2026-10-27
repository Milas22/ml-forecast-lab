"""Tests for the Home Assistant connection settings used outside the Supervisor.

Standalone Docker deployments have no Supervisor, so the base URL, the
token and TLS verification come from HA_URL / HA_TOKEN / HA_VERIFY_SSL.
Inside the add-on none of these are set and the Supervisor defaults must
stay exactly as before.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from ml_forecast_lab import ha_interface
from ml_forecast_lab.ha_interface import (
    HAInterface,
    ha_base_url,
    ha_ssl_kwargs,
    ha_token,
    ha_verify_ssl,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start every test with none of the connection variables set."""
    for var in ("HA_URL", "HA_TOKEN", "HA_VERIFY_SSL", "SUPERVISOR_TOKEN"):
        monkeypatch.delenv(var, raising=False)


class _FakeResponse:
    status = 200
    headers = {"content-type": "application/json"}

    async def json(self):
        return {"ok": True}

    async def text(self):
        return "ok"


class _FakeRequest:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Records the kwargs of every request() call."""

    closed = False

    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return _FakeRequest(_FakeResponse())


# ---- ha_base_url ----

def test_base_url_defaults_to_supervisor_proxy():
    assert ha_base_url() == "http://supervisor/core"


def test_base_url_uses_ha_url_and_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("HA_URL", "https://ha.example.org:8123/")
    assert ha_base_url() == "https://ha.example.org:8123"


def test_empty_ha_url_falls_back_to_supervisor(monkeypatch):
    monkeypatch.setenv("HA_URL", "")
    assert ha_base_url() == "http://supervisor/core"


# ---- ha_token ----

def test_token_defaults_to_empty_without_any_variable():
    assert ha_token() == ""


def test_token_uses_supervisor_token_inside_addon(monkeypatch):
    monkeypatch.setenv("SUPERVISOR_TOKEN", "sup")
    assert ha_token() == "sup"


def test_ha_token_wins_over_supervisor_token(monkeypatch):
    monkeypatch.setenv("SUPERVISOR_TOKEN", "sup")
    monkeypatch.setenv("HA_TOKEN", "llat")
    assert ha_token() == "llat"


# ---- ha_verify_ssl ----

def test_verify_ssl_defaults_to_true():
    assert ha_verify_ssl() is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", " FALSE "])
def test_verify_ssl_can_be_disabled(monkeypatch, value):
    monkeypatch.setenv("HA_VERIFY_SSL", value)
    assert ha_verify_ssl() is False


@pytest.mark.parametrize("value", ["true", "1", "yes", ""])
def test_verify_ssl_other_values_keep_verification(monkeypatch, value):
    monkeypatch.setenv("HA_VERIFY_SSL", value)
    assert ha_verify_ssl() is True


# ---- ha_ssl_kwargs ----

def test_ssl_kwargs_empty_while_verifying():
    assert ha_ssl_kwargs() == {}


def test_ssl_kwargs_disable_only_when_opted_out(monkeypatch):
    monkeypatch.setenv("HA_VERIFY_SSL", "false")
    assert ha_ssl_kwargs() == {"ssl": False}


def test_ssl_kwargs_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv("HA_VERIFY_SSL", "false")
    assert ha_ssl_kwargs(True) == {}
    monkeypatch.delenv("HA_VERIFY_SSL")
    assert ha_ssl_kwargs(False) == {"ssl": False}


# ---- HAInterface ----

def test_interface_defaults_unchanged_for_addon(monkeypatch):
    monkeypatch.setenv("SUPERVISOR_TOKEN", "sup")
    iface = HAInterface(session=_FakeSession())
    assert iface.ha_url == "http://supervisor/core"
    assert iface.ha_key == "sup"
    assert iface.verify_ssl is True


def test_interface_picks_up_standalone_settings(monkeypatch):
    monkeypatch.setenv("HA_URL", "https://ha.example.org")
    monkeypatch.setenv("HA_TOKEN", "llat")
    monkeypatch.setenv("HA_VERIFY_SSL", "false")
    iface = HAInterface(session=_FakeSession())
    assert iface.ha_url == "https://ha.example.org"
    assert iface.ha_key == "llat"
    assert iface.verify_ssl is False


def test_explicit_arguments_still_win(monkeypatch):
    monkeypatch.setenv("HA_URL", "https://ha.example.org")
    monkeypatch.setenv("HA_TOKEN", "llat")
    iface = HAInterface(ha_url="http://other", ha_key="k", session=_FakeSession())
    assert iface.ha_url == "http://other"
    assert iface.ha_key == "k"


def test_missing_token_logs_an_error(caplog):
    with caplog.at_level(logging.ERROR, logger=ha_interface.__name__):
        HAInterface(session=_FakeSession())
    assert "HA_TOKEN" in caplog.text


def test_api_call_passes_ssl_setting_to_request(monkeypatch):
    monkeypatch.setenv("HA_URL", "https://ha.example.org")
    monkeypatch.setenv("HA_TOKEN", "llat")
    monkeypatch.setenv("HA_VERIFY_SSL", "false")
    session = _FakeSession()
    iface = HAInterface(session=session)

    result = asyncio.run(iface.api_call("GET", "/api/config"))

    assert result == {"ok": True}
    call = session.calls[0]
    assert call["url"] == "https://ha.example.org/api/config"
    assert call["headers"]["Authorization"] == "Bearer llat"
    assert call["ssl"] is False


def test_api_call_omits_ssl_while_verifying(monkeypatch):
    monkeypatch.setenv("HA_URL", "https://ha.example.org")
    monkeypatch.setenv("HA_TOKEN", "llat")
    session = _FakeSession()
    iface = HAInterface(session=session)

    result = asyncio.run(iface.api_call("GET", "/api/config"))

    assert result == {"ok": True}
    assert "ssl" not in session.calls[0]
