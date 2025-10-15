from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import pytest

from bot_v2.config.types import Profile
from bot_v2.orchestration.configuration.core import BotConfig
from bot_v2.utilities import config as config_utils


def _make_config(**overrides: Any) -> BotConfig:
    base_kwargs = {
        "profile": Profile.DEV,
        "symbols": ["BTC-USD", "ETH-USD"],
        "trading_days": ["mon", "tue"],
        "metadata": {"ignored": True},
    }
    base_kwargs.update(overrides)
    return BotConfig(**base_kwargs)


def test_config_baseline_payload_normalizes_sequences() -> None:
    baseline = config_utils.ConfigBaselinePayload.from_config(
        _make_config(), derivatives_enabled=True
    )

    assert "metadata" not in baseline.fields
    assert baseline.data["symbols"] == ("BTC-USD", "ETH-USD")
    assert baseline.data["trading_days"] == ("mon", "tue")
    assert baseline.data["derivatives_enabled"] is True

    presented = baseline.to_dict()
    assert presented["symbols"] == ["BTC-USD", "ETH-USD"]
    assert presented["trading_days"] == ["mon", "tue"]


def test_config_baseline_payload_diff_reports_changes() -> None:
    current = config_utils.ConfigBaselinePayload.from_config(
        _make_config(trading_days=["mon", "tue"]), derivatives_enabled=True
    )
    updated = config_utils.ConfigBaselinePayload.from_config(
        _make_config(trading_days=["mon", "wed"]), derivatives_enabled=False
    )

    diff = current.diff(updated)

    assert diff["derivatives_enabled"] == {"current": True, "new": False}
    assert diff["trading_days"] == {"current": ["mon", "tue"], "new": ["mon", "wed"]}


def test_parse_slippage_multipliers_handles_valid_entries() -> None:
    result = config_utils.parse_slippage_multipliers("BTC-USD:1.25, ETH-USD : 0.5")

    assert result == {
        "BTC-USD": Decimal("1.25"),
        "ETH-USD": Decimal("0.5"),
    }


@pytest.mark.parametrize("payload", ["", None, "   "])
def test_parse_slippage_multipliers_handles_empty_inputs(payload: str | None) -> None:
    assert config_utils.parse_slippage_multipliers(payload) == {}


@pytest.mark.parametrize(
    "payload,match",
    [
        ("BTC-USD", "Invalid slippage entry"),
        ("BTC-USD:not-a-number", "Invalid multiplier"),
        (":1.0", "Missing symbol"),
    ],
)
def test_parse_slippage_multipliers_raises_for_invalid_payloads(payload: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        config_utils.parse_slippage_multipliers(payload)


def test_load_slippage_multipliers_uses_env_mapping_override() -> None:
    result = config_utils.load_slippage_multipliers(
        env={config_utils.SLIPPAGE_ENV_KEY: "BTC-USD:1.10"}
    )

    assert result == {"BTC-USD": Decimal("1.10")}


def test_load_slippage_multipliers_logs_invalid_runtime_values(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class DummySettings:
        def __init__(self) -> None:
            self.raw_env = {config_utils.SLIPPAGE_ENV_KEY: "BTC-USD:not-a-number"}

    caplog.set_level(logging.WARNING, config_utils.logger.name)
    monkeypatch.setattr(config_utils, "load_runtime_settings", lambda: DummySettings())

    result = config_utils.load_slippage_multipliers()

    assert result == {}
    assert f"Invalid {config_utils.SLIPPAGE_ENV_KEY} entry" in caplog.text
