from decimal import Decimal

import pytest

from bot_v2.config.env_utils import (
    EnvVarError,
    get_env_bool,
    get_env_decimal,
    parse_env_mapping,
)


def test_parse_env_mapping_success(monkeypatch):
    monkeypatch.setenv("TEST_MAPPING", "BTC-PERP:10, ETH-PERP: 5")

    result = parse_env_mapping("TEST_MAPPING", int)

    assert result == {"BTC-PERP": 10, "ETH-PERP": 5}


def test_parse_env_mapping_invalid_pair_raises(monkeypatch):
    monkeypatch.setenv("TEST_MAPPING", "BTC-PERP-10")

    with pytest.raises(EnvVarError) as exc:
        parse_env_mapping("TEST_MAPPING", int)

    assert "TEST_MAPPING" in str(exc.value)


def test_get_env_bool_accepts_various_truthy_values(monkeypatch):
    monkeypatch.setenv("TEST_BOOL", "Yes")
    assert get_env_bool("TEST_BOOL") is True

    monkeypatch.setenv("TEST_BOOL", "off")
    assert get_env_bool("TEST_BOOL") is False


def test_get_env_bool_invalid_value(monkeypatch):
    monkeypatch.setenv("TEST_BOOL", "maybe")

    with pytest.raises(EnvVarError):
        get_env_bool("TEST_BOOL")


def test_get_env_decimal(monkeypatch):
    monkeypatch.setenv("TEST_DECIMAL", "100.5")

    value = get_env_decimal("TEST_DECIMAL")

    assert isinstance(value, Decimal)
    assert value == Decimal("100.5")
