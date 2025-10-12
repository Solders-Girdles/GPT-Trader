from decimal import Decimal

import pytest

from bot_v2.config.env_utils import (
    EnvVarError,
    get_env_bool,
    get_env_decimal,
    get_env_int,
    parse_env_list,
    parse_env_mapping,
)


def test_parse_env_mapping_success(monkeypatch):
    monkeypatch.setenv("TEST_MAPPING", "BTC-PERP:10, ETH-PERP: 5")

    result = parse_env_mapping("TEST_MAPPING", int, kv_delimiter=":")

    assert result == {"BTC-PERP": 10, "ETH-PERP": 5}


def test_parse_env_mapping_uses_default(monkeypatch):
    monkeypatch.delenv("TEST_MAPPING", raising=False)

    result = parse_env_mapping("TEST_MAPPING", int, default={"DEFAULT": 1})

    assert result == {"DEFAULT": 1}


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


def test_parse_env_list(monkeypatch):
    monkeypatch.setenv("TEST_LIST", "alpha, beta ,gamma")

    values = parse_env_list("TEST_LIST", str)

    assert values == ["alpha", "beta", "gamma"]


def test_get_env_int_default(monkeypatch):
    monkeypatch.delenv("TEST_INT", raising=False)

    assert get_env_int("TEST_INT", default=7) == 7
