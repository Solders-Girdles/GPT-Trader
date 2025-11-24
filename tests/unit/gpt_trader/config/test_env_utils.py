from decimal import Decimal

import pytest

from gpt_trader.config.env_utils import (
    EnvVarError,
    coerce_env_value,
    get_env_bool,
    get_env_decimal,
    get_env_int,
    parse_env_list,
    parse_env_mapping,
    require_env_value,
)


class StubSettings:
    def __init__(self, raw_env: dict[str, str]) -> None:
        self.raw_env = raw_env


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


def test_coerce_env_value_uses_default_when_missing():
    settings = StubSettings({})

    result = coerce_env_value("MISSING", int, default=5, settings=settings)

    assert result == 5


def test_coerce_env_value_required_raises():
    settings = StubSettings({})

    with pytest.raises(EnvVarError):
        coerce_env_value("REQUIRED", int, required=True, settings=settings)


def test_require_env_value_returns_cast_result():
    settings = StubSettings({"REQUIRED": "42"})

    value = require_env_value("REQUIRED", int, settings=settings)

    assert value == 42


def test_parse_env_list_disallows_empty_entries():
    settings = StubSettings({"TEST_LIST": "alpha,,beta"})

    with pytest.raises(EnvVarError) as exc:
        parse_env_list("TEST_LIST", str, allow_empty=False, settings=settings)

    assert "empty list entry" in str(exc.value)


def test_parse_env_list_cast_failure():
    settings = StubSettings({"TEST_LIST": "1, two"})

    with pytest.raises(EnvVarError) as exc:
        parse_env_list("TEST_LIST", int, settings=settings)

    assert "could not cast value 'two'" in str(exc.value)


def test_parse_env_mapping_disallows_empty_key():
    settings = StubSettings({"TEST_MAP": ":1"})

    with pytest.raises(EnvVarError) as exc:
        parse_env_mapping("TEST_MAP", int, settings=settings)

    assert "empty key" in str(exc.value)


def test_parse_env_mapping_disallows_empty_value():
    settings = StubSettings({"TEST_MAP": "BTC-PERP:"})

    with pytest.raises(EnvVarError) as exc:
        parse_env_mapping("TEST_MAP", int, settings=settings)

    assert "empty value" in str(exc.value)


def test_parse_env_mapping_required_missing():
    settings = StubSettings({})

    with pytest.raises(EnvVarError):
        parse_env_mapping("MAPPING", int, required=True, settings=settings)


def test_parse_env_mapping_skips_empty_entries_when_allowed():
    settings = StubSettings({"TEST_MAP": "BTC:1,,ETH:2"})

    result = parse_env_mapping("TEST_MAP", int, allow_empty=True, settings=settings)

    assert result == {"BTC": 1, "ETH": 2}
