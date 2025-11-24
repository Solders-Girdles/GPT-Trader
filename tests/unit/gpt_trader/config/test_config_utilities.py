from __future__ import annotations

import os
from decimal import Decimal
from unittest.mock import patch

import pytest

from gpt_trader.config.config_utilities import (
    ValidationError,
    parse_bool_env,
    parse_decimal_env,
    parse_float_env,
    parse_int_env,
    parse_list_env,
    parse_mapping_env,
    validate_required_env,
)


def test_parse_list_env_success():
    with patch.dict(os.environ, {"TEST_LIST": "alpha,beta,gamma"}, clear=True):
        result = parse_list_env("TEST_LIST", str)
    assert result == ["alpha", "beta", "gamma"]


def test_parse_mapping_env_wraps_errors():
    with patch.dict(os.environ, {"BAD_MAP": "value-without-key"}, clear=True):
        with pytest.raises(ValidationError) as exc:
            parse_mapping_env("BAD_MAP", int)
    assert "Failed to parse BAD_MAP" in str(exc.value)


def test_numeric_env_helpers_handle_defaults():
    with patch.dict(
        os.environ,
        {
            "BOOL_FLAG": "true",
            "DECIMAL_VALUE": "1.5",
            "INT_VALUE": "7",
            "FLOAT_VALUE": "2.75",
        },
        clear=True,
    ):
        assert parse_bool_env("BOOL_FLAG", default=False) is True
        assert parse_decimal_env("DECIMAL_VALUE") == Decimal("1.5")
        assert parse_int_env("INT_VALUE") == 7
        assert parse_float_env("FLOAT_VALUE") == 2.75


def test_validate_required_env_raises_when_missing():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValidationError):
            validate_required_env(["MISSING_VAR"])
