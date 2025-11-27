"""Tests for bot_config rules module."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic_core import PydanticCustomError

from gpt_trader.orchestration.configuration.bot_config.rules import (
    DECIMAL_RULE,
    FLOAT_RULE,
    INT_RULE,
    STRING_RULE,
    SYMBOL_LIST_RULE,
    SYMBOL_RULE,
    apply_rule,
    ensure_condition,
)


class TestRuleInstances:
    """Tests for pre-instantiated rule instances."""

    def test_int_rule_exists(self) -> None:
        assert INT_RULE is not None

    def test_decimal_rule_exists(self) -> None:
        assert DECIMAL_RULE is not None

    def test_float_rule_exists(self) -> None:
        assert FLOAT_RULE is not None

    def test_string_rule_exists(self) -> None:
        assert STRING_RULE is not None

    def test_symbol_rule_exists(self) -> None:
        assert SYMBOL_RULE is not None

    def test_symbol_list_rule_exists(self) -> None:
        assert SYMBOL_LIST_RULE is not None


class TestApplyRule:
    """Tests for apply_rule function."""

    def test_apply_int_rule_success(self) -> None:
        result = apply_rule(
            INT_RULE,
            "42",
            field_label="count",
            error_code="invalid_int",
            error_template="Invalid integer: {value}",
        )
        assert result == 42

    def test_apply_decimal_rule_success(self) -> None:
        result = apply_rule(
            DECIMAL_RULE,
            "10.5",
            field_label="amount",
            error_code="invalid_decimal",
            error_template="Invalid decimal: {value}",
        )
        assert result == Decimal("10.5")

    def test_apply_float_rule_success(self) -> None:
        result = apply_rule(
            FLOAT_RULE,
            "3.14",
            field_label="ratio",
            error_code="invalid_float",
            error_template="Invalid float: {value}",
        )
        assert result == pytest.approx(3.14)

    def test_apply_string_rule_strips_whitespace(self) -> None:
        result = apply_rule(
            STRING_RULE,
            "  hello  ",
            field_label="name",
            error_code="invalid_string",
            error_template="Invalid string: {value}",
        )
        assert result == "hello"

    def test_apply_symbol_rule_success(self) -> None:
        result = apply_rule(
            SYMBOL_RULE,
            "BTC-USD",
            field_label="symbol",
            error_code="invalid_symbol",
            error_template="Invalid symbol: {value}",
        )
        assert result == "BTC-USD"

    def test_apply_rule_raises_pydantic_error_on_failure(self) -> None:
        with pytest.raises(PydanticCustomError):
            apply_rule(
                INT_RULE,
                "not_an_integer",
                field_label="count",
                error_code="invalid_int",
                error_template="Invalid integer: {value}",
            )


class TestEnsureCondition:
    """Tests for ensure_condition function."""

    def test_does_not_raise_when_condition_is_false(self) -> None:
        # Should not raise
        ensure_condition(
            False,
            error_code="test_error",
            error_template="This should not be raised",
            context={"value": "test"},
        )

    def test_raises_when_condition_is_true(self) -> None:
        with pytest.raises(PydanticCustomError):
            ensure_condition(
                True,
                error_code="test_error",
                error_template="Condition was true",
                context={"value": "test"},
            )

    def test_error_code_is_preserved(self) -> None:
        try:
            ensure_condition(
                True,
                error_code="custom_error_code",
                error_template="Error occurred",
                context={"key": "value"},
            )
            pytest.fail("Expected PydanticCustomError")
        except PydanticCustomError as exc:
            assert exc.type == "custom_error_code"


class TestSymbolListRule:
    """Tests for SYMBOL_LIST_RULE."""

    def test_valid_symbol_list(self) -> None:
        result = apply_rule(
            SYMBOL_LIST_RULE,
            ["BTC-USD", "ETH-USD"],
            field_label="symbols",
            error_code="invalid_symbols",
            error_template="Invalid symbols: {value}",
        )
        assert result == ["BTC-USD", "ETH-USD"]

    def test_empty_list(self) -> None:
        result = apply_rule(
            SYMBOL_LIST_RULE,
            [],
            field_label="symbols",
            error_code="invalid_symbols",
            error_template="Invalid symbols: {value}",
        )
        assert result == []

    def test_single_symbol_list(self) -> None:
        result = apply_rule(
            SYMBOL_LIST_RULE,
            ["SOL-USD"],
            field_label="symbols",
            error_code="invalid_symbols",
            error_template="Invalid symbols: {value}",
        )
        assert result == ["SOL-USD"]
