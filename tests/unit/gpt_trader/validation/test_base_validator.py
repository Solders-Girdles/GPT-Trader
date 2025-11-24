"""Tests for the foundational Validator helper."""

from __future__ import annotations

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation import Validator


def test_validator_no_predicate_is_pass_through():
    validator = Validator()
    assert validator.validate(42, "answer") == 42


def test_validator_predicate_bool_result():
    validator = Validator(predicate=lambda value: value > 0)

    assert validator.validate(10, "positive") == 10

    with pytest.raises(ValidationError, match="positive"):
        validator.validate(-1, "positive")


def test_validator_predicate_tuple_allows_transformation():
    validator = Validator(predicate=lambda value: (bool(value.strip()), value.strip().upper()))

    assert validator.validate("  btc-perp  ", "symbol") == "BTC-PERP"

    with pytest.raises(ValidationError, match="symbol"):
        validator.validate("   ", "symbol")


def test_validator_wraps_predicate_exceptions():
    def explode(_):
        raise RuntimeError("boom")

    validator = Validator(error_message="bad field", predicate=explode)

    with pytest.raises(ValidationError, match="bad field"):
        validator.validate("value", "bad_field")
