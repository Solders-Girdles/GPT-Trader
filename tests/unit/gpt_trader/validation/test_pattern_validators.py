"""Tests for pattern-based validators."""

from __future__ import annotations

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation.pattern_validators import (
    StrategyNameValidator,
    SymbolValidator,
)


class TestSymbolValidator:
    """Tests for SymbolValidator."""

    def test_validates_equity_symbol(self) -> None:
        validator = SymbolValidator()
        assert validator.validate("AAPL", "symbol") == "AAPL"

    def test_validates_lowercase_equity(self) -> None:
        validator = SymbolValidator()
        assert validator.validate("msft", "symbol") == "MSFT"

    def test_validates_crypto_spot_pair(self) -> None:
        validator = SymbolValidator()
        assert validator.validate("BTC-USD", "symbol") == "BTC-USD"

    def test_validates_crypto_perpetual(self) -> None:
        validator = SymbolValidator()
        assert validator.validate("ETH-PERP", "symbol") == "ETH-PERP"

    def test_validates_mixed_case(self) -> None:
        validator = SymbolValidator()
        assert validator.validate("btc-usd", "symbol") == "BTC-USD"

    def test_validates_numeric_in_symbol(self) -> None:
        validator = SymbolValidator()
        assert validator.validate("BTC2-USD", "symbol") == "BTC2-USD"

    def test_validates_short_symbol(self) -> None:
        validator = SymbolValidator()
        assert validator.validate("A", "symbol") == "A"

    def test_validates_long_symbol(self) -> None:
        validator = SymbolValidator()
        assert validator.validate("GOOGL", "symbol") == "GOOGL"

    def test_rejects_empty_string(self) -> None:
        validator = SymbolValidator()
        with pytest.raises(ValidationError, match="must be a valid symbol"):
            validator.validate("", "symbol")

    def test_rejects_special_characters(self) -> None:
        validator = SymbolValidator()
        with pytest.raises(ValidationError, match="must be a valid symbol"):
            validator.validate("BTC@USD", "symbol")

    def test_rejects_spaces(self) -> None:
        validator = SymbolValidator()
        with pytest.raises(ValidationError, match="must be a valid symbol"):
            validator.validate("BTC USD", "symbol")

    def test_rejects_non_string(self) -> None:
        validator = SymbolValidator()
        with pytest.raises(ValidationError, match="must be a string"):
            validator.validate(123, "symbol")

    def test_rejects_too_long_base(self) -> None:
        validator = SymbolValidator()
        # Base part > 10 characters
        with pytest.raises(ValidationError, match="must be a valid symbol"):
            validator.validate("VERYLONGSYMBOL", "symbol")

    def test_field_name_in_error(self) -> None:
        validator = SymbolValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("invalid!", "trading_pair")
        assert exc_info.value.context["field"] == "trading_pair"


class TestStrategyNameValidator:
    """Tests for StrategyNameValidator."""

    def test_validates_pascal_case(self) -> None:
        validator = StrategyNameValidator()
        assert validator.validate("SimpleMAStrategy", "strategy") == "SimpleMAStrategy"

    def test_validates_single_word(self) -> None:
        validator = StrategyNameValidator()
        assert validator.validate("Momentum", "strategy") == "Momentum"

    def test_validates_snake_case(self) -> None:
        validator = StrategyNameValidator()
        assert validator.validate("mean_reversion", "strategy") == "mean_reversion"

    def test_validates_with_numbers(self) -> None:
        validator = StrategyNameValidator()
        assert validator.validate("Strategy2024", "strategy") == "Strategy2024"

    def test_validates_with_dashes(self) -> None:
        validator = StrategyNameValidator()
        assert validator.validate("my-strategy", "strategy") == "my-strategy"

    def test_validates_mixed_separators(self) -> None:
        validator = StrategyNameValidator()
        assert validator.validate("My_Cool-Strategy2", "strategy") == "My_Cool-Strategy2"

    def test_rejects_starting_with_number(self) -> None:
        validator = StrategyNameValidator()
        with pytest.raises(ValidationError, match="must start with a letter"):
            validator.validate("2024Strategy", "strategy")

    def test_rejects_starting_with_underscore(self) -> None:
        validator = StrategyNameValidator()
        with pytest.raises(ValidationError, match="must start with a letter"):
            validator.validate("_private", "strategy")

    def test_rejects_special_characters(self) -> None:
        validator = StrategyNameValidator()
        with pytest.raises(ValidationError, match="must start with a letter"):
            validator.validate("my@strategy", "strategy")

    def test_rejects_spaces(self) -> None:
        validator = StrategyNameValidator()
        with pytest.raises(ValidationError, match="must start with a letter"):
            validator.validate("my strategy", "strategy")

    def test_rejects_non_string(self) -> None:
        validator = StrategyNameValidator()
        with pytest.raises(ValidationError, match="must be a string"):
            validator.validate(42, "strategy")

    def test_field_name_in_error(self) -> None:
        validator = StrategyNameValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("123bad", "strategy_name")
        assert exc_info.value.context["field"] == "strategy_name"
