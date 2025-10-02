from __future__ import annotations

import pytest

from bot_v2.errors import ValidationError
from bot_v2.validation import (
    CompositeValidator,
    PercentageValidator,
    RegexValidator,
    SymbolValidator,
    Validator,
    validate_config,
    validate_inputs,
)


@validate_inputs(symbol=SymbolValidator(), leverage=PercentageValidator(as_decimal=False))
def open_position(symbol: str, leverage: float, side: str = "long") -> dict[str, float | str]:
    return {"symbol": symbol, "leverage": leverage, "side": side}


def test_validate_inputs_transforms_arguments():
    result = open_position("eth-usd", 50)
    assert result == {"symbol": "ETH-USD", "leverage": 0.5, "side": "long"}


def test_validate_inputs_raises_for_invalid_values():
    with pytest.raises(ValidationError):
        open_position("invalid symbol", 50)

    with pytest.raises(ValidationError):
        open_position("eth-usd", 150)


def test_composite_validator_executes_in_sequence():
    strip_validator = Validator(predicate=lambda value: (bool(value.strip()), value.strip()))
    upper_validator = Validator(predicate=lambda value: (True, value.upper()))
    composite = CompositeValidator(strip_validator, upper_validator, RegexValidator(r"^[A-Z]{3}$"))

    assert composite.validate("  abc ", "code") == "ABC"

    with pytest.raises(ValidationError):
        composite.validate("  12! ", "code")


def test_validate_config_applies_schema_and_warns_extra(caplog):
    schema = {
        "symbol": SymbolValidator(),
        "risk": PercentageValidator(as_decimal=False),
    }
    config = {"symbol": "btc-perp", "risk": 25, "unused": True}

    with caplog.at_level("WARNING"):
        validated = validate_config(config, schema)

    assert validated == {"symbol": "BTC-PERP", "risk": 0.25}
    assert "unused" in caplog.text


def test_validate_config_missing_key_raises():
    schema = {"symbol": SymbolValidator(), "risk": PercentageValidator(as_decimal=False)}
    with pytest.raises(ValidationError):
        validate_config({"symbol": "AAPL"}, schema)
