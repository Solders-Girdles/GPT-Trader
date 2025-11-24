# Validation Rules Overview

The validation package contains two complementary building blocks:

* **Legacy validators** (e.g. `SymbolValidator`, `PercentageValidator`) which raise
  `gpt_trader.errors.ValidationError` and are primarily used outside Pydantic models.
* **Composable rules** defined in `validation/rules.py` that are pure functions. They can be
  chained via `RuleChain`, plugged into Pydantic `field_validator`s, or reused in non-Pydantic
  contexts. Rules raise `RuleError` when coercion or validation fails.

## Common Rules

| Rule | Purpose | Example |
|------|---------|---------|
| `BooleanRule` | Parse friendly boolean tokens | `BooleanRule(default=False)("yes")  # True` |
| `IntegerRule` / `FloatRule` / `DecimalRule` | Coerce numeric values with defaults | `FloatRule(default=1.0)(None)` |
| `PercentageRule` | Ensure value is between 0 and 1 | `PercentageRule()(0.8)` |
| `TimeOfDayRule` | Validate `HH:MM` strings | `TimeOfDayRule()("09:30")` |
| `ListRule` | Parse lists with item coercion | `ListRule(item_rule=SymbolRule())("BTC-USD,ETH-USD")` |
| `MappingRule` | Parse mappings from strings or dicts | `MappingRule(value_converter=int)("BTC:1,ETH:2")` |
| `SymbolRule` | Validate and normalise trading symbols | `SymbolRule()("btc-perp")  # \"BTC-PERP\"` |

## Using Rules with Pydantic

```python
from pydantic import field_validator
from gpt_trader.validation import FloatRule, ListRule, SymbolRule

PRICE_RULE = FloatRule()
SYMBOL_LIST = ListRule(item_rule=SymbolRule(), allow_blank_items=False)

class ExampleModel(BaseModel):
    symbol_allowlist: list[str]
    max_slippage_pct: float

    @field_validator("symbol_allowlist", mode="before")
    @classmethod
    def _normalise_symbols(cls, value: object) -> list[str]:
        return SYMBOL_LIST(value, "symbol_allowlist")

    @field_validator("max_slippage_pct", mode="before")
    @classmethod
    def _coerce_slippage(cls, value: object) -> float:
        return PRICE_RULE(value, "max_slippage_pct")
```

Pydantic validators typically convert `RuleError` into `PydanticCustomError` via helper functions like `_apply_rule`.

## Handling Errors Outside Pydantic

When using rules directly, catch `RuleError` to surface application-specific messages:

```python
from gpt_trader.validation import RuleError, SymbolRule

rule = SymbolRule()

try:
    symbol = rule(user_input, "trade_symbol")
except RuleError as exc:
    raise ConfigurationError(f"Invalid symbol: {exc}") from exc
```

## Tips

* Use `RuleChain` to build multi-step pipelines: stripping input, defaulting values, then applying numeric/regex checks.
* For environment parsing, combine rules with helper adapters (see `config/env_utils.py`) to ensure consistent coercion across env vars and Pydantic configs.
* Prefer creating new rule classes rather than bespoke validators when a pattern recurs—this keeps error handling uniform and testable.
