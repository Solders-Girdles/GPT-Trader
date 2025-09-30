# Behavioral Scenario Utilities Demo

The `tests/fixtures/behavioral` package provides reusable scenarios for validating
PnL calculations, funding flows, and risk-limit enforcement without contacting
live markets.  This page replaces the former `test_behavioral_demo` suite so the
examples remain easy to discover without blocking CI.

## Quick Start

```python
from decimal import Decimal
from tests.fixtures.behavioral import (
    create_realistic_btc_scenario,
    run_behavioral_validation,
)

scenario = create_realistic_btc_scenario(
    scenario_type="profit",
    position_size=Decimal("0.1"),
    hold_days=1,
)

actual_results = {
    "final_position": Decimal("0"),
    "realized_pnl": scenario.expected_pnl,
    "total_fees": scenario.expected_fees,
    "funding_paid": scenario.funding_payments,
}

passed, errors = run_behavioral_validation(scenario, actual_results)
assert passed and not errors
```

### Included Helpers

- `create_realistic_btc_scenario` / `create_realistic_eth_scenario`
  produce deterministic trade paths with modern price levels.
- `create_market_stress_scenario` collects extreme-but-plausible market moves
  for stress testing execution logic.
- `create_funding_scenario` and `create_risk_limit_test_scenario` model
  non-price risks such as funding costs and leverage caps.
- `validate_pnl_calculation` offers a light-weight check for ad-hoc trades.

All scenarios return plain dataclasses and decimal values, making them safe to
use directly inside notebooks, docs, or targeted unit tests.

## Usage Notes

- Scenarios bake in **static price levels** representative of late 2024 market
  conditions.  Treat them as fixtures, not real-market references.
- When adding new scenarios, document them here and keep expectations fuzzy
  enough that they age gracefully.
- Example notebooks and tutorials can import directly from
  `tests.fixtures.behavioral`—no pytest markers required.

## When to Add a Test

Use these helpers to back a focused unit test when you need deterministic
behavioral coverage (e.g., ensuring a new execution engine respects risk caps).
For narrative walkthroughs or onboarding material, prefer referencing this page
or embedding the snippets into docs notebooks instead of adding broad demo tests
under `tests/unit/`.
