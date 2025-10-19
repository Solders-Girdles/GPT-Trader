"""
Behavioral testing fixtures and helpers.

Provides deterministic test data generators for behavioral testing.
"""

from .clock import FakeClock
from .helpers import (
    create_funding_scenario,
    create_market_stress_scenario,
    create_realistic_btc_scenario,
    create_realistic_eth_scenario,
    create_risk_limit_test_scenario,
    run_behavioral_validation,
)
from .market_data import (
    MarketDataGenerator,
    RealisticMarketData,
    TradeScenarioBuilder,
    ValidationConstants,
)
from .trade_scenarios import (
    TradeExecution,
    TradeScenario,
    create_funding_payment_scenario,
    create_long_profit_scenario,
    create_partial_close_scenario,
    create_position_flip_scenario,
    create_short_loss_scenario,
    create_stop_loss_scenario,
)
from .validators import (
    validate_funding_accrual,
    validate_order_execution,
    validate_pnl_calculation,
    validate_position_state,
    validate_risk_limits,
)

__all__ = [
    # Market data utilities
    "MarketDataGenerator",
    "TradeScenarioBuilder",
    "ValidationConstants",
    "RealisticMarketData",
    # Trade scenarios
    "TradeExecution",
    "TradeScenario",
    "create_long_profit_scenario",
    "create_short_loss_scenario",
    "create_funding_payment_scenario",
    "create_stop_loss_scenario",
    "create_partial_close_scenario",
    "create_position_flip_scenario",
    # Validators
    "validate_pnl_calculation",
    "validate_position_state",
    "validate_risk_limits",
    "validate_order_execution",
    "validate_funding_accrual",
    # Scenario helpers
    "create_realistic_btc_scenario",
    "create_realistic_eth_scenario",
    "create_funding_scenario",
    "create_risk_limit_test_scenario",
    "run_behavioral_validation",
    "create_market_stress_scenario",
    # Clock helpers
    "FakeClock",
]
