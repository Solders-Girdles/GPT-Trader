"""
Behavioral testing fixtures and utilities.

Provides deterministic test data generators for behavioral testing.
"""

from .market_data import (
    MarketDataGenerator,
    TradeScenarioBuilder,
    ValidationConstants,
    RealisticMarketData
)
from .trade_scenarios import (
    TradeExecution,
    TradeScenario,
    create_long_profit_scenario,
    create_short_loss_scenario,
    create_funding_payment_scenario,
    create_stop_loss_scenario,
    create_partial_close_scenario,
    create_position_flip_scenario
)
from .validators import (
    validate_pnl_calculation,
    validate_position_state,
    validate_risk_limits,
    validate_order_execution,
    validate_funding_accrual
)
from .utils import (
    create_realistic_btc_scenario,
    create_realistic_eth_scenario,
    create_funding_scenario,
    create_risk_limit_test_scenario,
    run_behavioral_validation,
    create_market_stress_scenario
)
from .clock import FakeClock

__all__ = [
    # Market data utilities
    'MarketDataGenerator',
    'TradeScenarioBuilder', 
    'ValidationConstants',
    'RealisticMarketData',
    
    # Trade scenarios
    'TradeExecution',
    'TradeScenario',
    'create_long_profit_scenario',
    'create_short_loss_scenario', 
    'create_funding_payment_scenario',
    'create_stop_loss_scenario',
    'create_partial_close_scenario',
    'create_position_flip_scenario',
    
    # Validators
    'validate_pnl_calculation',
    'validate_position_state',
    'validate_risk_limits',
    'validate_order_execution',
    'validate_funding_accrual',
    
    # Utilities
    'create_realistic_btc_scenario',
    'create_realistic_eth_scenario',
    'create_funding_scenario',
    'create_risk_limit_test_scenario',
    'run_behavioral_validation',
    'create_market_stress_scenario',

    # Clock helpers
    'FakeClock'
]
