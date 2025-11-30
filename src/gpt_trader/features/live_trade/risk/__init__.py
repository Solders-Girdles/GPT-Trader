"""
Risk management for live perpetuals trading.

This package provides real-time risk controls for the live trading engine:

- **Pre-trade validation**: Leverage limits, exposure caps, liquidation buffers
- **Runtime monitoring**: Daily PnL tracking, mark price staleness detection
- **Circuit breakers**: Volatility triggers, automatic reduce-only mode

Key Components
--------------
- ``LiveRiskManager``: Central risk control class
- ``ValidationError``: Raised when risk checks fail

Configuration
-------------
Risk parameters are loaded from ``RiskConfig`` (see ``orchestration.configuration.risk``).

Key config fields:

- ``max_leverage``: Global leverage cap (default: 5x)
- ``daily_loss_limit_pct``: Daily loss threshold for circuit breaker
- ``max_exposure_pct``: Maximum portfolio exposure
- ``day_leverage_max_per_symbol``: Per-symbol day session limits
- ``night_leverage_max_per_symbol``: Per-symbol night session limits

Example::

    from gpt_trader.features.live_trade.risk import LiveRiskManager, ValidationError
    from gpt_trader.orchestration.configuration.risk import RiskConfig

    config = RiskConfig(max_leverage=3, daily_loss_limit_pct=0.05)
    risk_manager = LiveRiskManager(config=config)

    try:
        risk_manager.pre_trade_validate(symbol, side, quantity, price, ...)
    except ValidationError as e:
        # Order blocked by risk rules
        pass
"""

from .manager import LiveRiskManager, ValidationError

__all__ = ["LiveRiskManager", "ValidationError"]
