from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class MockPosition:
    """Mock position object for testing."""

    liquidation_price: Decimal | None = None
    mark_price: Decimal | None = None
    mark: Decimal | None = None


@dataclass
class MockConfig:
    """Mock config for testing."""

    min_liquidation_buffer_pct: Decimal = Decimal("0.1")
    max_leverage: Decimal = Decimal("10")
    daily_loss_limit_pct: Decimal | None = None
    mark_staleness_threshold: float = 30.0
    volatility_threshold_pct: Decimal | None = None
    max_exposure_pct: float = 100.0  # 10000% - effectively no limit for leverage tests
