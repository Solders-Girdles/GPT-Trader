"""Common fixtures for strategy handler tests."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.position_size_calculator import PositionSizeCalculator
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioSnapshot,
    PortfolioTier,
    PositionConstraints,
    PositionInfo,
    RiskProfile,
    TierConfig,
    TradingRules,
)


class SimpleFrame:
    """Lightweight stand-in for DataFrame objects used in handlers."""

    def __init__(self, data: dict[str, list[float]]) -> None:
        self.data = data

    def __len__(self) -> int:  # pragma: no cover - trivial
        column = next(iter(self.data.values()), [])
        return len(column)


@pytest.fixture
def simple_frame_factory() -> Callable[[list[float]], SimpleFrame]:
    """Factory producing SimpleFrame instances with close price series."""

    def factory(prices: list[float]) -> SimpleFrame:
        return SimpleFrame({"Close": prices})

    return factory


@pytest.fixture
def micro_tier_config() -> TierConfig:
    """Representative micro-tier configuration."""

    return TierConfig(
        name="micro",
        range=(0, 5000),
        positions=PositionConstraints(1, 3, 2, 100.0),
        min_position_size=100.0,
        strategies=["momentum"],
        risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
        trading=TradingRules(5, "cash", 2, True),
    )


@pytest.fixture
def portfolio_snapshot() -> PortfolioSnapshot:
    """Sample portfolio snapshot used across strategy tests."""

    return PortfolioSnapshot(
        total_value=3000.0,
        cash=2000.0,
        positions=[PositionInfo("AAPL", 10, 100.0, 100.0, 1000.0, 0.0, 0.0, 1)],
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        quarterly_pnl_pct=0.0,
        current_tier=PortfolioTier.MICRO,
        positions_count=1,
        largest_position_pct=33.33,
        sector_exposures={},
    )


@pytest.fixture
def position_size_calculator_mock() -> Mock:
    """Mocked position size calculator returning scaled confidence."""

    calculator = Mock(spec=PositionSizeCalculator)

    def calculate(confidence: float, *_: object, **__: object) -> float:
        return round(confidence * 1000, 2)

    calculator.calculate.side_effect = calculate
    return calculator
