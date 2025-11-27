"""Pre-built chaos scenarios for common failure modes."""

from decimal import Decimal

from gpt_trader.backtesting.types import ChaosScenario


class MissingCandles(ChaosScenario):
    """Scenario simulating data gaps and missing candles."""

    def __init__(
        self,
        probability: Decimal = Decimal("0.05"),  # 5% chance
        name: str = "missing_candles",
    ):
        super().__init__(
            name=name,
            enabled=True,
            missing_candles_probability=probability,
        )


class StaleMarks(ChaosScenario):
    """Scenario simulating delayed/stale market data."""

    def __init__(
        self,
        delay_seconds: int = 30,  # 30 second delay
        name: str = "stale_marks",
    ):
        super().__init__(
            name=name,
            enabled=True,
            stale_marks_delay_seconds=delay_seconds,
        )


class WideSpread(ChaosScenario):
    """Scenario simulating illiquid markets with wide spreads."""

    def __init__(
        self,
        multiplier: Decimal = Decimal("5.0"),  # 5x wider spreads
        name: str = "wide_spread",
    ):
        super().__init__(
            name=name,
            enabled=True,
            spread_multiplier=multiplier,
        )


class ChaoticOrderErrors(ChaosScenario):
    """Scenario simulating random order rejections."""

    def __init__(
        self,
        probability: Decimal = Decimal("0.1"),  # 10% rejection rate
        name: str = "order_errors",
    ):
        super().__init__(
            name=name,
            enabled=True,
            order_error_probability=probability,
        )


class ChaoticPartialFills(ChaosScenario):
    """Scenario simulating partial order fills."""

    def __init__(
        self,
        probability: Decimal = Decimal("0.2"),  # 20% partial fill rate
        fill_pct: Decimal = Decimal("50"),  # Fill 50% of order
        name: str = "partial_fills",
    ):
        super().__init__(
            name=name,
            enabled=True,
            partial_fill_probability=probability,
            partial_fill_pct=fill_pct,
        )


class NetworkLatency(ChaosScenario):
    """Scenario simulating network delays."""

    def __init__(
        self,
        latency_ms: int = 500,  # 500ms latency
        name: str = "network_latency",
    ):
        super().__init__(
            name=name,
            enabled=True,
            network_latency_ms=latency_ms,
        )


# Pre-configured scenario bundles
def create_volatile_market_scenario() -> ChaosScenario:
    """Create a scenario simulating a volatile market period."""
    return ChaosScenario(
        name="volatile_market",
        enabled=True,
        spread_multiplier=Decimal("3.0"),
        partial_fill_probability=Decimal("0.3"),
        order_error_probability=Decimal("0.05"),
    )


def create_degraded_connectivity_scenario() -> ChaosScenario:
    """Create a scenario simulating poor network conditions."""
    return ChaosScenario(
        name="degraded_connectivity",
        enabled=True,
        missing_candles_probability=Decimal("0.1"),
        stale_marks_delay_seconds=60,
        network_latency_ms=1000,
        order_error_probability=Decimal("0.15"),
    )


def create_flash_crash_scenario() -> ChaosScenario:
    """Create a scenario simulating a flash crash event."""
    return ChaosScenario(
        name="flash_crash",
        enabled=True,
        spread_multiplier=Decimal("10.0"),  # 10x spreads
        partial_fill_probability=Decimal("0.5"),  # 50% partial fills
        order_error_probability=Decimal("0.2"),  # 20% rejections
        network_latency_ms=2000,  # 2 second latency
    )


def create_normal_conditions_scenario() -> ChaosScenario:
    """Create a baseline scenario with minimal chaos."""
    return ChaosScenario(
        name="normal_conditions",
        enabled=True,
        missing_candles_probability=Decimal("0.001"),  # 0.1% missing
        spread_multiplier=Decimal("1.0"),  # Normal spreads
        partial_fill_probability=Decimal("0.05"),  # 5% partial fills
        network_latency_ms=50,  # 50ms latency
    )
