"""Tests for chaos scenario factories."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.backtesting.chaos.scenarios import (
    ChaoticOrderErrors,
    ChaoticPartialFills,
    MissingCandles,
    NetworkLatency,
    StaleMarks,
    WideSpread,
    create_degraded_connectivity_scenario,
    create_flash_crash_scenario,
    create_normal_conditions_scenario,
    create_volatile_market_scenario,
)


class TestMissingCandles:
    """Tests for MissingCandles scenario."""

    def test_default_probability(self) -> None:
        scenario = MissingCandles()
        assert scenario.missing_candles_probability == Decimal("0.05")

    def test_custom_probability(self) -> None:
        scenario = MissingCandles(probability=Decimal("0.1"))
        assert scenario.missing_candles_probability == Decimal("0.1")

    def test_default_name(self) -> None:
        scenario = MissingCandles()
        assert scenario.name == "missing_candles"

    def test_custom_name(self) -> None:
        scenario = MissingCandles(name="custom_missing")
        assert scenario.name == "custom_missing"

    def test_enabled_by_default(self) -> None:
        scenario = MissingCandles()
        assert scenario.enabled is True


class TestStaleMarks:
    """Tests for StaleMarks scenario."""

    def test_default_delay(self) -> None:
        scenario = StaleMarks()
        assert scenario.stale_marks_delay_seconds == 30

    def test_custom_delay(self) -> None:
        scenario = StaleMarks(delay_seconds=60)
        assert scenario.stale_marks_delay_seconds == 60

    def test_default_name(self) -> None:
        scenario = StaleMarks()
        assert scenario.name == "stale_marks"

    def test_enabled_by_default(self) -> None:
        scenario = StaleMarks()
        assert scenario.enabled is True


class TestWideSpread:
    """Tests for WideSpread scenario."""

    def test_default_multiplier(self) -> None:
        scenario = WideSpread()
        assert scenario.spread_multiplier == Decimal("5.0")

    def test_custom_multiplier(self) -> None:
        scenario = WideSpread(multiplier=Decimal("10.0"))
        assert scenario.spread_multiplier == Decimal("10.0")

    def test_default_name(self) -> None:
        scenario = WideSpread()
        assert scenario.name == "wide_spread"


class TestChaoticOrderErrors:
    """Tests for ChaoticOrderErrors scenario."""

    def test_default_probability(self) -> None:
        scenario = ChaoticOrderErrors()
        assert scenario.order_error_probability == Decimal("0.1")

    def test_custom_probability(self) -> None:
        scenario = ChaoticOrderErrors(probability=Decimal("0.2"))
        assert scenario.order_error_probability == Decimal("0.2")

    def test_default_name(self) -> None:
        scenario = ChaoticOrderErrors()
        assert scenario.name == "order_errors"


class TestChaoticPartialFills:
    """Tests for ChaoticPartialFills scenario."""

    def test_default_probability(self) -> None:
        scenario = ChaoticPartialFills()
        assert scenario.partial_fill_probability == Decimal("0.2")

    def test_default_fill_pct(self) -> None:
        scenario = ChaoticPartialFills()
        assert scenario.partial_fill_pct == Decimal("50")

    def test_custom_params(self) -> None:
        scenario = ChaoticPartialFills(probability=Decimal("0.5"), fill_pct=Decimal("25"))
        assert scenario.partial_fill_probability == Decimal("0.5")
        assert scenario.partial_fill_pct == Decimal("25")

    def test_default_name(self) -> None:
        scenario = ChaoticPartialFills()
        assert scenario.name == "partial_fills"


class TestNetworkLatency:
    """Tests for NetworkLatency scenario."""

    def test_default_latency(self) -> None:
        scenario = NetworkLatency()
        assert scenario.network_latency_ms == 500

    def test_custom_latency(self) -> None:
        scenario = NetworkLatency(latency_ms=1000)
        assert scenario.network_latency_ms == 1000

    def test_default_name(self) -> None:
        scenario = NetworkLatency()
        assert scenario.name == "network_latency"


class TestVolatileMarketScenario:
    """Tests for create_volatile_market_scenario factory."""

    def test_creates_scenario(self) -> None:
        scenario = create_volatile_market_scenario()
        assert scenario.name == "volatile_market"
        assert scenario.enabled is True

    def test_spread_multiplier(self) -> None:
        scenario = create_volatile_market_scenario()
        assert scenario.spread_multiplier == Decimal("3.0")

    def test_partial_fill_probability(self) -> None:
        scenario = create_volatile_market_scenario()
        assert scenario.partial_fill_probability == Decimal("0.3")

    def test_order_error_probability(self) -> None:
        scenario = create_volatile_market_scenario()
        assert scenario.order_error_probability == Decimal("0.05")


class TestDegradedConnectivityScenario:
    """Tests for create_degraded_connectivity_scenario factory."""

    def test_creates_scenario(self) -> None:
        scenario = create_degraded_connectivity_scenario()
        assert scenario.name == "degraded_connectivity"
        assert scenario.enabled is True

    def test_missing_candles_probability(self) -> None:
        scenario = create_degraded_connectivity_scenario()
        assert scenario.missing_candles_probability == Decimal("0.1")

    def test_stale_marks_delay(self) -> None:
        scenario = create_degraded_connectivity_scenario()
        assert scenario.stale_marks_delay_seconds == 60

    def test_network_latency(self) -> None:
        scenario = create_degraded_connectivity_scenario()
        assert scenario.network_latency_ms == 1000


class TestFlashCrashScenario:
    """Tests for create_flash_crash_scenario factory."""

    def test_creates_scenario(self) -> None:
        scenario = create_flash_crash_scenario()
        assert scenario.name == "flash_crash"
        assert scenario.enabled is True

    def test_extreme_spread_multiplier(self) -> None:
        scenario = create_flash_crash_scenario()
        assert scenario.spread_multiplier == Decimal("10.0")

    def test_high_partial_fill_probability(self) -> None:
        scenario = create_flash_crash_scenario()
        assert scenario.partial_fill_probability == Decimal("0.5")

    def test_high_error_probability(self) -> None:
        scenario = create_flash_crash_scenario()
        assert scenario.order_error_probability == Decimal("0.2")

    def test_high_latency(self) -> None:
        scenario = create_flash_crash_scenario()
        assert scenario.network_latency_ms == 2000


class TestNormalConditionsScenario:
    """Tests for create_normal_conditions_scenario factory."""

    def test_creates_scenario(self) -> None:
        scenario = create_normal_conditions_scenario()
        assert scenario.name == "normal_conditions"
        assert scenario.enabled is True

    def test_minimal_missing_candles(self) -> None:
        scenario = create_normal_conditions_scenario()
        assert scenario.missing_candles_probability == Decimal("0.001")

    def test_normal_spread_multiplier(self) -> None:
        scenario = create_normal_conditions_scenario()
        assert scenario.spread_multiplier == Decimal("1.0")

    def test_low_partial_fill_probability(self) -> None:
        scenario = create_normal_conditions_scenario()
        assert scenario.partial_fill_probability == Decimal("0.05")

    def test_low_latency(self) -> None:
        scenario = create_normal_conditions_scenario()
        assert scenario.network_latency_ms == 50
