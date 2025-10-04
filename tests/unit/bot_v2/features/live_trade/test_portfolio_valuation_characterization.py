"""
Characterization tests for PortfolioValuationService.

These tests lock in existing behavior before refactoring.
Zero regressions required.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.features.live_trade.portfolio_valuation import (
    MarkDataSource,
    PortfolioSnapshot,
    PortfolioValuationService,
)


class TestPortfolioSnapshotCharacterization:
    """Characterization tests for PortfolioSnapshot."""

    def test_snapshot_initialization_with_defaults(self):
        """Test snapshot creates with default values."""
        now = datetime.now()
        snapshot = PortfolioSnapshot(
            timestamp=now,
            total_equity_usd=Decimal("10000"),
            cash_balance=Decimal("5000"),
            positions_value=Decimal("5000"),
            realized_pnl=Decimal("100"),
            unrealized_pnl=Decimal("50"),
            funding_pnl=Decimal("-10"),
            fees_paid=Decimal("25"),
        )

        assert snapshot.timestamp == now
        assert snapshot.total_equity_usd == Decimal("10000")
        assert snapshot.positions == {}
        assert snapshot.leverage == Decimal("0")
        assert snapshot.stale_marks == set()

    def test_snapshot_to_dict_serialization(self):
        """Test snapshot serializes to dict correctly."""
        now = datetime(2025, 1, 1, 12, 0, 0)
        snapshot = PortfolioSnapshot(
            timestamp=now,
            total_equity_usd=Decimal("10000"),
            cash_balance=Decimal("5000"),
            positions_value=Decimal("5000"),
            realized_pnl=Decimal("100"),
            unrealized_pnl=Decimal("50"),
            funding_pnl=Decimal("-10"),
            fees_paid=Decimal("25"),
            leverage=Decimal("2.5"),
            margin_used=Decimal("2000"),
            margin_available=Decimal("3000"),
        )

        result = snapshot.to_dict()

        assert result["timestamp"] == "2025-01-01T12:00:00"
        assert result["total_equity_usd"] == 10000.0
        assert result["leverage"] == 2.5
        assert result["stale_marks"] == []

    def test_snapshot_with_positions_serialization(self):
        """Test snapshot with positions serializes correctly."""
        now = datetime.now()
        positions = {
            "BTC-USD": {
                "side": "long",
                "quantity": Decimal("1.5"),
                "mark_price": Decimal("50000"),
                "notional_value": Decimal("75000"),
            }
        }

        snapshot = PortfolioSnapshot(
            timestamp=now,
            total_equity_usd=Decimal("10000"),
            cash_balance=Decimal("5000"),
            positions_value=Decimal("75000"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            funding_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
            positions=positions,
        )

        result = snapshot.to_dict()

        assert "BTC-USD" in result["positions"]
        assert result["positions"]["BTC-USD"]["quantity"] == 1.5

    def test_snapshot_with_stale_marks(self):
        """Test snapshot tracks stale marks."""
        now = datetime.now()
        snapshot = PortfolioSnapshot(
            timestamp=now,
            total_equity_usd=Decimal("10000"),
            cash_balance=Decimal("5000"),
            positions_value=Decimal("5000"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            funding_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
            stale_marks={"BTC-USD", "ETH-USD"},
        )

        result = snapshot.to_dict()

        assert set(result["stale_marks"]) == {"BTC-USD", "ETH-USD"}


class TestMarkDataSourceCharacterization:
    """Characterization tests for MarkDataSource."""

    def test_mark_source_initialization(self):
        """Test mark source initializes with threshold."""
        source = MarkDataSource(staleness_threshold_seconds=30)

        assert source.staleness_threshold == timedelta(seconds=30)

    def test_update_and_get_mark(self):
        """Test updating and retrieving mark price."""
        source = MarkDataSource(staleness_threshold_seconds=30)
        now = datetime.now()

        source.update_mark("BTC-USD", Decimal("50000"), now)

        result = source.get_mark("BTC-USD")
        assert result is not None
        price, is_stale = result
        assert price == Decimal("50000")
        assert is_stale is False

    def test_get_mark_missing_symbol(self):
        """Test getting mark for missing symbol returns None."""
        source = MarkDataSource()

        result = source.get_mark("MISSING-USD")

        assert result is None

    def test_mark_staleness_detection(self):
        """Test mark becomes stale after threshold."""
        source = MarkDataSource(staleness_threshold_seconds=1)
        old_time = datetime.now() - timedelta(seconds=5)

        source.update_mark("BTC-USD", Decimal("50000"), old_time)

        result = source.get_mark("BTC-USD")
        assert result is not None
        price, is_stale = result
        assert is_stale is True

    def test_get_all_marks_excludes_stale(self):
        """Test get_all_marks excludes stale prices."""
        source = MarkDataSource(staleness_threshold_seconds=1)
        now = datetime.now()
        old_time = now - timedelta(seconds=5)

        source.update_mark("BTC-USD", Decimal("50000"), now)
        source.update_mark("ETH-USD", Decimal("3000"), old_time)

        marks = source.get_all_marks()

        assert "BTC-USD" in marks
        assert "ETH-USD" not in marks

    def test_get_stale_symbols(self):
        """Test get_stale_symbols identifies stale data."""
        source = MarkDataSource(staleness_threshold_seconds=1)
        now = datetime.now()
        old_time = now - timedelta(seconds=5)

        source.update_mark("BTC-USD", Decimal("50000"), now)
        source.update_mark("ETH-USD", Decimal("3000"), old_time)

        stale = source.get_stale_symbols()

        assert "ETH-USD" in stale
        assert "BTC-USD" not in stale


class TestPortfolioValuationServiceInitialization:
    """Characterization tests for service initialization."""

    def test_service_initializes_with_defaults(self):
        """Test service initializes with default values."""
        service = PortfolioValuationService()

        assert service.pnl_tracker is not None
        assert service.mark_source is not None
        assert service.snapshot_interval == timedelta(minutes=5)

    def test_service_initializes_with_custom_params(self):
        """Test service initializes with custom parameters."""
        service = PortfolioValuationService(mark_staleness_seconds=60, snapshot_interval_minutes=10)

        assert service.mark_source.staleness_threshold == timedelta(seconds=60)
        assert service.snapshot_interval == timedelta(minutes=10)

    def test_service_cache_initialized_empty(self):
        """Test service cache starts empty."""
        service = PortfolioValuationService()

        assert service._cache_timestamp is None
        assert service._cached_balances == {}
        assert service._cached_positions == {}


class TestAccountDataManagement:
    """Characterization tests for account data caching."""

    def test_update_account_data(self):
        """Test updating account balances and positions."""
        service = PortfolioValuationService()

        balances = [
            Balance(
                asset="USD", available=Decimal("10000"), total=Decimal("10000"), hold=Decimal("0")
            )
        ]
        positions = [
            Position(
                symbol="BTC-USD",
                quantity=Decimal("1.5"),
                entry_price=Decimal("48000"),
                mark_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            )
        ]

        service.update_account_data(balances, positions)

        assert "USD" in service._cached_balances
        assert "BTC-USD" in service._cached_positions
        assert service._cache_timestamp is not None

    def test_cache_validity_check_valid(self):
        """Test cache validity returns True when fresh."""
        service = PortfolioValuationService()
        service._cache_timestamp = datetime.now()

        is_valid = service._is_cache_valid()

        assert is_valid is True

    def test_cache_validity_check_stale(self):
        """Test cache validity returns False when stale."""
        service = PortfolioValuationService()
        service._cache_timestamp = datetime.now() - timedelta(seconds=60)

        is_valid = service._is_cache_valid()

        assert is_valid is False

    def test_cache_validity_check_missing(self):
        """Test cache validity returns False when no timestamp."""
        service = PortfolioValuationService()

        is_valid = service._is_cache_valid()

        assert is_valid is False


class TestMarkPriceUpdates:
    """Characterization tests for mark price updates."""

    def test_update_mark_prices(self):
        """Test updating mark prices."""
        service = PortfolioValuationService()

        mark_prices = {"BTC-USD": Decimal("50000"), "ETH-USD": Decimal("3000")}

        service.update_mark_prices(mark_prices)

        btc_mark = service.mark_source.get_mark("BTC-USD")
        assert btc_mark is not None
        assert btc_mark[0] == Decimal("50000")

    def test_update_mark_prices_updates_pnl_tracker(self):
        """Test mark prices update PnL tracker."""
        service = PortfolioValuationService()

        mark_prices = {"BTC-USD": Decimal("50000")}

        # This should not raise
        service.update_mark_prices(mark_prices)


class TestTradeUpdates:
    """Characterization tests for trade execution updates."""

    def test_update_trade_execution(self):
        """Test updating with trade execution."""
        service = PortfolioValuationService()

        result = service.update_trade(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            fees=Decimal("50"),
        )

        assert "realized_pnl" in result

    def test_update_trade_reduce_position(self):
        """Test updating with reduce-only trade."""
        service = PortfolioValuationService()

        # First add position
        service.update_trade("BTC-USD", "buy", Decimal("1.0"), Decimal("50000"))

        # Then reduce
        result = service.update_trade(
            "BTC-USD",
            "sell",
            Decimal("0.5"),
            Decimal("51000"),
            is_reduce=True,
        )

        assert "realized_pnl" in result


class TestPortfolioValuation:
    """Characterization tests for portfolio valuation computation."""

    def test_compute_valuation_with_cash_only(self):
        """Test valuation with cash balance only."""
        service = PortfolioValuationService()

        balances = [
            Balance(
                asset="USD", available=Decimal("10000"), total=Decimal("10000"), hold=Decimal("0")
            )
        ]
        service.update_account_data(balances, [])

        snapshot = service.compute_current_valuation()

        assert snapshot.cash_balance == Decimal("10000")
        assert snapshot.positions_value == Decimal("0")

    def test_compute_valuation_with_position(self):
        """Test valuation with active position."""
        service = PortfolioValuationService()

        balances = [
            Balance(
                asset="USD", available=Decimal("5000"), total=Decimal("5000"), hold=Decimal("0")
            )
        ]
        positions = [
            Position(
                symbol="BTC-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("48000"),
                mark_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            )
        ]

        service.update_account_data(balances, positions)
        service.update_mark_prices({"BTC-USD": Decimal("50000")})
        service.update_trade("BTC-USD", "buy", Decimal("1.0"), Decimal("48000"))

        snapshot = service.compute_current_valuation()

        assert snapshot.positions_value == Decimal("50000")
        assert "BTC-USD" in snapshot.positions

    def test_margin_calculation(self):
        """Test margin and leverage calculation."""
        service = PortfolioValuationService()

        balances = [
            Balance(
                asset="USD", available=Decimal("10000"), total=Decimal("10000"), hold=Decimal("0")
            )
        ]
        positions = [
            Position(
                symbol="BTC-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000"),
                mark_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            )
        ]

        service.update_account_data(balances, positions)
        service.update_mark_prices({"BTC-USD": Decimal("50000")})

        snapshot = service.compute_current_valuation()

        # Margin used = positions_value * 0.1 (10x leverage assumption)
        assert snapshot.margin_used == Decimal("5000")
        # Leverage = positions_value / cash
        assert snapshot.leverage == Decimal("5")

    def test_valuation_handles_missing_mark_price(self):
        """Test valuation handles position without mark price."""
        service = PortfolioValuationService()

        balances = [
            Balance(
                asset="USD", available=Decimal("10000"), total=Decimal("10000"), hold=Decimal("0")
            )
        ]
        positions = [
            Position(
                symbol="BTC-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000"),
                mark_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            )
        ]

        service.update_account_data(balances, positions)
        # Don't update marks

        snapshot = service.compute_current_valuation()

        assert "BTC-USD" in snapshot.missing_positions

    def test_valuation_detects_stale_marks(self):
        """Test valuation detects stale mark prices."""
        service = PortfolioValuationService(mark_staleness_seconds=1)

        balances = [
            Balance(
                asset="USD", available=Decimal("10000"), total=Decimal("10000"), hold=Decimal("0")
            )
        ]
        positions = [
            Position(
                symbol="BTC-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000"),
                mark_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            )
        ]

        service.update_account_data(balances, positions)

        # Update mark with old timestamp
        old_time = datetime.now() - timedelta(seconds=5)
        service.mark_source.update_mark("BTC-USD", Decimal("50000"), old_time)

        snapshot = service.compute_current_valuation()

        assert "BTC-USD" in snapshot.stale_marks

    def test_valuation_handles_multiple_currencies(self):
        """Test cash calculation with multiple currencies."""
        service = PortfolioValuationService()

        balances = [
            Balance(
                asset="USD", available=Decimal("5000"), total=Decimal("5000"), hold=Decimal("0")
            ),
            Balance(
                asset="USDC", available=Decimal("3000"), total=Decimal("3000"), hold=Decimal("0")
            ),
            Balance(
                asset="USDT", available=Decimal("2000"), total=Decimal("2000"), hold=Decimal("0")
            ),
        ]

        service.update_account_data(balances, [])

        snapshot = service.compute_current_valuation()

        assert snapshot.cash_balance == Decimal("10000")


class TestSnapshotManagement:
    """Characterization tests for snapshot storage."""

    def test_should_create_snapshot_on_first_call(self):
        """Test first snapshot is always created."""
        service = PortfolioValuationService()

        should_create = service._should_create_snapshot()

        assert should_create is True

    def test_should_not_create_snapshot_before_interval(self):
        """Test snapshot not created before interval."""
        service = PortfolioValuationService(snapshot_interval_minutes=5)
        service._last_snapshot_time = datetime.now()

        should_create = service._should_create_snapshot()

        assert should_create is False

    def test_should_create_snapshot_after_interval(self):
        """Test snapshot created after interval."""
        service = PortfolioValuationService(snapshot_interval_minutes=1)
        service._last_snapshot_time = datetime.now() - timedelta(minutes=2)

        should_create = service._should_create_snapshot()

        assert should_create is True

    def test_store_snapshot(self):
        """Test storing snapshot."""
        service = PortfolioValuationService()
        now = datetime.now()

        snapshot = PortfolioSnapshot(
            timestamp=now,
            total_equity_usd=Decimal("10000"),
            cash_balance=Decimal("10000"),
            positions_value=Decimal("0"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            funding_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
        )

        service._store_snapshot(snapshot)

        assert len(service._snapshots) == 1
        assert service._last_snapshot_time == now

    def test_snapshot_retention_limit(self):
        """Test snapshot retention enforces max limit."""
        service = PortfolioValuationService()

        # Create 2020 snapshots (over limit of 2016)
        for i in range(2020):
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now() + timedelta(minutes=i),
                total_equity_usd=Decimal("10000"),
                cash_balance=Decimal("10000"),
                positions_value=Decimal("0"),
                realized_pnl=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                funding_pnl=Decimal("0"),
                fees_paid=Decimal("0"),
            )
            service._store_snapshot(snapshot)

        assert len(service._snapshots) == 2016


class TestEquityCurve:
    """Characterization tests for equity curve generation."""

    def test_get_equity_curve_empty(self):
        """Test equity curve with no snapshots."""
        service = PortfolioValuationService()

        curve = service.get_equity_curve(hours_back=24)

        assert curve == []

    def test_get_equity_curve_with_snapshots(self):
        """Test equity curve filters by time period."""
        service = PortfolioValuationService()
        now = datetime.now()

        # Create snapshots
        old_snapshot = PortfolioSnapshot(
            timestamp=now - timedelta(hours=48),
            total_equity_usd=Decimal("9000"),
            cash_balance=Decimal("9000"),
            positions_value=Decimal("0"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            funding_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
        )

        recent_snapshot = PortfolioSnapshot(
            timestamp=now - timedelta(hours=12),
            total_equity_usd=Decimal("10000"),
            cash_balance=Decimal("10000"),
            positions_value=Decimal("0"),
            realized_pnl=Decimal("100"),
            unrealized_pnl=Decimal("50"),
            funding_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
        )

        service._snapshots = [old_snapshot, recent_snapshot]

        curve = service.get_equity_curve(hours_back=24)

        assert len(curve) == 1
        assert curve[0]["equity"] == 10000.0


class TestDailyMetrics:
    """Characterization tests for daily metrics."""

    def test_get_daily_metrics_no_snapshots(self):
        """Test daily metrics with no snapshots."""
        service = PortfolioValuationService()

        metrics = service.get_daily_metrics()

        assert metrics == {}

    def test_get_daily_metrics_with_snapshot(self):
        """Test daily metrics uses latest snapshot."""
        service = PortfolioValuationService()

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_equity_usd=Decimal("10000"),
            cash_balance=Decimal("10000"),
            positions_value=Decimal("0"),
            realized_pnl=Decimal("100"),
            unrealized_pnl=Decimal("50"),
            funding_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
        )

        service._snapshots = [snapshot]

        metrics = service.get_daily_metrics()

        # Should delegate to PnL tracker
        assert isinstance(metrics, dict)
