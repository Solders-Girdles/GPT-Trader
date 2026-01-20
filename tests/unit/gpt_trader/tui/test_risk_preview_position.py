"""Tests for risk_preview position impact computation."""

from decimal import Decimal

import pytest

from gpt_trader.tui.risk_preview import _compute_position_impacts, compute_preview
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.thresholds import DEFAULT_RISK_THRESHOLDS
from gpt_trader.tui.types import PortfolioSummary, Position, RiskState


class TestComputePositionImpacts:
    """Tests for _compute_position_impacts helper."""

    @pytest.fixture
    def tui_state_with_positions(self) -> TuiState:
        """Create TuiState with positions for testing."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.05,  # 5% daily limit
            current_daily_loss_pct=-0.01,
        )
        # Add positions
        btc_pos = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            unrealized_pnl=Decimal("-200"),  # -2% of equity
            mark_price=Decimal("48000"),
            side="LONG",
            leverage=1,
        )
        eth_pos = Position(
            symbol="ETH-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("3000"),
            unrealized_pnl=Decimal("-50"),  # -0.5% of equity
            mark_price=Decimal("2950"),
            side="LONG",
            leverage=1,
        )
        state.position_data = PortfolioSummary(
            positions={"BTC-USD": btc_pos, "ETH-USD": eth_pos},
            equity=Decimal("10000"),
        )
        return state

    def test_no_positions_returns_empty(self):
        """Empty positions returns empty impacts."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.05,
        )
        state.position_data = PortfolioSummary(positions={}, equity=Decimal("10000"))

        impacts = _compute_position_impacts(state, -0.10, 0.05, DEFAULT_RISK_THRESHOLDS)
        assert impacts == []

    def test_zero_equity_returns_empty(self):
        """Zero equity returns empty impacts."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(max_leverage=5.0, daily_loss_limit_pct=0.05)
        state.position_data = PortfolioSummary(
            positions={"BTC-USD": Position(symbol="BTC-USD", quantity=Decimal("1"))},
            equity=Decimal("0"),
        )

        impacts = _compute_position_impacts(state, -0.10, 0.05, DEFAULT_RISK_THRESHOLDS)
        assert impacts == []

    def test_negative_shock_creates_impacts_for_long_positions(
        self, tui_state_with_positions: TuiState
    ):
        """Negative shock creates impacts for long positions losing value."""
        impacts = _compute_position_impacts(
            tui_state_with_positions, -0.10, 0.05, DEFAULT_RISK_THRESHOLDS
        )

        # Should have at least one impact for DailyLossGuard
        daily_loss_impacts = [i for i in impacts if i.guard_name == "DailyLossGuard"]
        assert len(daily_loss_impacts) > 0

        # Check BTC impact is present
        btc_impacts = [i for i in daily_loss_impacts if i.symbol == "BTC-USD"]
        assert len(btc_impacts) == 1
        assert btc_impacts[0].projected_pnl_pct < 0
        assert "→" in btc_impacts[0].reason

    def test_positive_shock_gains_for_long_positions(self, tui_state_with_positions: TuiState):
        """Positive shock creates gains for long positions (no breach)."""
        impacts = _compute_position_impacts(
            tui_state_with_positions, +0.10, 0.05, DEFAULT_RISK_THRESHOLDS
        )

        # With positive shock, projected should improve from current
        for impact in impacts:
            assert impact.projected_pnl_pct >= impact.current_pnl_pct

    def test_leverage_guard_triggers_on_excessive_leverage(self):
        """LeverageGuard triggers when position leverage exceeds max."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=2.0,  # Max 2x
            daily_loss_limit_pct=0.05,
        )
        # Position with 5x leverage
        high_lev_pos = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            mark_price=Decimal("50000"),
            side="LONG",
            leverage=5,  # Exceeds max of 2x
        )
        state.position_data = PortfolioSummary(
            positions={"BTC-USD": high_lev_pos},
            equity=Decimal("10000"),
        )

        impacts = _compute_position_impacts(state, -0.05, 0.05, DEFAULT_RISK_THRESHOLDS)

        leverage_impacts = [i for i in impacts if i.guard_name == "LeverageGuard"]
        assert len(leverage_impacts) == 1
        assert leverage_impacts[0].symbol == "BTC-USD"
        assert "5x >" in leverage_impacts[0].reason and "max" in leverage_impacts[0].reason

    def test_impacts_sorted_by_projected_loss(self, tui_state_with_positions: TuiState):
        """Impacts are sorted by projected loss (worst first)."""
        impacts = _compute_position_impacts(
            tui_state_with_positions, -0.10, 0.05, DEFAULT_RISK_THRESHOLDS
        )

        if len(impacts) > 1:
            for i in range(len(impacts) - 1):
                assert impacts[i].projected_pnl_pct <= impacts[i + 1].projected_pnl_pct


class TestComputePreviewWithPositionImpacts:
    """Tests for compute_preview returning position impacts."""

    @pytest.fixture
    def tui_state_with_positions(self) -> TuiState:
        """Create TuiState with positions for testing."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.05,
            current_daily_loss_pct=-0.01,
        )
        btc_pos = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            unrealized_pnl=Decimal("-500"),
            mark_price=Decimal("49000"),
            side="LONG",
            leverage=1,
        )
        state.position_data = PortfolioSummary(
            positions={"BTC-USD": btc_pos},
            equity=Decimal("10000"),
        )
        return state

    def test_compute_preview_includes_position_impacts(self, tui_state_with_positions: TuiState):
        """compute_preview returns position_impacts field."""
        result = compute_preview(tui_state_with_positions, -0.10, label="-10%")

        assert hasattr(result, "position_impacts")
        assert isinstance(result.position_impacts, list)

    def test_no_limit_returns_empty_position_impacts(self):
        """No daily limit configured returns empty position impacts."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.0,  # No limit
        )

        result = compute_preview(state, -0.10, label="-10%")
        assert result.position_impacts == []
