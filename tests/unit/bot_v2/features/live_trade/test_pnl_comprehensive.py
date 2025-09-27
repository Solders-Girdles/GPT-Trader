"""
Comprehensive P&L tests for perpetual futures trading.

Combines behavioral and realistic test scenarios to ensure correctness
of all P&L calculations including realized, unrealized, and funding.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from bot_v2.features.live_trade.pnl_tracker import (
    PositionState, FundingCalculator, PnLTracker
)

# Skip this module if the expected comprehensive PnL API isn't available
_ps_required = ['product_id', 'side', 'entry_price', 'size']
_ps_methods = ['calculate_unrealized_pnl']
_tracker_required = ['add_fill', 'get_position', 'apply_funding', 'calculate_unrealized_pnl', 'get_total_unrealized_pnl']
_funding_required = ['calculate_payment']

def _has_all(obj, names):
    return all(hasattr(obj, n) for n in names)

if not (
    _has_all(PositionState, _ps_methods) and
    _has_all(PnLTracker, _tracker_required) and
    _has_all(FundingCalculator, _funding_required)
):
    pytest.skip(
        "Skipping comprehensive PnL tests: Advanced PnL API not available in this build",
        allow_module_level=True,
    )


class TestBasicPnLCalculations:
    """Test fundamental P&L calculations with deterministic scenarios."""
    
    def test_long_position_profit(self):
        """Test P&L for profitable long position."""
        position = PositionState(
            product_id="BTC-PERP",
            side="long",
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            timestamp=datetime.now()
        )
        
        # Price increases to 52000
        current_price = Decimal("52000")
        unrealized_pnl = position.calculate_unrealized_pnl(current_price)
        
        # (52000 - 50000) * 0.1 = 200
        assert unrealized_pnl == Decimal("200")
    
    def test_long_position_loss(self):
        """Test P&L for losing long position."""
        position = PositionState(
            product_id="BTC-PERP",
            side="long", 
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            timestamp=datetime.now()
        )
        
        # Price decreases to 48000
        current_price = Decimal("48000")
        unrealized_pnl = position.calculate_unrealized_pnl(current_price)
        
        # (48000 - 50000) * 0.1 = -200
        assert unrealized_pnl == Decimal("-200")
    
    def test_short_position_profit(self):
        """Test P&L for profitable short position."""
        position = PositionState(
            product_id="BTC-PERP",
            side="short",
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            timestamp=datetime.now()
        )
        
        # Price decreases to 48000 (profit for short)
        current_price = Decimal("48000")
        unrealized_pnl = position.calculate_unrealized_pnl(current_price)
        
        # (50000 - 48000) * 0.1 = 200
        assert unrealized_pnl == Decimal("200")
    
    def test_short_position_loss(self):
        """Test P&L for losing short position."""
        position = PositionState(
            product_id="BTC-PERP",
            side="short",
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            timestamp=datetime.now()
        )
        
        # Price increases to 52000 (loss for short)
        current_price = Decimal("52000")
        unrealized_pnl = position.calculate_unrealized_pnl(current_price)
        
        # (50000 - 52000) * 0.1 = -200
        assert unrealized_pnl == Decimal("-200")


class TestPartialFills:
    """Test P&L with partial fills and average entry prices."""
    
    def test_multiple_fills_average_price(self):
        """Test average entry price with multiple fills."""
        tracker = PnLTracker()
        
        # First fill: 0.05 BTC at 50000
        tracker.add_fill(
            product_id="BTC-PERP",
            side="buy",
            size=Decimal("0.05"),
            price=Decimal("50000"),
            timestamp=datetime.now()
        )
        
        # Second fill: 0.05 BTC at 51000
        tracker.add_fill(
            product_id="BTC-PERP",
            side="buy",
            size=Decimal("0.05"),
            price=Decimal("51000"),
            timestamp=datetime.now()
        )
        
        position = tracker.get_position("BTC-PERP")
        # Average: (0.05 * 50000 + 0.05 * 51000) / 0.1 = 50500
        assert position.entry_price == Decimal("50500")
        assert position.size == Decimal("0.1")
    
    def test_partial_close_realized_pnl(self):
        """Test realized P&L when partially closing position."""
        tracker = PnLTracker()
        
        # Open long: 0.1 BTC at 50000
        tracker.add_fill(
            product_id="BTC-PERP",
            side="buy",
            size=Decimal("0.1"),
            price=Decimal("50000"),
            timestamp=datetime.now()
        )
        
        # Partial close: sell 0.05 BTC at 52000
        realized = tracker.add_fill(
            product_id="BTC-PERP",
            side="sell",
            size=Decimal("0.05"),
            price=Decimal("52000"),
            timestamp=datetime.now()
        )
        
        # Realized P&L: (52000 - 50000) * 0.05 = 100
        assert realized == Decimal("100")
        
        # Remaining position: 0.05 BTC long
        position = tracker.get_position("BTC-PERP")
        assert position.size == Decimal("0.05")


class TestFundingPayments:
    """Test funding payment calculations for perpetuals."""
    
    def test_positive_funding_long_pays(self):
        """Test long position pays when funding is positive."""
        calculator = FundingCalculator()
        
        position_value = Decimal("10000")  # 0.2 BTC at 50000
        funding_rate = Decimal("0.01")  # 1% (positive)
        
        # Long pays: -10000 * 0.01 = -100
        payment = calculator.calculate_payment(
            position_value=position_value,
            funding_rate=funding_rate,
            side="long"
        )
        assert payment == Decimal("-100")
    
    def test_positive_funding_short_receives(self):
        """Test short position receives when funding is positive."""
        calculator = FundingCalculator()
        
        position_value = Decimal("10000")
        funding_rate = Decimal("0.01")  # 1% (positive)
        
        # Short receives: 10000 * 0.01 = 100
        payment = calculator.calculate_payment(
            position_value=position_value,
            funding_rate=funding_rate,
            side="short"
        )
        assert payment == Decimal("100")
    
    def test_negative_funding_long_receives(self):
        """Test long position receives when funding is negative."""
        calculator = FundingCalculator()
        
        position_value = Decimal("10000")
        funding_rate = Decimal("-0.01")  # -1% (negative)
        
        # Long receives: -10000 * (-0.01) = 100
        payment = calculator.calculate_payment(
            position_value=position_value,
            funding_rate=funding_rate,
            side="long"
        )
        assert payment == Decimal("100")
    
    def test_cumulative_funding_tracking(self):
        """Test tracking cumulative funding over time."""
        tracker = PnLTracker()
        
        # Open position
        tracker.add_fill(
            product_id="BTC-PERP",
            side="buy",
            size=Decimal("0.1"),
            price=Decimal("50000"),
            timestamp=datetime.now()
        )
        
        # Apply multiple funding payments
        tracker.apply_funding(
            product_id="BTC-PERP",
            funding_rate=Decimal("0.01"),
            timestamp=datetime.now()
        )
        
        tracker.apply_funding(
            product_id="BTC-PERP",
            funding_rate=Decimal("-0.005"),
            timestamp=datetime.now() + timedelta(hours=8)
        )
        
        position = tracker.get_position("BTC-PERP")
        # First: -5000 * 0.01 = -50
        # Second: -5000 * (-0.005) = 25
        # Total: -50 + 25 = -25
        assert position.cumulative_funding == Decimal("-25")


class TestComplexScenarios:
    """Test complex trading scenarios with multiple positions."""
    
    def test_flip_position_long_to_short(self):
        """Test flipping from long to short position."""
        tracker = PnLTracker()
        
        # Open long: 0.1 BTC at 50000
        tracker.add_fill(
            product_id="BTC-PERP",
            side="buy",
            size=Decimal("0.1"),
            price=Decimal("50000"),
            timestamp=datetime.now()
        )
        
        # Flip to short: sell 0.2 BTC at 51000
        # This closes 0.1 long and opens 0.1 short
        realized = tracker.add_fill(
            product_id="BTC-PERP",
            side="sell",
            size=Decimal("0.2"),
            price=Decimal("51000"),
            timestamp=datetime.now()
        )
        
        # Realized from closing long: (51000 - 50000) * 0.1 = 100
        assert realized == Decimal("100")
        
        # New position: 0.1 BTC short at 51000
        position = tracker.get_position("BTC-PERP")
        assert position.side == "short"
        assert position.size == Decimal("0.1")
        assert position.entry_price == Decimal("51000")
    
    def test_multiple_products_tracking(self):
        """Test tracking P&L across multiple products."""
        tracker = PnLTracker()
        
        # BTC position
        tracker.add_fill(
            product_id="BTC-PERP",
            side="buy",
            size=Decimal("0.1"),
            price=Decimal("50000"),
            timestamp=datetime.now()
        )
        
        # ETH position
        tracker.add_fill(
            product_id="ETH-PERP",
            side="sell",
            size=Decimal("1.0"),
            price=Decimal("3000"),
            timestamp=datetime.now()
        )
        
        # Calculate total unrealized P&L
        btc_pnl = tracker.calculate_unrealized_pnl(
            "BTC-PERP", Decimal("51000")
        )
        eth_pnl = tracker.calculate_unrealized_pnl(
            "ETH-PERP", Decimal("2900")
        )
        
        # BTC: (51000 - 50000) * 0.1 = 100
        # ETH: (3000 - 2900) * 1.0 = 100
        assert btc_pnl == Decimal("100")
        assert eth_pnl == Decimal("100")
        
        total_pnl = tracker.get_total_unrealized_pnl({
            "BTC-PERP": Decimal("51000"),
            "ETH-PERP": Decimal("2900")
        })
        assert total_pnl == Decimal("200")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_size_position(self):
        """Test handling of zero-size positions."""
        position = PositionState(
            product_id="BTC-PERP",
            side="long",
            entry_price=Decimal("50000"),
            size=Decimal("0"),
            timestamp=datetime.now()
        )
        
        # Zero size should result in zero P&L
        unrealized_pnl = position.calculate_unrealized_pnl(Decimal("60000"))
        assert unrealized_pnl == Decimal("0")
    
    def test_exact_breakeven_price(self):
        """Test P&L at exact breakeven price."""
        position = PositionState(
            product_id="BTC-PERP",
            side="long",
            entry_price=Decimal("50000"),
            size=Decimal("0.1"),
            timestamp=datetime.now()
        )
        
        # Same price as entry
        unrealized_pnl = position.calculate_unrealized_pnl(Decimal("50000"))
        assert unrealized_pnl == Decimal("0")
    
    def test_very_small_position_precision(self):
        """Test precision with very small positions."""
        position = PositionState(
            product_id="BTC-PERP",
            side="long",
            entry_price=Decimal("50000"),
            size=Decimal("0.001"),  # Minimum size
            timestamp=datetime.now()
        )
        
        # Small price move
        current_price = Decimal("50010")
        unrealized_pnl = position.calculate_unrealized_pnl(current_price)
        
        # (50010 - 50000) * 0.001 = 0.01
        assert unrealized_pnl == Decimal("0.01")
    
    def test_rounding_consistency(self):
        """Test consistent rounding in calculations."""
        tracker = PnLTracker()
        
        # Multiple fills that could cause rounding issues
        fills = [
            (Decimal("0.033"), Decimal("50000.33")),
            (Decimal("0.033"), Decimal("50001.67")),
            (Decimal("0.034"), Decimal("49998.50"))
        ]
        
        for size, price in fills:
            tracker.add_fill(
                product_id="BTC-PERP",
                side="buy",
                size=size,
                price=price,
                timestamp=datetime.now()
            )
        
        position = tracker.get_position("BTC-PERP")
        
        # Total size should be exact
        assert position.size == Decimal("0.1")
        
        # Average price should be properly weighted
        total_value = sum(size * price for size, price in fills)
        total_size = sum(size for size, _ in fills)
        expected_avg = total_value / total_size
        
        # Allow small rounding difference
        assert abs(position.entry_price - expected_avg) < Decimal("0.01")
