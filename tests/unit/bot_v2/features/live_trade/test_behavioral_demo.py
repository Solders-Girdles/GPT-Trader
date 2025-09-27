"""
Demonstration of behavioral testing utilities.

Shows how to use the shared behavioral testing framework.
"""

import pytest
from decimal import Decimal
from datetime import datetime

# Import behavioral testing utilities
from tests.fixtures.behavioral import (
    create_realistic_btc_scenario,
    create_realistic_eth_scenario,
    create_funding_scenario,
    create_risk_limit_test_scenario,
    run_behavioral_validation,
    RealisticMarketData,
    validate_pnl_calculation,
    create_market_stress_scenario
)


class TestBehavioralUtilitiesDemo:
    """Demonstration of behavioral testing utilities."""
    
    def test_realistic_btc_profit_scenario(self):
        """Demo: Create and validate realistic BTC profit scenario."""
        # Create scenario using current market prices
        scenario = create_realistic_btc_scenario(
            scenario_type="profit",
            position_size=Decimal('0.1'),  # 0.1 BTC
            hold_days=1
        )
        
        # Scenario should use current BTC price (~$95k)
        btc_price = RealisticMarketData.CURRENT_PRICES['BTC-PERP']
        entry_trade = scenario.trades[0]
        assert entry_trade.price == btc_price
        assert entry_trade.side == "buy"
        assert entry_trade.quantity == Decimal('0.1')
        
        # Exit should be higher for profit scenario
        exit_trade = scenario.trades[1]
        assert exit_trade.price > entry_trade.price
        assert exit_trade.side == "sell"
        assert exit_trade.is_reduce is True
        
        # P&L should be positive
        assert scenario.expected_pnl > 0
        assert scenario.expected_final_position == Decimal('0')
        
        # Validate P&L calculation manually
        expected_pnl = (exit_trade.price - entry_trade.price) * Decimal('0.1')
        assert abs(scenario.expected_pnl - expected_pnl) < Decimal('0.01')
    
    def test_realistic_eth_short_scenario(self):
        """Demo: Create realistic ETH short scenario."""
        scenario = create_realistic_eth_scenario(
            scenario_type="short_profit",
            position_size=Decimal('3.0'),
            volatility="high"
        )
        
        # Should start with short position
        entry_trade = scenario.trades[0]
        assert entry_trade.side == "sell"
        assert entry_trade.quantity == Decimal('3.0')
        
        # Exit should cover the short
        exit_trade = scenario.trades[1]
        assert exit_trade.side == "buy"
        assert exit_trade.is_reduce is True
        
        # For short profit, exit price should be lower than entry
        assert exit_trade.price < entry_trade.price
        assert scenario.expected_pnl > 0  # Profitable short
    
    def test_funding_payment_scenario(self):
        """Demo: Create and validate funding scenario."""
        scenario = create_funding_scenario(
            symbol="BTC-PERP",
            position_size=Decimal('0.5'),
            funding_periods=3,
            funding_rate=Decimal('0.0001')
        )
        
        # Should have zero P&L (no price movement)
        assert scenario.expected_pnl == Decimal('0')
        
        # Should have negative funding (longs pay)
        assert scenario.funding_payments < 0
        
        # Validate funding calculation
        btc_price = RealisticMarketData.CURRENT_PRICES['BTC-PERP']
        notional = Decimal('0.5') * btc_price
        expected_funding = notional * Decimal('0.0001') * 3  # 3 periods
        assert abs(abs(scenario.funding_payments) - expected_funding) < Decimal('1')
    
    def test_risk_limit_scenarios(self):
        """Demo: Risk limit testing scenarios."""
        # Test position limit - should pass
        scenario_pass = create_risk_limit_test_scenario("position", should_pass=True)
        assert scenario_pass['expected_pass'] is True
        assert scenario_pass['quantity'] <= scenario_pass['risk_limits']['max_position_size']
        
        # Test position limit - should fail
        scenario_fail = create_risk_limit_test_scenario("position", should_pass=False)
        assert scenario_fail['expected_pass'] is False
        assert scenario_fail['quantity'] > scenario_fail['risk_limits']['max_position_size']
        
        # Test leverage limit
        leverage_test = create_risk_limit_test_scenario("leverage", should_pass=True)
        notional = leverage_test['quantity'] * leverage_test['price']
        leverage = notional / leverage_test['equity']
        assert leverage <= leverage_test['risk_limits']['max_leverage']
    
    def test_behavioral_validation_framework(self):
        """Demo: End-to-end behavioral validation."""
        # Create a scenario
        scenario = create_realistic_btc_scenario("profit", Decimal('0.05'))
        
        # Simulate trading system results
        actual_results = {
            'final_position': Decimal('0'),
            'realized_pnl': scenario.expected_pnl,
            'total_fees': scenario.expected_fees,
            'funding_paid': scenario.funding_payments
        }
        
        # Run validation
        passed, errors = run_behavioral_validation(scenario, actual_results)
        
        assert passed is True
        assert len(errors) == 0
        
        # Test with incorrect results
        wrong_results = {
            'final_position': Decimal('0.01'),  # Should be zero
            'realized_pnl': scenario.expected_pnl + Decimal('100'),  # Wrong P&L
            'total_fees': scenario.expected_fees
        }
        
        passed_wrong, errors_wrong = run_behavioral_validation(scenario, wrong_results)
        assert passed_wrong is False
        assert len(errors_wrong) > 0
    
    def test_pnl_validation_edge_cases(self):
        """Demo: P&L validation with complex trades."""
        trades = [
            {'side': 'buy', 'quantity': Decimal('1'), 'price': Decimal('95000')},
            {'side': 'sell', 'quantity': Decimal('0.5'), 'price': Decimal('96000')},  # Partial close
            {'side': 'sell', 'quantity': Decimal('0.5'), 'price': Decimal('97000')}   # Final close
        ]
        
        # Expected: (96000-95000)*0.5 + (97000-95000)*0.5 = 500 + 1000 = 1500
        expected_pnl = Decimal('1500')
        
        passed, message = validate_pnl_calculation(trades, expected_pnl)
        assert passed is True
        assert "validated" in message.lower()
    
    def test_market_stress_scenarios(self):
        """Demo: Market stress testing scenarios."""
        stress_scenarios = create_market_stress_scenario()
        
        # Should have multiple scenarios
        assert len(stress_scenarios) >= 3
        
        # Should include different types
        scenario_types = [s.name for s in stress_scenarios]
        assert any('btc' in name for name in scenario_types)
        assert any('eth' in name for name in scenario_types)
        assert any('funding' in name for name in scenario_types)
        
        # All scenarios should have realistic prices
        for scenario in stress_scenarios:
            for trade in scenario.trades:
                # Prices should be in realistic ranges
                if 'BTC' in scenario.symbol:
                    assert Decimal('80000') < trade.price < Decimal('120000')
                elif 'ETH' in scenario.symbol:
                    assert Decimal('2000') < trade.price < Decimal('5000')
    
    def test_current_market_prices_are_realistic(self):
        """Demo: Verify market data reflects current reality."""
        btc_price = RealisticMarketData.CURRENT_PRICES['BTC-PERP']
        eth_price = RealisticMarketData.CURRENT_PRICES['ETH-PERP']
        
        # These should reflect current market conditions (as of late 2024)
        # BTC should be in the ~$95k range
        assert Decimal('80000') < btc_price < Decimal('120000')
        
        # ETH should be in the ~$3.3k range
        assert Decimal('2500') < eth_price < Decimal('4500')
        
        # Prices should be realistic relative to each other
        # BTC should be roughly 25-35x ETH price
        ratio = btc_price / eth_price
        assert Decimal('20') < ratio < Decimal('40')


# ★ Insight ─────────────────────────────────────
# This demonstration shows the complete behavioral testing workflow:
# 1. Creating scenarios with realistic current market prices
# 2. Validating mathematical correctness without mocks
# 3. Testing risk limits and safety mechanisms
# 4. Comprehensive validation framework for all outcomes
# The framework ensures tests reflect real trading conditions.
# ─────────────────────────────────────────────────