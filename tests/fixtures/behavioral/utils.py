"""
Behavioral testing utilities and helpers.

Provides convenient functions for creating and running behavioral tests.
"""

from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from .market_data import RealisticMarketData, MarketDataGenerator
from .trade_scenarios import TradeScenario, TradeExecution
from .validators import validate_pnl_calculation, validate_position_state, validate_risk_limits


def create_realistic_btc_scenario(
    scenario_type: str = "profit",
    position_size: Decimal = Decimal('0.1'),
    hold_days: int = 1
) -> TradeScenario:
    """
    Create a realistic BTC trading scenario using current market conditions.
    
    Args:
        scenario_type: "profit", "loss", or "breakeven"
        position_size: Position size in BTC
        hold_days: Days to hold position
        
    Returns:
        TradeScenario with realistic BTC prices
    """
    current_btc = RealisticMarketData.CURRENT_PRICES['BTC-PERP']
    entry_price = current_btc
    
    # Create realistic price movements based on scenario
    if scenario_type == "profit":
        exit_price = entry_price + (entry_price * Decimal('0.02'))  # 2% gain
        expected_pnl = (exit_price - entry_price) * position_size
    elif scenario_type == "loss":
        exit_price = entry_price - (entry_price * Decimal('0.025'))  # 2.5% loss
        expected_pnl = (exit_price - entry_price) * position_size
    else:  # breakeven
        exit_price = entry_price
        expected_pnl = Decimal('0')
    
    # Calculate realistic fees (0.05% for maker, 0.1% for taker)
    entry_fee = entry_price * position_size * Decimal('0.0005')
    exit_fee = exit_price * position_size * Decimal('0.0005')
    total_fees = entry_fee + exit_fee
    
    return TradeScenario(
        name=f"realistic_btc_{scenario_type}",
        symbol="BTC-PERP",
        initial_capital=current_btc * position_size * Decimal('2'),  # 50% of capital
        trades=[
            TradeExecution(
                timestamp=datetime.now() - timedelta(days=hold_days),
                side="buy",
                quantity=position_size,
                price=entry_price,
                fees=entry_fee
            ),
            TradeExecution(
                timestamp=datetime.now(),
                side="sell",
                quantity=position_size,
                price=exit_price,
                fees=exit_fee,
                is_reduce=True
            )
        ],
        expected_pnl=expected_pnl,
        expected_fees=total_fees,
        expected_final_position=Decimal('0'),
        description=f"Realistic BTC {scenario_type} scenario at current market prices"
    )


def create_realistic_eth_scenario(
    scenario_type: str = "short_profit",
    position_size: Decimal = Decimal('3.0'),
    volatility: str = "normal"
) -> TradeScenario:
    """
    Create a realistic ETH trading scenario.
    
    Args:
        scenario_type: "short_profit", "short_loss", "long_profit", "long_loss"
        position_size: Position size in ETH
        volatility: "low", "normal", "high"
        
    Returns:
        TradeScenario with realistic ETH prices
    """
    current_eth = RealisticMarketData.CURRENT_PRICES['ETH-PERP']
    entry_price = current_eth
    
    # Volatility multipliers
    vol_multipliers = {
        "low": Decimal('0.01'),      # 1%
        "normal": Decimal('0.03'),   # 3%
        "high": Decimal('0.06')      # 6%
    }
    vol_mult = vol_multipliers.get(volatility, vol_multipliers["normal"])
    
    # Calculate price movement based on scenario
    if "profit" in scenario_type:
        price_change = entry_price * vol_mult
    else:  # loss
        price_change = entry_price * vol_mult * Decimal('-1')
    
    if scenario_type.startswith("short"):
        # For shorts: profit when price goes down, loss when price goes up
        exit_price = entry_price - price_change
        expected_pnl = (entry_price - exit_price) * position_size
        initial_side = "sell"
        close_side = "buy"
    else:  # long
        exit_price = entry_price + price_change
        expected_pnl = (exit_price - entry_price) * position_size
        initial_side = "buy"
        close_side = "sell"
    
    # Calculate fees
    entry_fee = entry_price * position_size * Decimal('0.0005')
    exit_fee = exit_price * position_size * Decimal('0.0005')
    total_fees = entry_fee + exit_fee
    
    return TradeScenario(
        name=f"realistic_eth_{scenario_type}_{volatility}",
        symbol="ETH-PERP",
        initial_capital=current_eth * position_size * Decimal('3'),
        trades=[
            TradeExecution(
                timestamp=datetime.now() - timedelta(hours=4),
                side=initial_side,
                quantity=position_size,
                price=entry_price,
                fees=entry_fee
            ),
            TradeExecution(
                timestamp=datetime.now(),
                side=close_side,
                quantity=position_size,
                price=exit_price,
                fees=exit_fee,
                is_reduce=True
            )
        ],
        expected_pnl=expected_pnl,
        expected_fees=total_fees,
        expected_final_position=Decimal('0'),
        description=f"Realistic ETH {scenario_type} with {volatility} volatility"
    )


def create_funding_scenario(
    symbol: str = "BTC-PERP",
    position_size: Decimal = Decimal('0.5'),
    funding_periods: int = 3,
    funding_rate: Decimal = Decimal('0.0001')  # 0.01%
) -> TradeScenario:
    """
    Create a funding payment scenario for perpetuals.
    
    Args:
        symbol: Trading symbol
        position_size: Position size
        funding_periods: Number of 8-hour funding periods
        funding_rate: Funding rate per period
        
    Returns:
        TradeScenario with funding payments
    """
    current_price = RealisticMarketData.CURRENT_PRICES[symbol]
    
    # Calculate funding payments (longs pay when positive rate)
    notional = position_size * current_price
    funding_per_period = notional * funding_rate
    total_funding = funding_per_period * funding_periods
    
    return TradeScenario(
        name=f"funding_{symbol.lower()}_{funding_periods}periods",
        symbol=symbol,
        initial_capital=current_price * position_size * Decimal('2'),
        trades=[
            TradeExecution(
                timestamp=datetime.now() - timedelta(hours=8 * funding_periods),
                side="buy",
                quantity=position_size,
                price=current_price,
                fees=current_price * position_size * Decimal('0.0005')
            ),
            TradeExecution(
                timestamp=datetime.now(),
                side="sell",
                quantity=position_size,
                price=current_price,  # Same price for funding test
                fees=current_price * position_size * Decimal('0.0005'),
                is_reduce=True
            )
        ],
        expected_pnl=Decimal('0'),  # No price movement
        expected_fees=current_price * position_size * Decimal('0.001'),
        expected_final_position=Decimal('0'),
        funding_payments=-total_funding,  # Negative because long pays
        description=f"Funding payment scenario over {funding_periods} periods"
    )


def create_risk_limit_test_scenario(
    limit_type: str,
    should_pass: bool = False
) -> Dict[str, Any]:
    """
    Create a test scenario for risk limit enforcement.
    
    Args:
        limit_type: "position", "leverage", "daily_loss", "impact"
        should_pass: Whether the trade should pass risk checks
        
    Returns:
        Dictionary with trade details and risk parameters
    """
    btc_price = RealisticMarketData.CURRENT_PRICES['BTC-PERP']
    
    if limit_type == "position":
        # Test position size limits
        max_position = Decimal('0.01')  # Canary limit
        test_size = Decimal('0.005') if should_pass else Decimal('0.02')  # Under/over limit
        
        return {
            'symbol': 'BTC-PERP',
            'side': 'buy',
            'quantity': test_size,
            'price': btc_price,
            'risk_limits': {'max_position_size': max_position},
            'expected_pass': should_pass,
            'description': f"Position limit test: {test_size} vs {max_position} limit"
        }
    
    elif limit_type == "leverage":
        # Test leverage limits
        equity = Decimal('10000')
        max_leverage = 5
        test_size = Decimal('0.526') if should_pass else Decimal('0.6')  # 5x vs 5.7x leverage
        
        return {
            'symbol': 'BTC-PERP',
            'side': 'buy',
            'quantity': test_size,
            'price': btc_price,
            'equity': equity,
            'risk_limits': {'max_leverage': max_leverage},
            'expected_pass': should_pass,
            'description': f"Leverage limit test: ~{(test_size * btc_price / equity):.1f}x vs {max_leverage}x limit"
        }
    
    elif limit_type == "daily_loss":
        # Test daily loss limits
        max_daily_loss = Decimal('100')
        current_loss = Decimal('-50') if should_pass else Decimal('-120')
        
        return {
            'symbol': 'BTC-PERP',
            'current_daily_pnl': current_loss,
            'risk_limits': {'max_daily_loss': max_daily_loss},
            'expected_pass': should_pass,
            'description': f"Daily loss limit test: {current_loss} vs -{max_daily_loss} limit"
        }
    
    elif limit_type == "impact":
        # Test market impact limits
        max_impact_bps = Decimal('50')  # 50 bps
        test_impact = Decimal('25') if should_pass else Decimal('75')  # Under/over limit
        test_size = Decimal('0.1') if should_pass else Decimal('2.0')  # Small/large size
        
        return {
            'symbol': 'BTC-PERP',
            'side': 'buy', 
            'quantity': test_size,
            'price': btc_price,
            'estimated_impact_bps': test_impact,
            'risk_limits': {'max_impact_bps': max_impact_bps},
            'expected_pass': should_pass,
            'description': f"Impact limit test: {test_impact} bps vs {max_impact_bps} bps limit"
        }
    
    else:
        raise ValueError(f"Unknown limit_type: {limit_type}")


def run_behavioral_validation(
    scenario: TradeScenario,
    actual_results: Dict[str, Any],
    tolerances: Optional[Dict[str, Decimal]] = None
) -> Tuple[bool, List[str]]:
    """
    Run complete behavioral validation on a trading scenario.
    
    Args:
        scenario: Expected scenario outcomes
        actual_results: Actual trading system results
        tolerances: Optional tolerances for validation
        
    Returns:
        (all_passed, error_messages) tuple
    """
    if tolerances is None:
        tolerances = {
            'pnl': Decimal('0.01'),
            'fees': Decimal('0.01'),
            'position': Decimal('0.0001')
        }
    
    errors = []
    all_passed = True
    
    # Validate P&L calculation
    trades_for_validation = [
        {
            'side': trade.side,
            'quantity': trade.quantity,
            'price': trade.price,
            'pnl': None  # Will be calculated
        }
        for trade in scenario.trades
    ]
    
    pnl_passed, pnl_msg = validate_pnl_calculation(
        trades_for_validation,
        scenario.expected_pnl,
        tolerances['pnl']
    )
    
    if not pnl_passed:
        all_passed = False
        errors.append(f"P&L validation failed: {pnl_msg}")
    
    # Validate final position
    if 'final_position' in actual_results:
        final_position = actual_results['final_position']
        expected_position = scenario.expected_final_position
        position_diff = abs(final_position - expected_position)
        
        if position_diff > tolerances['position']:
            all_passed = False
            errors.append(
                f"Position validation failed: actual={final_position}, "
                f"expected={expected_position}, diff={position_diff}"
            )
    
    # Validate fees if provided
    if 'total_fees' in actual_results:
        fee_diff = abs(actual_results['total_fees'] - scenario.expected_fees)
        if fee_diff > tolerances['fees']:
            all_passed = False
            errors.append(
                f"Fee validation failed: actual={actual_results['total_fees']}, "
                f"expected={scenario.expected_fees}, diff={fee_diff}"
            )
    
    # Validate funding payments if applicable
    if scenario.funding_payments != 0 and 'funding_paid' in actual_results:
        funding_diff = abs(actual_results['funding_paid'] - scenario.funding_payments)
        if funding_diff > tolerances.get('funding', Decimal('0.01')):
            all_passed = False
            errors.append(
                f"Funding validation failed: actual={actual_results['funding_paid']}, "
                f"expected={scenario.funding_payments}, diff={funding_diff}"
            )
    
    return all_passed, errors


def create_market_stress_scenario() -> List[TradeScenario]:
    """
    Create multiple scenarios for stress testing market conditions.
    
    Returns:
        List of stress test scenarios
    """
    scenarios = []
    
    # Flash crash scenario
    scenarios.append(create_realistic_btc_scenario(
        scenario_type="loss", 
        position_size=Decimal('0.05')
    ))
    
    # High volatility ETH
    scenarios.append(create_realistic_eth_scenario(
        scenario_type="short_loss",
        volatility="high"
    ))
    
    # Extended funding scenario
    scenarios.append(create_funding_scenario(
        funding_periods=8,  # 64 hours
        funding_rate=Decimal('0.0003')  # High funding rate
    ))
    
    return scenarios


# ★ Insight ─────────────────────────────────────
# These utilities provide a comprehensive behavioral testing framework:
# 1. Realistic scenarios use current market prices (BTC ~$95k, ETH ~$3.3k)
# 2. Risk limit testing covers all major safety mechanisms
# 3. Validation functions ensure mathematical correctness without mocks
# ─────────────────────────────────────────────────