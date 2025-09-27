"""
Signal generation for backtesting.
"""

import pandas as pd
from .strategies import create_local_strategy
from .validation import validate_signals


def generate_signals(
    strategy_name: str,
    data: pd.DataFrame,
    **strategy_params
) -> pd.Series:
    """
    Generate trading signals using specified strategy.
    
    Args:
        strategy_name: Name of strategy to use
        data: Market data
        **strategy_params: Additional strategy parameters
        
    Returns:
        Series of signals (1=buy, -1=sell, 0=hold)
    """
    # Create strategy instance (local - no external dependencies!)
    strategy = create_local_strategy(strategy_name, **strategy_params)
    
    # Check if we have enough data
    required_periods = strategy.get_required_periods()
    if len(data) < required_periods:
        raise ValueError(
            f"Insufficient data: need {required_periods} periods, "
            f"got {len(data)}"
        )
    
    # Generate signals
    signals = strategy.run(data)
    
    # Validate signals
    if not validate_signals(signals):
        raise ValueError(f"Invalid signals from {strategy_name}")
    
    return signals