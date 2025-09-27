"""
Quote providers for paper trading engine.
Allows offline testing without network dependencies.
"""

from typing import Dict, Optional, Callable
import random


def create_static_quote_provider(prices: Dict[str, float]) -> Callable[[str], Optional[float]]:
    """
    Create a quote provider with static prices.
    
    Args:
        prices: Dictionary mapping symbols to their mid prices
        
    Returns:
        Quote provider function
        
    Example:
        >>> provider = create_static_quote_provider({"BTC-USD": 50000.0, "ETH-USD": 3000.0})
        >>> provider("BTC-USD")
        50000.0
    """
    def get_quote(symbol: str) -> Optional[float]:
        return prices.get(symbol)
    return get_quote


def create_random_walk_provider(
    base_prices: Dict[str, float], 
    volatility: float = 0.01
) -> Callable[[str], Optional[float]]:
    """
    Create a quote provider with random walk prices.
    Useful for testing with price movement.
    
    Args:
        base_prices: Initial prices for symbols
        volatility: Max percentage change per call (default 1%)
        
    Returns:
        Quote provider function that returns slightly varying prices
    """
    current_prices = base_prices.copy()
    
    def get_quote(symbol: str) -> Optional[float]:
        if symbol not in current_prices:
            return None
        
        # Random walk: price changes by up to ±volatility%
        change = random.uniform(-volatility, volatility)
        current_prices[symbol] *= (1 + change)
        return current_prices[symbol]
    
    return get_quote


def create_spread_provider(
    mid_prices: Dict[str, float],
    spread_bps: float = 10  # basis points
) -> Callable[[str], Optional[float]]:
    """
    Create a quote provider that simulates bid-ask spread.
    Returns mid price but can be extended to return bid/ask.
    
    Args:
        mid_prices: Mid prices for symbols
        spread_bps: Spread in basis points (default 10 = 0.1%)
        
    Returns:
        Quote provider function
    """
    def get_quote(symbol: str) -> Optional[float]:
        if symbol not in mid_prices:
            return None
        
        # For now return mid, but this could return bid/ask tuple
        return mid_prices[symbol]
    
    return get_quote


# Default test quote provider with common symbols
DEFAULT_TEST_PRICES = {
    "BTC-USD": 50000.0,
    "ETH-USD": 3000.0,
    "SOL-USD": 100.0,
    "ADA-USD": 0.50,
    "XRP-USD": 0.10,
    "LINK-USD": 15.0,
    "MATIC-USD": 1.0,
    "DOT-USD": 25.0,
    "UNI-USD": 20.0,
    "AVAX-USD": 35.0,
}


def create_default_test_provider() -> Callable[[str], Optional[float]]:
    """
    Create a default test quote provider with common crypto prices.
    
    Returns:
        Quote provider with static test prices
    """
    return create_static_quote_provider(DEFAULT_TEST_PRICES)