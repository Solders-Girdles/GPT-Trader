"""
Demo scenarios for testing different market conditions and bot states.

Each scenario configures the mock data generator to simulate specific
trading conditions for UI testing.
"""

from gpt_trader.tui.demo.mock_data import MockDataGenerator


def winning_day_scenario() -> MockDataGenerator:
    """
    Scenario: Profitable trading day.

    - All positions in profit
    - Positive P&L trending up
    - Active trading with successful executions
    """
    generator = MockDataGenerator(
        symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
        base_prices={
            "BTC-USD": 45000.0,
            "ETH-USD": 2500.0,
            "SOL-USD": 100.0,
        },
        starting_equity=10000.0,
        total_equity=11250.0,  # +12.5% gain
    )

    # Set profitable positions
    generator.positions = {
        "BTC-USD": {"quantity": 0.05, "entry_price": 43000.0, "side": "BUY"},
        "ETH-USD": {"quantity": 1.0, "entry_price": 2300.0, "side": "BUY"},
        "SOL-USD": {"quantity": 10.0, "entry_price": 95.0, "side": "BUY"},
    }

    # Add some successful trades
    generator.trades = [
        {
            "trade_id": "trade_001",
            "product_id": "BTC-USD",
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": "0.025",
            "price": "43000.00",
            "order_id": "order_001",
            "time": "2025-01-15T10:30:00",
            "fee": "6.45",
        },
        {
            "trade_id": "trade_002",
            "product_id": "ETH-USD",
            "symbol": "ETH-USD",
            "side": "BUY",
            "quantity": "0.5",
            "price": "2300.00",
            "order_id": "order_002",
            "time": "2025-01-15T11:15:00",
            "fee": "3.45",
        },
    ]

    return generator


def losing_day_scenario() -> MockDataGenerator:
    """
    Scenario: Unprofitable trading day.

    - Positions underwater
    - Negative P&L
    - Risk limits being approached
    """
    generator = MockDataGenerator(
        symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
        base_prices={
            "BTC-USD": 42000.0,
            "ETH-USD": 2200.0,
            "SOL-USD": 88.0,
        },
        starting_equity=10000.0,
        total_equity=9200.0,  # -8% loss
    )

    # Set losing positions
    generator.positions = {
        "BTC-USD": {"quantity": 0.1, "entry_price": 45000.0, "side": "BUY"},
        "ETH-USD": {"quantity": 2.0, "entry_price": 2500.0, "side": "BUY"},
    }

    return generator


def high_volatility_scenario() -> MockDataGenerator:
    """
    Scenario: High volatility market.

    - Large price swings
    - Rapid position changes
    - Many active orders
    """
    generator = MockDataGenerator(
        symbols=["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD"],
        base_prices={
            "BTC-USD": 45000.0,
            "ETH-USD": 2500.0,
            "SOL-USD": 100.0,
            "AVAX-USD": 35.0,
            "LINK-USD": 15.0,
        },
    )

    # Multiple positions
    generator.positions = {
        "BTC-USD": {"quantity": 0.05, "entry_price": 44500.0, "side": "BUY"},
        "ETH-USD": {"quantity": 1.5, "entry_price": 2450.0, "side": "BUY"},
        "SOL-USD": {"quantity": 20.0, "entry_price": 102.0, "side": "SELL"},
        "AVAX-USD": {"quantity": 50.0, "entry_price": 36.0, "side": "SELL"},
    }

    # Many active orders
    generator.orders = [
        {
            "order_id": "order_101",
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": "0.025",
            "price": "44800.00",
            "status": "OPEN",
            "order_type": "LIMIT",
            "time_in_force": "GTC",
            "creation_time": "2025-01-15T12:00:00",
        },
        {
            "order_id": "order_102",
            "symbol": "ETH-USD",
            "side": "SELL",
            "quantity": "0.5",
            "price": "2520.00",
            "status": "OPEN",
            "order_type": "LIMIT",
            "time_in_force": "GTC",
            "creation_time": "2025-01-15T12:05:00",
        },
        {
            "order_id": "order_103",
            "symbol": "LINK-USD",
            "side": "BUY",
            "quantity": "100",
            "price": "14.80",
            "status": "OPEN",
            "order_type": "LIMIT",
            "time_in_force": "GTC",
            "creation_time": "2025-01-15T12:10:00",
        },
    ]

    return generator


def quiet_market_scenario() -> MockDataGenerator:
    """
    Scenario: Quiet market conditions.

    - No open positions
    - Few trades
    - Low volatility
    - Waiting for opportunities
    """
    generator = MockDataGenerator(
        symbols=["BTC-USD", "ETH-USD"],
        base_prices={
            "BTC-USD": 45000.0,
            "ETH-USD": 2500.0,
        },
    )

    # No positions
    generator.positions = {}

    # No active orders
    generator.orders = []

    # Just a couple historical trades
    generator.trades = [
        {
            "trade_id": "trade_old_001",
            "product_id": "BTC-USD",
            "symbol": "BTC-USD",
            "side": "SELL",
            "quantity": "0.01",
            "price": "44900.00",
            "order_id": "order_old_001",
            "time": "2025-01-15T08:00:00",
            "fee": "2.69",
        }
    ]

    return generator


def risk_limit_scenario() -> MockDataGenerator:
    """
    Scenario: Approaching risk limits.

    - Large drawdown approaching daily loss limit
    - Risk manager in reduce-only mode
    - Position sizing restricted
    """
    generator = MockDataGenerator(
        symbols=["BTC-USD", "ETH-USD"],
        base_prices={
            "BTC-USD": 42000.0,
            "ETH-USD": 2200.0,
        },
        starting_equity=10000.0,
        total_equity=9550.0,  # -4.5% (approaching -5% limit)
    )

    # Losing positions
    generator.positions = {
        "BTC-USD": {"quantity": 0.1, "entry_price": 45000.0, "side": "BUY"},
    }

    return generator


def mixed_positions_scenario() -> MockDataGenerator:
    """
    Scenario: Mixed long and short positions.

    - Some positions winning, some losing
    - Both long and short sides active
    - Neutral overall P&L
    """
    generator = MockDataGenerator(
        symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
        base_prices={
            "BTC-USD": 45000.0,
            "ETH-USD": 2500.0,
            "SOL-USD": 100.0,
        },
        starting_equity=10000.0,
        total_equity=10100.0,  # +1% slight gain
    )

    # Mixed positions
    generator.positions = {
        "BTC-USD": {"quantity": 0.05, "entry_price": 43000.0, "side": "BUY"},  # Winning
        "ETH-USD": {"quantity": 1.0, "entry_price": 2600.0, "side": "BUY"},  # Losing
        "SOL-USD": {"quantity": 10.0, "entry_price": 105.0, "side": "SELL"},  # Winning
    }

    return generator


# Scenario registry for easy access
SCENARIOS = {
    "winning": winning_day_scenario,
    "losing": losing_day_scenario,
    "volatile": high_volatility_scenario,
    "quiet": quiet_market_scenario,
    "risk_limit": risk_limit_scenario,
    "mixed": mixed_positions_scenario,
}


def get_scenario(name: str = "mixed") -> MockDataGenerator:
    """Get a demo scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {', '.join(SCENARIOS.keys())}")
    return SCENARIOS[name]()
