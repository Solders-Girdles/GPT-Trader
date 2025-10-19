"""
Trade scenario builders for behavioral testing.

Creates complete trading scenarios with known outcomes for validation.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass
class TradeExecution:
    """Represents a single trade execution."""

    timestamp: datetime
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    price: Decimal
    fees: Decimal = Decimal("0")
    is_reduce: bool = False


@dataclass
class TradeScenario:
    """Complete trading scenario with expected outcomes."""

    name: str
    symbol: str
    initial_capital: Decimal
    trades: list[TradeExecution]
    expected_pnl: Decimal
    expected_fees: Decimal
    expected_final_position: Decimal
    funding_payments: Decimal = Decimal("0")
    description: str = ""

    @property
    def expected_net_pnl(self) -> Decimal:
        """Net P&L after fees and funding."""
        return self.expected_pnl - self.expected_fees + self.funding_payments


def create_long_profit_scenario() -> TradeScenario:
    """
    Create a simple long position profit scenario.

    Scenario:
    - Buy 1 BTC at $50,000
    - Sell 1 BTC at $51,000
    - Fees: 0.1% per trade
    - Expected gross P&L: $1,000
    - Expected fees: $101 (50 + 51)
    - Expected net P&L: $899
    """
    return TradeScenario(
        name="long_profit",
        symbol="BTC-PERP",
        initial_capital=Decimal("100000"),
        trades=[
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                side="buy",
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
                fees=Decimal("50"),  # 0.1% of 50000
            ),
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 11, 0, 0),
                side="sell",
                quantity=Decimal("1.0"),
                price=Decimal("51000"),
                fees=Decimal("51"),  # 0.1% of 51000
                is_reduce=True,
            ),
        ],
        expected_pnl=Decimal("1000"),
        expected_fees=Decimal("101"),
        expected_final_position=Decimal("0"),
        description="Simple long position with profit",
    )


def create_short_loss_scenario() -> TradeScenario:
    """
    Create a short position loss scenario.

    Scenario:
    - Sell 2 ETH at $3,000 (short)
    - Buy 2 ETH at $3,100 (cover)
    - Fees: 0.1% per trade
    - Expected gross P&L: -$200
    - Expected fees: $62 (60 + 62)
    - Expected net P&L: -$262
    """
    return TradeScenario(
        name="short_loss",
        symbol="ETH-PERP",
        initial_capital=Decimal("50000"),
        trades=[
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                side="sell",
                quantity=Decimal("2.0"),
                price=Decimal("3000"),
                fees=Decimal("6"),  # 0.1% of 6000
            ),
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 11, 0, 0),
                side="buy",
                quantity=Decimal("2.0"),
                price=Decimal("3100"),
                fees=Decimal("6.2"),  # 0.1% of 6200
                is_reduce=True,
            ),
        ],
        expected_pnl=Decimal("-200"),
        expected_fees=Decimal("12.2"),
        expected_final_position=Decimal("0"),
        description="Short position with loss",
    )


def create_funding_payment_scenario() -> TradeScenario:
    """
    Create a scenario with funding payments.

    Scenario:
    - Long 1 BTC at $50,000
    - Hold through 3 funding periods at 0.01% each
    - Sell at $50,500
    - Funding paid: 3 * (1 * 50000 * 0.0001) = $15
    - Gross P&L: $500
    - Net after funding: $485
    """
    return TradeScenario(
        name="funding_payment",
        symbol="BTC-PERP",
        initial_capital=Decimal("100000"),
        trades=[
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 0, 0, 0),
                side="buy",
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
                fees=Decimal("50"),
            ),
            TradeExecution(
                timestamp=datetime(2024, 1, 2, 0, 0, 0),  # Next day
                side="sell",
                quantity=Decimal("1.0"),
                price=Decimal("50500"),
                fees=Decimal("50.5"),
                is_reduce=True,
            ),
        ],
        expected_pnl=Decimal("500"),
        expected_fees=Decimal("100.5"),
        expected_final_position=Decimal("0"),
        funding_payments=Decimal("-15"),  # Negative because long pays
        description="Long position with funding payments",
    )


def create_stop_loss_scenario() -> TradeScenario:
    """
    Create a stop-loss execution scenario.

    Scenario:
    - Buy 0.5 BTC at $50,000
    - Stop-loss triggers at $49,000
    - Expected loss: $500
    - Fees: $74.5 (25 + 24.5)
    """
    return TradeScenario(
        name="stop_loss",
        symbol="BTC-PERP",
        initial_capital=Decimal("100000"),
        trades=[
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                side="buy",
                quantity=Decimal("0.5"),
                price=Decimal("50000"),
                fees=Decimal("25"),
            ),
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 10, 30, 0),
                side="sell",
                quantity=Decimal("0.5"),
                price=Decimal("49000"),  # Stop triggered
                fees=Decimal("24.5"),
                is_reduce=True,
            ),
        ],
        expected_pnl=Decimal("-500"),
        expected_fees=Decimal("49.5"),
        expected_final_position=Decimal("0"),
        description="Stop-loss triggered scenario",
    )


def create_partial_close_scenario() -> TradeScenario:
    """
    Create a partial position close scenario.

    Scenario:
    - Buy 3 BTC at $50,000
    - Sell 1 BTC at $51,000 (partial close)
    - Sell 2 BTC at $52,000 (full close)
    - Expected P&L: 1*1000 + 2*2000 = $5000
    """
    return TradeScenario(
        name="partial_close",
        symbol="BTC-PERP",
        initial_capital=Decimal("200000"),
        trades=[
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                side="buy",
                quantity=Decimal("3.0"),
                price=Decimal("50000"),
                fees=Decimal("150"),
            ),
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 11, 0, 0),
                side="sell",
                quantity=Decimal("1.0"),
                price=Decimal("51000"),
                fees=Decimal("51"),
                is_reduce=True,
            ),
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                side="sell",
                quantity=Decimal("2.0"),
                price=Decimal("52000"),
                fees=Decimal("104"),
                is_reduce=True,
            ),
        ],
        expected_pnl=Decimal("5000"),
        expected_fees=Decimal("305"),
        expected_final_position=Decimal("0"),
        description="Partial position closing with scaling out",
    )


def create_position_flip_scenario() -> TradeScenario:
    """
    Create a position flip scenario (long to short).

    Scenario:
    - Buy 1 BTC at $50,000 (long)
    - Sell 2 BTC at $51,000 (close long, open short)
    - Buy 1 BTC at $50,500 (close short)
    - P&L: Long leg: $1000, Short leg: $500
    """
    return TradeScenario(
        name="position_flip",
        symbol="BTC-PERP",
        initial_capital=Decimal("100000"),
        trades=[
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                side="buy",
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
                fees=Decimal("50"),
            ),
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 11, 0, 0),
                side="sell",
                quantity=Decimal("2.0"),  # Close 1, open 1 short
                price=Decimal("51000"),
                fees=Decimal("102"),
            ),
            TradeExecution(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                side="buy",
                quantity=Decimal("1.0"),
                price=Decimal("50500"),
                fees=Decimal("50.5"),
                is_reduce=True,
            ),
        ],
        expected_pnl=Decimal("1500"),  # 1000 from long + 500 from short
        expected_fees=Decimal("202.5"),
        expected_final_position=Decimal("0"),
        description="Position flip from long to short",
    )
