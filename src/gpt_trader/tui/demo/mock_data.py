"""
Mock data generator for TUI demo mode.

Generates realistic-looking trading data that simulates a live bot
without needing real exchange connections.
"""

import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MockDataGenerator:
    """Generates realistic mock data for TUI testing."""

    symbols: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD"])
    base_prices: dict[str, float] = field(default_factory=dict)
    price_history: dict[str, list[float]] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    cycle_count: int = 0
    total_equity: float = 10000.0
    starting_equity: float = 10000.0

    # Position state
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize base prices and history."""
        if not self.base_prices:
            self.base_prices = {
                "BTC-USD": 45000.0 + random.uniform(-2000, 2000),
                "ETH-USD": 2500.0 + random.uniform(-200, 200),
                "SOL-USD": 100.0 + random.uniform(-10, 10),
            }

        for symbol in self.symbols:
            if symbol not in self.price_history:
                self.price_history[symbol] = [self.base_prices[symbol]]

        # Initialize with some starting positions for demo
        if not self.positions:
            self.positions = {
                "BTC-USD": {
                    "symbol": "BTC-USD",
                    "quantity": 0.25,
                    "side": "LONG",
                    "entry_price": self.base_prices["BTC-USD"] * 0.98,  # Entry 2% lower
                },
                "ETH-USD": {
                    "symbol": "ETH-USD",
                    "quantity": 3.5,
                    "side": "LONG",
                    "entry_price": self.base_prices["ETH-USD"] * 1.01,  # Entry 1% higher
                },
            }

    def update_prices(self) -> dict[str, str]:
        """Generate new price updates with realistic random walk."""
        prices = {}

        for symbol in self.symbols:
            current = self.base_prices[symbol]

            # Random walk: ±0.5% movement
            change_pct = random.uniform(-0.005, 0.005)
            new_price = current * (1 + change_pct)

            self.base_prices[symbol] = new_price
            self.price_history[symbol].append(new_price)

            # Keep history at max 50 items
            if len(self.price_history[symbol]) > 50:
                self.price_history[symbol] = self.price_history[symbol][-50:]

            prices[symbol] = f"{new_price:.2f}"

        return prices

    def generate_market_data(self) -> dict[str, Any]:
        """Generate market data status."""
        prices = self.update_prices()

        return {
            "last_prices": prices,
            "last_price_update": time.time(),
            "price_history": {
                symbol: [float(p) for p in history[-20:]]  # Last 20 for chart
                for symbol, history in self.price_history.items()
            },
        }

    def generate_positions(self) -> dict[str, Any]:
        """Generate position data with unrealized P&L."""
        positions = {}
        total_upnl = 0.0

        for symbol, pos in self.positions.items():
            current_price = self.base_prices[symbol]
            entry_price = pos["entry_price"]
            quantity = pos["quantity"]

            # Calculate unrealized P&L
            upnl = (current_price - entry_price) * quantity
            total_upnl += upnl

            positions[symbol] = {
                "quantity": quantity,
                "entry_price": entry_price,
                "unrealized_pnl": f"{upnl:.2f}",
                "mark_price": f"{current_price:.2f}",
                "side": pos["side"],
            }

        # Update equity
        self.total_equity = self.starting_equity + total_upnl

        return {
            "positions": positions,
            "total_unrealized_pnl": f"{total_upnl:.2f}",
            "equity": f"{self.total_equity:.2f}",
        }

    def generate_orders(self) -> list[dict[str, Any]]:
        """Generate active orders."""
        return self.orders.copy()

    def generate_trades(self) -> list[dict[str, Any]]:
        """Generate recent trade history."""
        return self.trades[-50:]  # Last 50 trades

    def generate_account_data(self) -> dict[str, Any]:
        """Generate account summary data."""
        daily_pnl = self.total_equity - self.starting_equity
        daily_pnl_pct = (daily_pnl / self.starting_equity) * 100

        balances = [
            {
                "asset": "USD",
                "total": f"{self.total_equity * 0.3:.2f}",
                "available": f"{self.total_equity * 0.25:.2f}",
                "hold": f"{self.total_equity * 0.05:.2f}",
            },
            {
                "asset": "BTC",
                "total": "0.025",
                "available": "0.020",
                "hold": "0.005",
            },
            {
                "asset": "ETH",
                "total": "0.5",
                "available": "0.4",
                "hold": "0.1",
            },
        ]

        return {
            "volume_30d": f"{random.uniform(50000, 150000):.2f}",
            "fees_30d": f"{random.uniform(100, 500):.2f}",
            "fee_tier": "Advanced Trade",
            "balances": balances,
            "daily_pnl": f"{daily_pnl:.2f}",
            "daily_pnl_pct": f"{daily_pnl_pct:.2f}",
        }

    def generate_strategy_data(self) -> dict[str, Any]:
        """Generate strategy decision data."""
        decisions = []

        for symbol in self.symbols:
            action = random.choice(["BUY", "SELL", "HOLD"])
            confidence = random.uniform(0.5, 0.95)

            decisions.append(
                {
                    "symbol": symbol,
                    "action": action,
                    "reason": f"Technical indicators favor {action.lower()}",
                    "confidence": confidence,
                    "indicators": {
                        "rsi": random.uniform(30, 70),
                        "macd": random.uniform(-50, 50),
                        "trend": random.choice(["bullish", "bearish", "neutral"]),
                    },
                    "timestamp": time.time(),
                }
            )

        return {
            "active_strategies": ["Momentum", "Mean Reversion"],
            "last_decisions": decisions,
        }

    def generate_risk_data(self) -> dict[str, Any]:
        """Generate risk management data."""
        daily_loss_pct = ((self.total_equity - self.starting_equity) / self.starting_equity) * 100

        return {
            "max_leverage": 2.0,
            "daily_loss_limit_pct": 5.0,
            "current_daily_loss_pct": abs(min(0, daily_loss_pct)),
            "reduce_only_mode": False,
            "reduce_only_reason": "",
            "active_guards": ["daily_loss_limit", "position_size_limit"],
        }

    def generate_system_data(self) -> dict[str, Any]:
        """Generate system health data."""
        return {
            "api_latency": random.uniform(50, 200),
            "connection_status": random.choice(["CONNECTED", "CONNECTED", "CONNECTED", "DEGRADED"]),
            "rate_limit_usage": f"{random.randint(10, 40)}%",
            "memory_usage": f"{random.randint(200, 400)}MB",
            "cpu_usage": f"{random.randint(5, 25)}%",
        }

    def generate_engine_status(self) -> dict[str, Any]:
        """Generate engine status data."""
        uptime = time.time() - self.start_time

        return {
            "running": True,
            "uptime": uptime,
            "cycle_count": self.cycle_count,
            "last_cycle_time": time.time(),
            "errors": [],
        }

    def simulate_trade(self, symbol: str, side: str, quantity: float) -> None:
        """Simulate executing a trade."""
        price = self.base_prices[symbol]
        trade_id = f"trade_{int(time.time() * 1000)}"
        order_id = f"order_{int(time.time() * 1000)}"

        # Add to trades
        trade = {
            "trade_id": trade_id,
            "product_id": symbol,
            "symbol": symbol,
            "side": side,
            "quantity": f"{quantity:.4f}",
            "price": f"{price:.2f}",
            "order_id": order_id,
            "time": datetime.now().isoformat() + "Z",  # ISO format with UTC marker
            "fee": f"{price * quantity * 0.006:.2f}",  # 0.6% fee
        }
        self.trades.append(trade)

        # Update positions
        if symbol in self.positions:
            existing = self.positions[symbol]
            if existing["side"] == side:
                # Add to position
                total_quantity = existing["quantity"] + quantity
                total_cost = (existing["entry_price"] * existing["quantity"]) + (price * quantity)
                new_entry = total_cost / total_quantity
                self.positions[symbol] = {
                    "quantity": total_quantity,
                    "entry_price": new_entry,
                    "side": side,
                }
            else:
                # Closing position
                if existing["quantity"] <= quantity:
                    # Full close
                    del self.positions[symbol]
                else:
                    # Partial close
                    self.positions[symbol]["quantity"] -= quantity
        else:
            # New position
            self.positions[symbol] = {"quantity": quantity, "entry_price": price, "side": side}

    def simulate_cycle(self) -> None:
        """Simulate one trading cycle."""
        self.cycle_count += 1

        # Random chance to execute a trade
        if random.random() < 0.2:  # 20% chance per cycle
            symbol = random.choice(self.symbols)
            side = random.choice(["BUY", "SELL"])
            quantity = random.uniform(0.001, 0.01)
            self.simulate_trade(symbol, side, quantity)

        # Random chance to place an order
        if random.random() < 0.1 and len(self.orders) < 5:  # 10% chance, max 5 orders
            symbol = random.choice(self.symbols)
            side = random.choice(["BUY", "SELL"])
            price = self.base_prices[symbol] * (1 + random.uniform(-0.02, 0.02))
            order_id = f"order_{int(time.time() * 1000)}"

            order_quantity = random.uniform(0.001, 0.01)
            # Simulate partial fills (30% chance)
            filled_pct = (
                random.choice([0.0, 0.0, 0.0, 0.25, 0.5, 0.75]) if random.random() < 0.3 else 0.0
            )
            filled_order_quantity = order_quantity * filled_pct

            self.orders.append(
                {
                    "order_id": order_id,
                    "product_id": symbol,  # Coinbase uses product_id
                    "symbol": symbol,
                    "side": side,
                    "size": f"{order_quantity:.6f}",
                    "quantity": f"{order_quantity:.6f}",
                    "price": f"{price:.2f}",
                    "average_filled_price": f"{price:.2f}" if filled_order_quantity > 0 else None,
                    "filled_size": f"{filled_order_quantity:.6f}",
                    "status": "OPEN" if filled_pct < 1.0 else "FILLED",
                    "order_type": "LIMIT",
                    "time_in_force": "GTC",
                    "created_time": datetime.now().isoformat() + "Z",  # Coinbase format
                    "creation_time": datetime.now().isoformat(),  # Legacy field
                }
            )

        # Random chance to cancel an order
        if self.orders and random.random() < 0.15:  # 15% chance
            cancelled = random.choice(self.orders)
            self.orders.remove(cancelled)

    def generate_full_status(self) -> dict[str, Any]:
        """Generate complete bot status."""
        self.simulate_cycle()

        return {
            "engine": self.generate_engine_status(),
            "market": self.generate_market_data(),
            "positions": self.generate_positions(),
            "orders": self.generate_orders(),
            "trades": self.generate_trades(),
            "account": self.generate_account_data(),
            "strategy": self.generate_strategy_data(),
            "risk": self.generate_risk_data(),
            "system": self.generate_system_data(),
            "heartbeat": {
                "enabled": True,
                "heartbeat_count": self.cycle_count,
                "last_heartbeat": time.time(),
                "is_healthy": True,
            },
            "healthy": True,
            "health_issues": [],
        }
