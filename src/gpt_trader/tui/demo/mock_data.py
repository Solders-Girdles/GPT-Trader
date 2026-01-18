"""
Mock data generator for TUI demo mode.

Generates realistic-looking trading data that simulates a live bot
without needing real exchange connections.

Supports optional seeding for reproducible demo/test scenarios.
"""

import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MockDataGenerator:
    """Generates realistic mock data for TUI testing.

    Attributes:
        seed: Optional random seed for reproducible output. When provided,
            the same seed produces identical data sequences across runs.
    """

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

    # Optional seed for reproducibility
    seed: int | None = None

    # Internal RNG instance (set in __post_init__)
    _rng: random.Random = field(default_factory=random.Random, repr=False)

    def __post_init__(self) -> None:
        """Initialize RNG, base prices and history."""
        # Create seeded or unseeded RNG
        self._rng = random.Random(self.seed)

        if not self.base_prices:
            self.base_prices = {
                "BTC-USD": 45000.0 + self._rng.uniform(-2000, 2000),
                "ETH-USD": 2500.0 + self._rng.uniform(-200, 200),
                "SOL-USD": 100.0 + self._rng.uniform(-10, 10),
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

        # Initialize with some sample orders showing different fill states
        if not self.orders:
            btc_price = self.base_prices["BTC-USD"]
            eth_price = self.base_prices["ETH-USD"]
            self.orders = [
                {
                    "order_id": "demo_order_1",
                    "product_id": "BTC-USD",
                    "symbol": "BTC-USD",
                    "side": "BUY",
                    "size": "0.05",
                    "quantity": "0.05",
                    "price": f"{btc_price * 0.99:.2f}",  # Limit below market
                    "filled_size": "0.02",  # 40% filled
                    "filled_quantity": "0.02",
                    "average_filled_price": f"{btc_price * 0.985:.2f}",
                    "status": "OPEN",
                    "order_type": "LIMIT",
                    "time_in_force": "GTC",
                    "created_time": datetime.now().isoformat() + "Z",
                    "creation_time": time.time() - 45,  # 45 seconds ago
                },
                {
                    "order_id": "demo_order_2",
                    "product_id": "ETH-USD",
                    "symbol": "ETH-USD",
                    "side": "SELL",
                    "size": "1.0",
                    "quantity": "1.0",
                    "price": f"{eth_price * 1.02:.2f}",  # Limit above market
                    "filled_size": "0",  # Not filled yet
                    "filled_quantity": "0",
                    "average_filled_price": None,
                    "status": "OPEN",
                    "order_type": "LIMIT",
                    "time_in_force": "GTC",
                    "created_time": datetime.now().isoformat() + "Z",
                    "creation_time": time.time() - 15,  # 15 seconds ago
                },
            ]

    def update_prices(self) -> dict[str, str]:
        """Generate new price updates with realistic random walk."""
        prices = {}

        for symbol in self.symbols:
            current = self.base_prices[symbol]

            # Random walk: ±0.5% movement
            change_pct = self._rng.uniform(-0.005, 0.005)
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

        # Calculate realized P&L from trades (sum of profitable closes)
        # For demo, simulate realized P&L based on trade count
        total_realized_pnl = len(self.trades) * self._rng.uniform(-5, 15)

        return {
            "positions": positions,
            "total_unrealized_pnl": f"{total_upnl:.2f}",
            "equity": f"{self.total_equity:.2f}",
            "total_realized_pnl": f"{total_realized_pnl:.2f}",
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
            "volume_30d": f"{self._rng.uniform(50000, 150000):.2f}",
            "fees_30d": f"{self._rng.uniform(100, 500):.2f}",
            "fee_tier": "Advanced Trade",
            "balances": balances,
            "daily_pnl": f"{daily_pnl:.2f}",
            "daily_pnl_pct": f"{daily_pnl_pct:.2f}",
        }

    def generate_strategy_data(self) -> dict[str, Any]:
        """Generate strategy decision data."""
        decisions = []

        # Possible guards that can block decisions
        blocking_guards = [
            "DailyLossGuard",
            "VolatilityGuard",
            "PositionSizeGuard",
            "MaxDrawdownGuard",
        ]

        for symbol in self.symbols:
            action = self._rng.choice(["BUY", "SELL", "HOLD"])
            confidence = self._rng.uniform(0.5, 0.95)

            # Occasionally block BUY/SELL decisions (20% chance)
            blocked_by = ""
            if action in ("BUY", "SELL") and self._rng.random() < 0.2:
                blocked_by = self._rng.choice(blocking_guards)

            # Generate indicator values
            rsi = self._rng.uniform(30, 70)
            macd = self._rng.uniform(-50, 50)
            trend = self._rng.choice(["bullish", "bearish", "neutral"])

            # Generate contributions based on action and indicator values
            # RSI: < 30 is bullish (oversold), > 70 is bearish (overbought)
            rsi_contribution = (50 - rsi) / 50  # -0.4 to +0.4
            if action == "BUY":
                rsi_contribution = (
                    abs(rsi_contribution) if rsi < 50 else -abs(rsi_contribution) * 0.5
                )
            elif action == "SELL":
                rsi_contribution = (
                    -abs(rsi_contribution) if rsi > 50 else abs(rsi_contribution) * 0.5
                )

            # MACD: positive is bullish, negative is bearish
            macd_contribution = macd / 100  # -0.5 to +0.5

            # Trend: direct mapping
            trend_contribution = 0.3 if trend == "bullish" else -0.3 if trend == "bearish" else 0.0

            contributions = [
                {"name": "RSI", "value": round(rsi, 1), "contribution": round(rsi_contribution, 2)},
                {
                    "name": "MACD",
                    "value": round(macd, 1),
                    "contribution": round(macd_contribution, 2),
                },
                {"name": "Trend", "value": trend, "contribution": round(trend_contribution, 2)},
            ]

            decisions.append(
                {
                    "symbol": symbol,
                    "action": action,
                    "reason": f"Technical indicators favor {action.lower()}",
                    "confidence": confidence,
                    "indicators": {
                        "rsi": rsi,
                        "macd": macd,
                        "trend": trend,
                    },
                    "contributions": contributions,
                    "timestamp": time.time(),
                    "blocked_by": blocked_by,
                }
            )

        # Generate deterministic performance metrics
        # Use RNG for slight variation but keep values realistic
        live_win_rate = 0.58 + self._rng.uniform(-0.05, 0.05)
        live_profit_factor = 1.65 + self._rng.uniform(-0.1, 0.1)
        live_total_trades = 45 + int(self._rng.uniform(-5, 5))
        live_winning = int(live_total_trades * live_win_rate)
        live_losing = live_total_trades - live_winning

        backtest_win_rate = 0.56 + self._rng.uniform(-0.03, 0.03)
        backtest_profit_factor = 1.42 + self._rng.uniform(-0.08, 0.08)
        backtest_total_trades = 120 + int(self._rng.uniform(-10, 10))
        backtest_winning = int(backtest_total_trades * backtest_win_rate)
        backtest_losing = backtest_total_trades - backtest_winning

        return {
            "active_strategies": ["Momentum", "Mean Reversion"],
            "last_decisions": decisions,
            "performance": {
                "win_rate": live_win_rate,
                "profit_factor": live_profit_factor,
                "total_return": 0.082,  # 8.2%
                "max_drawdown": -0.041,  # -4.1%
                "total_trades": live_total_trades,
                "winning_trades": live_winning,
                "losing_trades": live_losing,
                "sharpe_ratio": 1.05,
            },
            "backtest_performance": {
                "win_rate": backtest_win_rate,
                "profit_factor": backtest_profit_factor,
                "total_return": 0.124,  # 12.4%
                "max_drawdown": -0.062,  # -6.2%
                "total_trades": backtest_total_trades,
                "winning_trades": backtest_winning,
                "losing_trades": backtest_losing,
            },
            # Strategy indicator parameters for live config display
            "parameters": {
                # RSI config
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                # MA config (Trend signal)
                "ma_fast_period": 5,
                "ma_slow_period": 20,
                "ma_type": "SMA",
                # Z-Score / Mean Reversion config
                "zscore_lookback": 20,
                "zscore_entry_threshold": 2.0,
                "zscore_exit_threshold": 0.5,
                # VWAP config
                "vwap_deviation_threshold": 0.01,
                # Spread config
                "spread_tight_bps": 5.0,
                "spread_normal_bps": 15.0,
                "spread_wide_bps": 30.0,
                # Orderbook config
                "orderbook_levels": 5,
                "orderbook_imbalance_threshold": 0.2,
            },
        }

    def generate_risk_data(self) -> dict[str, Any]:
        """Generate risk management data with enhanced guard info."""
        daily_loss_pct = ((self.total_equity - self.starting_equity) / self.starting_equity) * 100
        now = time.time()

        # Enhanced guard definitions with severity and timestamps
        available_guards = [
            {
                "name": "DailyLossGuard",
                "severity": "CRITICAL",
                "description": "Blocks trades when daily loss exceeds limit",
            },
            {
                "name": "MaxDrawdownGuard",
                "severity": "CRITICAL",
                "description": "Blocks trades at max drawdown threshold",
            },
            {
                "name": "VolatilityGuard",
                "severity": "HIGH",
                "description": "Blocks trades during extreme volatility",
            },
            {
                "name": "PositionSizeGuard",
                "severity": "MEDIUM",
                "description": "Limits position size per trade",
            },
            {
                "name": "ExposureGuard",
                "severity": "HIGH",
                "description": "Limits total market exposure",
            },
            {
                "name": "RateLimitGuard",
                "severity": "LOW",
                "description": "Throttles trade frequency",
            },
        ]

        # Select 2-4 random guards as "active"
        num_active = self._rng.randint(2, 4)
        active_guard_defs = self._rng.sample(available_guards, num_active)

        # Build enhanced guards with timestamps
        guards = []
        for guard_def in active_guard_defs:
            # Simulate last_triggered (some recent, some never)
            if self._rng.random() < 0.6:  # 60% chance was triggered
                # Random time in past: 10s to 2h ago
                last_triggered = now - self._rng.uniform(10, 7200)
                triggered_count = self._rng.randint(1, 15)
            else:
                last_triggered = 0.0
                triggered_count = 0

            guards.append(
                {
                    "name": guard_def["name"],
                    "severity": guard_def["severity"],
                    "last_triggered": last_triggered,
                    "triggered_count": triggered_count,
                    "description": guard_def["description"],
                }
            )

        return {
            "max_leverage": 2.0,
            "daily_loss_limit_pct": 5.0,
            "current_daily_loss_pct": abs(min(0, daily_loss_pct)),
            "reduce_only_mode": False,
            "reduce_only_reason": "",
            "guards": guards,
        }

    def generate_system_data(self) -> dict[str, Any]:
        """Generate system health data."""
        return {
            "api_latency": self._rng.uniform(50, 200),
            "connection_status": self._rng.choice(
                ["CONNECTED", "CONNECTED", "CONNECTED", "DEGRADED"]
            ),
            "rate_limit_usage": f"{self._rng.randint(10, 40)}%",
            "memory_usage": f"{self._rng.randint(200, 400)}MB",
            "cpu_usage": f"{self._rng.randint(5, 25)}%",
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
        if self._rng.random() < 0.2:  # 20% chance per cycle
            symbol = self._rng.choice(self.symbols)
            side = self._rng.choice(["BUY", "SELL"])
            quantity = self._rng.uniform(0.001, 0.01)
            self.simulate_trade(symbol, side, quantity)

        # Random chance to place an order
        if self._rng.random() < 0.1 and len(self.orders) < 5:  # 10% chance, max 5 orders
            symbol = self._rng.choice(self.symbols)
            side = self._rng.choice(["BUY", "SELL"])
            price = self.base_prices[symbol] * (1 + self._rng.uniform(-0.02, 0.02))
            order_id = f"order_{int(time.time() * 1000)}"

            order_quantity = self._rng.uniform(0.001, 0.01)
            # Simulate partial fills (30% chance)
            filled_pct = (
                self._rng.choice([0.0, 0.0, 0.0, 0.25, 0.5, 0.75])
                if self._rng.random() < 0.3
                else 0.0
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
        if self.orders and self._rng.random() < 0.15:  # 15% chance
            cancelled = self._rng.choice(self.orders)
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
