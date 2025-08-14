"""
Test data factories for generating realistic test data.

Provides factories for creating market data, strategies, portfolios, and trades.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import random


class MarketDataFactory:
    """Factory for generating realistic market data."""
    
    @staticmethod
    def create_ohlcv(
        symbol: str = "TEST",
        start_date: str = "2023-01-01",
        end_date: str = "2023-12-31",
        freq: str = "D",
        initial_price: float = 100.0,
        drift: float = 0.0005,
        volatility: float = 0.02,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate OHLCV data with specified characteristics."""
        if seed is not None:
            np.random.seed(seed)
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_periods = len(dates)
        
        # Generate returns
        returns = np.random.normal(drift, volatility, n_periods)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        data = pd.DataFrame(index=dates)
        data["Close"] = prices
        
        # Open is close from previous day with small gap
        data["Open"] = data["Close"].shift(1) * (1 + np.random.normal(0, 0.002, n_periods))
        data["Open"].iloc[0] = initial_price
        
        # High/Low based on daily range
        daily_range = np.abs(np.random.normal(0, volatility/2, n_periods))
        data["High"] = np.maximum(data["Open"], data["Close"]) * (1 + daily_range)
        data["Low"] = np.minimum(data["Open"], data["Close"]) * (1 - daily_range)
        
        # Volume correlated with volatility
        base_volume = 10000000
        volume_mult = 1 + np.abs(returns) * 10
        data["Volume"] = (base_volume * volume_mult).astype(int)
        
        return data
    
    @staticmethod
    def create_market_scenario(scenario: str = "normal", **kwargs) -> pd.DataFrame:
        """Create predefined market scenarios."""
        scenarios = {
            "bull": {"drift": 0.002, "volatility": 0.01},
            "bear": {"drift": -0.001, "volatility": 0.025},
            "volatile": {"drift": 0, "volatility": 0.04},
            "crash": {"drift": -0.005, "volatility": 0.05},
            "recovery": {"drift": 0.003, "volatility": 0.02},
            "sideways": {"drift": 0, "volatility": 0.01},
            "normal": {"drift": 0.0005, "volatility": 0.02}
        }
        
        params = scenarios.get(scenario, scenarios["normal"])
        params.update(kwargs)
        return MarketDataFactory.create_ohlcv(**params)
    
    @staticmethod
    def create_intraday_data(
        symbol: str = "TEST",
        date: str = "2023-01-01",
        interval_minutes: int = 5
    ) -> pd.DataFrame:
        """Generate intraday market data."""
        start = pd.Timestamp(f"{date} 09:30:00")
        end = pd.Timestamp(f"{date} 16:00:00")
        
        times = pd.date_range(start=start, end=end, freq=f"{interval_minutes}min")
        n_periods = len(times)
        
        # Intraday patterns
        morning_vol = 0.03
        midday_vol = 0.01
        closing_vol = 0.025
        
        volatilities = np.concatenate([
            np.full(n_periods // 4, morning_vol),
            np.full(n_periods // 2, midday_vol),
            np.full(n_periods - n_periods // 4 - n_periods // 2, closing_vol)
        ])
        
        returns = np.random.normal(0, volatilities[:n_periods])
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            "Close": prices,
            "Volume": np.random.randint(100000, 1000000, n_periods)
        }, index=times)


class StrategyFactory:
    """Factory for creating test strategy objects."""
    
    @staticmethod
    def create_strategy(
        name: str = "test_strategy",
        strategy_type: str = "trend_following",
        parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a test strategy configuration."""
        default_params = {
            "trend_following": {
                "lookback_period": 20,
                "entry_threshold": 0.02,
                "exit_threshold": -0.01,
                "stop_loss": 0.05,
                "take_profit": 0.10
            },
            "mean_reversion": {
                "lookback_period": 30,
                "z_score_entry": 2.0,
                "z_score_exit": 0.5,
                "stop_loss": 0.03,
                "take_profit": 0.05
            },
            "momentum": {
                "lookback_period": 10,
                "momentum_threshold": 0.03,
                "holding_period": 5,
                "stop_loss": 0.04,
                "take_profit": 0.08
            }
        }
        
        params = parameters or default_params.get(strategy_type, {})
        
        return {
            "name": name,
            "type": strategy_type,
            "parameters": params,
            "enabled": True,
            "risk_per_trade": 0.02,
            "max_positions": 5
        }
    
    @staticmethod
    def create_signal(
        signal_type: str = "BUY",
        confidence: float = 0.8,
        price: float = 100.0
    ) -> Dict[str, Any]:
        """Create a test trading signal."""
        return {
            "timestamp": datetime.now(),
            "signal": signal_type,
            "confidence": confidence,
            "price": price,
            "stop_loss": price * 0.95 if signal_type == "BUY" else price * 1.05,
            "take_profit": price * 1.05 if signal_type == "BUY" else price * 0.95,
            "size": 100,
            "metadata": {
                "strategy": "test_strategy",
                "indicators": {"RSI": 50, "MACD": 0.5}
            }
        }


class PortfolioFactory:
    """Factory for creating test portfolio objects."""
    
    @staticmethod
    def create_portfolio(
        cash: float = 100000.0,
        positions: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Create a test portfolio."""
        if positions is None:
            positions = [
                {"symbol": "AAPL", "quantity": 100, "entry_price": 150.0},
                {"symbol": "GOOGL", "quantity": 50, "entry_price": 2800.0},
                {"symbol": "MSFT", "quantity": 75, "entry_price": 300.0}
            ]
        
        portfolio = {
            "cash": cash,
            "positions": {},
            "total_value": cash,
            "returns": []
        }
        
        for pos in positions:
            current_price = pos["entry_price"] * (1 + np.random.normal(0, 0.1))
            portfolio["positions"][pos["symbol"]] = {
                "quantity": pos["quantity"],
                "entry_price": pos["entry_price"],
                "current_price": current_price,
                "market_value": pos["quantity"] * current_price,
                "unrealized_pnl": pos["quantity"] * (current_price - pos["entry_price"])
            }
            portfolio["total_value"] += portfolio["positions"][pos["symbol"]]["market_value"]
        
        return portfolio
    
    @staticmethod
    def create_allocation(
        symbols: List[str] = None,
        allocation_method: str = "equal_weight"
    ) -> Dict[str, float]:
        """Create portfolio allocation weights."""
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        n_symbols = len(symbols)
        
        if allocation_method == "equal_weight":
            weights = {symbol: 1.0 / n_symbols for symbol in symbols}
        elif allocation_method == "market_cap":
            # Simulate market cap weights
            caps = np.random.exponential(1, n_symbols)
            caps = caps / caps.sum()
            weights = {symbol: cap for symbol, cap in zip(symbols, caps)}
        elif allocation_method == "risk_parity":
            # Simulate risk parity weights
            vols = np.random.uniform(0.15, 0.35, n_symbols)
            inv_vols = 1.0 / vols
            inv_vols = inv_vols / inv_vols.sum()
            weights = {symbol: weight for symbol, weight in zip(symbols, inv_vols)}
        else:
            weights = {symbol: 1.0 / n_symbols for symbol in symbols}
        
        return weights


class TradeFactory:
    """Factory for creating test trade objects."""
    
    @staticmethod
    def create_trade(
        symbol: str = "TEST",
        trade_type: str = "BUY",
        quantity: int = 100,
        price: float = 100.0,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Create a test trade."""
        if timestamp is None:
            timestamp = datetime.now()
        
        commission = quantity * price * 0.001  # 0.1% commission
        
        return {
            "id": f"trade_{random.randint(10000, 99999)}",
            "timestamp": timestamp,
            "symbol": symbol,
            "type": trade_type,
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "total_value": quantity * price + commission,
            "status": "FILLED",
            "metadata": {
                "strategy": "test_strategy",
                "signal_confidence": 0.8
            }
        }
    
    @staticmethod
    def create_trade_history(
        n_trades: int = 10,
        symbols: Optional[List[str]] = None,
        start_date: str = "2023-01-01"
    ) -> List[Dict[str, Any]]:
        """Create a history of trades."""
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT"]
        
        trades = []
        current_date = pd.Timestamp(start_date)
        
        for i in range(n_trades):
            symbol = random.choice(symbols)
            trade_type = "BUY" if i % 2 == 0 else "SELL"
            quantity = random.randint(10, 200)
            price = 100 * (1 + np.random.normal(0, 0.1))
            
            trade = TradeFactory.create_trade(
                symbol=symbol,
                trade_type=trade_type,
                quantity=quantity,
                price=price,
                timestamp=current_date
            )
            trades.append(trade)
            
            # Advance time
            current_date += timedelta(days=random.randint(1, 5))
        
        return trades


class RiskMetricsFactory:
    """Factory for creating test risk metrics."""
    
    @staticmethod
    def create_metrics(
        returns: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Create test risk metrics."""
        if returns is None:
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
            returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        
        if benchmark_returns is None:
            benchmark_returns = pd.Series(
                np.random.normal(0.0005, 0.015, len(returns)),
                index=returns.index
            )
        
        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # vs Benchmark
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "win_rate": (returns > 0).mean(),
            "average_win": returns[returns > 0].mean() if (returns > 0).any() else 0,
            "average_loss": returns[returns < 0].mean() if (returns < 0).any() else 0,
            "var_95": np.percentile(returns, 5),
            "cvar_95": returns[returns <= np.percentile(returns, 5)].mean()
        }