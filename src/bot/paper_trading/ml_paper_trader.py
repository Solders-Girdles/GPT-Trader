"""
ML-Enhanced Paper Trading System

Production-ready paper trader that integrates all ML components
for autonomous portfolio management with simulated execution.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from src.bot.core.base import BaseComponent
from src.bot.ml.features.engineering import FeatureEngineeringPipeline
from src.bot.ml.models.regime_detector import MarketRegimeDetector
from src.bot.ml.models.strategy_selector import StrategyMetaSelector
from src.bot.ml.portfolio.optimizer import MarkowitzOptimizer
from src.bot.ml.portfolio.allocator import MLEnhancedAllocator
from src.bot.rebalancing.engine import RebalancingEngine
from src.bot.strategy.ml_enhanced import MLEnhancedStrategy
from src.bot.dataflow.sources.yfinance_source import YFinanceSource
from src.bot.monitor.performance_monitor import PerformanceMonitor
from src.bot.exceptions import TradingError

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """Simulated position in paper trading"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    value: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    
    def update_price(self, price: float):
        """Update position with current price"""
        self.current_price = price
        self.value = self.quantity * price
        self.pnl = (price - self.entry_price) * self.quantity
        self.pnl_percent = ((price - self.entry_price) / self.entry_price) * 100


@dataclass
class PaperOrder:
    """Simulated order in paper trading"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market' or 'limit'
    limit_price: Optional[float]
    status: str  # 'pending', 'filled', 'cancelled'
    filled_price: Optional[float]
    filled_time: Optional[datetime]
    commission: float = 0.0


@dataclass
class MLPaperTradingConfig:
    """Configuration for ML paper trading"""
    initial_capital: float = 100000
    max_positions: int = 10
    position_size_pct: float = 0.1  # 10% per position
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    
    # ML settings
    retrain_interval_days: int = 7
    prediction_horizon: int = 5  # days
    min_confidence: float = 0.6
    
    # Rebalancing settings
    rebalance_threshold: float = 0.05
    min_rebalance_interval: int = 24  # hours
    max_slippage: float = 0.002
    
    # Risk settings
    max_portfolio_risk: float = 0.20
    max_drawdown: float = 0.15
    var_limit: float = 0.05
    
    # Execution settings
    commission_rate: float = 0.001
    spread_cost: float = 0.0005
    market_impact: float = 0.0001


class MLPaperTrader(BaseComponent):
    """
    ML-Enhanced Paper Trading System
    
    Integrates all ML components for autonomous trading simulation.
    """
    
    def __init__(self, config: Optional[MLPaperTradingConfig] = None):
        super().__init__()
        self.config = config or MLPaperTradingConfig()
        
        # Initialize components
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.regime_detector = MarketRegimeDetector()
        self.strategy_selector = StrategyMetaSelector()
        self.portfolio_optimizer = MarkowitzOptimizer()
        self.ml_allocator = MLEnhancedAllocator()
        self.rebalancing_engine = RebalancingEngine()
        self.data_source = YFinanceSource()
        self.performance_monitor = PerformanceMonitor()
        
        # Paper trading state
        self.capital = self.config.initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: List[PaperOrder] = []
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        self.metrics_history: List[Dict] = []
        
        # ML model state
        self.models_trained = False
        self.last_retrain: Optional[datetime] = None
        self.current_regime: Optional[str] = None
        self.selected_strategy: Optional[str] = None
        
        # Trading state
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._stop_event = threading.Event()
        
        logger.info("MLPaperTrader initialized with capital: $%.2f", self.capital)
    
    async def initialize(self):
        """Initialize ML models and load historical data"""
        logger.info("Initializing MLPaperTrader...")
        
        # Load or train ML models
        await self._load_or_train_models()
        
        # Initialize portfolio
        self.portfolio_values.append((datetime.now(), self.capital))
        
        logger.info("MLPaperTrader initialization complete")
    
    async def _load_or_train_models(self):
        """Load existing models or train new ones"""
        model_dir = Path("models/ml_paper_trader")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        regime_model_path = model_dir / "regime_detector.joblib"
        strategy_model_path = model_dir / "strategy_selector.joblib"
        
        if regime_model_path.exists() and strategy_model_path.exists():
            logger.info("Loading existing ML models...")
            import joblib
            self.regime_detector = joblib.load(regime_model_path)
            self.strategy_selector = joblib.load(strategy_model_path)
            self.models_trained = True
        else:
            logger.info("Training new ML models...")
            await self._train_models()
    
    async def _train_models(self):
        """Train ML models on historical data"""
        logger.info("Fetching training data...")
        
        # Get historical data for training
        symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
        
        training_data = {}
        for symbol in symbols:
            data = await self.data_source.fetch_historical(
                symbol, start_date, end_date
            )
            if data is not None:
                training_data[symbol] = data
        
        if not training_data:
            logger.error("Failed to fetch training data")
            return
        
        # Generate features for each symbol
        all_features = []
        all_labels = []
        
        for symbol, data in training_data.items():
            features = self.feature_pipeline.generate_features(data)
            
            # Create labels (next period returns)
            returns = data['close'].pct_change(self.config.prediction_horizon)
            labels = returns.shift(-self.config.prediction_horizon)
            
            # Align features and labels
            valid_idx = ~labels.isna()
            all_features.append(features[valid_idx])
            all_labels.append(labels[valid_idx])
        
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        
        # Train regime detector (unsupervised)
        logger.info("Training regime detector...")
        self.regime_detector.train(X)
        
        # Create strategy labels based on returns
        strategy_labels = []
        for ret in y:
            if ret > 0.02:
                strategy_labels.append('trend_following')
            elif ret < -0.02:
                strategy_labels.append('mean_reversion')
            elif abs(ret) < 0.005:
                strategy_labels.append('market_neutral')
            else:
                strategy_labels.append('momentum')
        
        # Train strategy selector
        logger.info("Training strategy selector...")
        self.strategy_selector.train(X, pd.Series(strategy_labels))
        
        # Save models
        model_dir = Path("models/ml_paper_trader")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        import joblib
        joblib.dump(self.regime_detector, model_dir / "regime_detector.joblib")
        joblib.dump(self.strategy_selector, model_dir / "strategy_selector.joblib")
        
        self.models_trained = True
        self.last_retrain = datetime.now()
        
        logger.info("Model training complete")
    
    def start(self, symbols: List[str]):
        """Start paper trading"""
        if self.is_running:
            logger.warning("Paper trader already running")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        # Start trading loop in background
        self.executor.submit(self._trading_loop, symbols)
        
        logger.info("Paper trading started with symbols: %s", symbols)
    
    def stop(self):
        """Stop paper trading"""
        if not self.is_running:
            return
        
        logger.info("Stopping paper trader...")
        self.is_running = False
        self._stop_event.set()
        
        # Close all positions
        self._close_all_positions()
        
        # Generate final report
        self._generate_report()
        
        logger.info("Paper trader stopped")
    
    def _trading_loop(self, symbols: List[str]):
        """Main trading loop"""
        logger.info("Trading loop started")
        
        while self.is_running and not self._stop_event.is_set():
            try:
                # Get current market data
                market_data = self._fetch_current_data(symbols)
                
                if not market_data:
                    time.sleep(60)  # Wait before retry
                    continue
                
                # Update position prices
                self._update_positions(market_data)
                
                # Check if models need retraining
                if self._should_retrain():
                    asyncio.run(self._train_models())
                
                # Generate ML predictions
                predictions = self._generate_predictions(market_data)
                
                # Make trading decisions
                self._make_trading_decisions(predictions, market_data)
                
                # Check rebalancing
                if self._should_rebalance():
                    self._rebalance_portfolio(market_data)
                
                # Update performance metrics
                self._update_metrics()
                
                # Sleep based on trading frequency
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
        
        logger.info("Trading loop ended")
    
    def _fetch_current_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch current market data"""
        market_data = {}
        
        for symbol in symbols:
            try:
                # Get recent data for feature generation
                data = self.data_source.fetch_latest(symbol, periods=100)
                if data is not None and not data.empty:
                    market_data[symbol] = data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return market_data
    
    def _update_positions(self, market_data: Dict[str, pd.DataFrame]):
        """Update position prices and P&L"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['close'].iloc[-1]
                position.update_price(current_price)
                
                # Check stop loss and take profit
                if position.pnl_percent <= -self.config.stop_loss_pct * 100:
                    logger.info(f"Stop loss triggered for {symbol}")
                    self._close_position(symbol, "stop_loss")
                elif position.pnl_percent >= self.config.take_profit_pct * 100:
                    logger.info(f"Take profit triggered for {symbol}")
                    self._close_position(symbol, "take_profit")
    
    def _generate_predictions(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate ML predictions for all symbols"""
        predictions = {}
        
        for symbol, data in market_data.items():
            try:
                # Generate features
                features = self.feature_pipeline.generate_features(data)
                
                if features.empty:
                    continue
                
                # Get latest features
                latest_features = features.iloc[[-1]]
                
                # Detect market regime
                regime = self.regime_detector.predict(latest_features)[0]
                regime_name = self.regime_detector.get_regime_name(regime)
                regime_confidence = self.regime_detector.get_regime_confidence(latest_features)[1][0]
                
                # Select strategy
                strategy, confidence, probs = self.strategy_selector.select_strategy_with_confidence(
                    latest_features
                )
                
                predictions[symbol] = {
                    'regime': regime_name,
                    'regime_confidence': regime_confidence,
                    'strategy': strategy,
                    'strategy_confidence': confidence,
                    'strategy_probs': probs,
                    'features': latest_features
                }
                
            except Exception as e:
                logger.error(f"Error generating predictions for {symbol}: {e}")
        
        return predictions
    
    def _make_trading_decisions(self, predictions: Dict, market_data: Dict[str, pd.DataFrame]):
        """Make trading decisions based on ML predictions"""
        
        # Calculate portfolio value
        portfolio_value = self._calculate_portfolio_value()
        
        for symbol, pred in predictions.items():
            try:
                # Skip if confidence too low
                if pred['strategy_confidence'] < self.config.min_confidence:
                    continue
                
                # Get current price
                current_price = market_data[symbol]['close'].iloc[-1]
                
                # Check if we have a position
                has_position = symbol in self.positions
                
                # Trading logic based on strategy and regime
                if pred['regime'] in ['bull_quiet', 'bull_volatile']:
                    if pred['strategy'] in ['trend_following', 'momentum']:
                        if not has_position:
                            # Calculate position size
                            position_size = self._calculate_position_size(
                                portfolio_value, pred['strategy_confidence']
                            )
                            
                            # Open long position
                            self._open_position(symbol, position_size, current_price, 'buy')
                
                elif pred['regime'] in ['bear_quiet', 'bear_volatile']:
                    if has_position:
                        # Close position in bear market
                        self._close_position(symbol, 'regime_change')
                
                elif pred['regime'] == 'sideways':
                    if pred['strategy'] == 'mean_reversion':
                        # Implement mean reversion logic
                        pass
                
            except Exception as e:
                logger.error(f"Error making decision for {symbol}: {e}")
    
    def _should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced"""
        if not self.positions:
            return False
        
        # Check time since last rebalance
        if hasattr(self, 'last_rebalance'):
            hours_since = (datetime.now() - self.last_rebalance).total_seconds() / 3600
            if hours_since < self.config.min_rebalance_interval:
                return False
        
        # Check weight deviations
        portfolio_value = self._calculate_portfolio_value()
        
        for symbol, position in self.positions.items():
            weight = position.value / portfolio_value
            target_weight = 1.0 / len(self.positions)  # Equal weight for simplicity
            
            if abs(weight - target_weight) > self.config.rebalance_threshold:
                return True
        
        return False
    
    def _rebalance_portfolio(self, market_data: Dict[str, pd.DataFrame]):
        """Rebalance portfolio to target weights"""
        logger.info("Rebalancing portfolio...")
        
        portfolio_value = self._calculate_portfolio_value()
        n_positions = len(self.positions)
        
        if n_positions == 0:
            return
        
        target_value = portfolio_value / n_positions  # Equal weight
        
        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]['close'].iloc[-1]
            current_value = position.value
            
            # Calculate rebalancing trade
            value_diff = target_value - current_value
            shares_to_trade = value_diff / current_price
            
            if abs(shares_to_trade) * current_price > 100:  # Min trade size
                if shares_to_trade > 0:
                    self._execute_order(symbol, shares_to_trade, current_price, 'buy')
                else:
                    self._execute_order(symbol, abs(shares_to_trade), current_price, 'sell')
        
        self.last_rebalance = datetime.now()
        logger.info("Rebalancing complete")
    
    def _should_retrain(self) -> bool:
        """Check if models should be retrained"""
        if not self.last_retrain:
            return True
        
        days_since = (datetime.now() - self.last_retrain).days
        return days_since >= self.config.retrain_interval_days
    
    def _calculate_position_size(self, portfolio_value: float, confidence: float) -> float:
        """Calculate position size based on confidence and risk"""
        base_size = portfolio_value * self.config.position_size_pct
        
        # Adjust based on confidence
        size_multiplier = 0.5 + (confidence * 0.5)  # 50% to 100% of base size
        
        return base_size * size_multiplier
    
    def _open_position(self, symbol: str, size: float, price: float, side: str):
        """Open a new position"""
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return
        
        # Calculate shares
        shares = size / price
        
        # Execute order
        order = self._execute_order(symbol, shares, price, side)
        
        if order and order.status == 'filled':
            # Create position
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                quantity=shares,
                entry_price=order.filled_price,
                current_price=order.filled_price,
                entry_time=datetime.now(),
                value=shares * order.filled_price
            )
            
            # Update capital
            self.capital -= (shares * order.filled_price + order.commission)
            
            logger.info(f"Opened position: {symbol} - {shares:.2f} shares @ ${order.filled_price:.2f}")
    
    def _close_position(self, symbol: str, reason: str):
        """Close an existing position"""
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return
        
        position = self.positions[symbol]
        
        # Execute sell order
        order = self._execute_order(
            symbol, position.quantity, position.current_price, 'sell'
        )
        
        if order and order.status == 'filled':
            # Update capital
            self.capital += (position.quantity * order.filled_price - order.commission)
            
            # Record trade
            self.trade_history.append({
                'symbol': symbol,
                'entry_time': position.entry_time,
                'exit_time': datetime.now(),
                'entry_price': position.entry_price,
                'exit_price': order.filled_price,
                'quantity': position.quantity,
                'pnl': position.pnl,
                'pnl_percent': position.pnl_percent,
                'reason': reason
            })
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"Closed position: {symbol} - P&L: ${position.pnl:.2f} ({position.pnl_percent:.2f}%)")
    
    def _close_all_positions(self):
        """Close all open positions"""
        symbols = list(self.positions.keys())
        for symbol in symbols:
            self._close_position(symbol, 'shutdown')
    
    def _execute_order(self, symbol: str, quantity: float, price: float, side: str) -> PaperOrder:
        """Execute a paper order with simulated fills"""
        
        # Calculate slippage
        slippage = price * self.config.max_slippage * (1 if side == 'buy' else -1)
        filled_price = price + slippage
        
        # Calculate commission
        commission = abs(quantity * filled_price * self.config.commission_rate)
        
        # Create order
        order = PaperOrder(
            order_id=f"PO_{datetime.now().strftime('%Y%m%d%H%M%S')}_{symbol}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type='market',
            limit_price=None,
            status='filled',
            filled_price=filled_price,
            filled_time=datetime.now(),
            commission=commission
        )
        
        self.orders.append(order)
        
        return order
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        position_value = sum(pos.value for pos in self.positions.values())
        return self.capital + position_value
    
    def _update_metrics(self):
        """Update performance metrics"""
        current_value = self._calculate_portfolio_value()
        current_time = datetime.now()
        
        # Record portfolio value
        self.portfolio_values.append((current_time, current_value))
        
        # Calculate daily return if we have previous day's value
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2][1]
            daily_return = (current_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
        
        # Calculate metrics
        if len(self.daily_returns) > 20:  # Need sufficient data
            metrics = self._calculate_performance_metrics()
            self.metrics_history.append({
                'timestamp': current_time,
                **metrics
            })
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        returns = np.array(self.daily_returns)
        
        # Basic metrics
        total_return = (self._calculate_portfolio_value() - self.config.initial_capital) / self.config.initial_capital
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = avg_return / volatility * np.sqrt(252) if volatility > 0 else 0
        
        # Maximum drawdown
        portfolio_values = [v for _, v in self.portfolio_values]
        drawdowns = []
        peak = portfolio_values[0]
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Win rate
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        
        return {
            'total_return': total_return * 100,
            'avg_daily_return': avg_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'num_trades': len(self.trade_history),
            'open_positions': len(self.positions)
        }
    
    def _generate_report(self):
        """Generate comprehensive trading report"""
        logger.info("Generating trading report...")
        
        report = {
            'summary': {
                'initial_capital': self.config.initial_capital,
                'final_value': self._calculate_portfolio_value(),
                'total_trades': len(self.trade_history),
                'open_positions': len(self.positions)
            },
            'performance': self._calculate_performance_metrics() if self.daily_returns else {},
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl': pos.pnl,
                    'pnl_percent': pos.pnl_percent
                }
                for symbol, pos in self.positions.items()
            },
            'recent_trades': self.trade_history[-10:] if self.trade_history else []
        }
        
        # Save report
        report_path = Path("reports/paper_trading")
        report_path.mkdir(parents=True, exist_ok=True)
        
        report_file = report_path / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("PAPER TRADING REPORT")
        print("=" * 60)
        print(f"Initial Capital: ${report['summary']['initial_capital']:,.2f}")
        print(f"Final Value: ${report['summary']['final_value']:,.2f}")
        print(f"Total Return: {report['performance'].get('total_return', 0):.2f}%")
        print(f"Sharpe Ratio: {report['performance'].get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {report['performance'].get('max_drawdown', 0):.2f}%")
        print(f"Win Rate: {report['performance'].get('win_rate', 0):.2f}%")
        print(f"Total Trades: {report['summary']['total_trades']}")
        print("=" * 60)
    
    def get_status(self) -> Dict:
        """Get current trading status"""
        return {
            'is_running': self.is_running,
            'models_trained': self.models_trained,
            'current_regime': self.current_regime,
            'selected_strategy': self.selected_strategy,
            'portfolio_value': self._calculate_portfolio_value(),
            'capital': self.capital,
            'positions': len(self.positions),
            'open_orders': len([o for o in self.orders if o.status == 'pending']),
            'total_trades': len(self.trade_history),
            'performance': self._calculate_performance_metrics() if self.daily_returns else {}
        }


async def main():
    """Example usage of MLPaperTrader"""
    
    # Create configuration
    config = MLPaperTradingConfig(
        initial_capital=100000,
        max_positions=5,
        position_size_pct=0.15,
        min_confidence=0.65,
        rebalance_threshold=0.10
    )
    
    # Create trader
    trader = MLPaperTrader(config)
    
    # Initialize
    await trader.initialize()
    
    # Define trading universe
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Start trading
    trader.start(symbols)
    
    # Run for some time (in production, this would run continuously)
    try:
        print("Paper trading started. Press Ctrl+C to stop...")
        while True:
            # Print status every minute
            status = trader.get_status()
            print(f"\nPortfolio Value: ${status['portfolio_value']:,.2f}")
            print(f"Open Positions: {status['positions']}")
            print(f"Total Trades: {status['total_trades']}")
            
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nStopping paper trader...")
        trader.stop()


if __name__ == "__main__":
    asyncio.run(main())