"""
ML-specific CLI commands for training and automated trading
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import pandas as pd
import yfinance as yf

from .base import BaseCommand


class MLTrainCommand(BaseCommand):
    """Train ML models for portfolio management"""
    
    name = "ml-train"
    help = "Train ML models (regime detector, strategy selector)"
    
    @classmethod
    def add_parser(cls, subparsers):
        """Add ml-train subcommand parser"""
        parser = subparsers.add_parser(
            cls.name,
            help=cls.help,
            description="Train machine learning models for autonomous portfolio management"
        )
        
        # Model selection
        parser.add_argument(
            '--model',
            choices=['regime', 'strategy', 'both'],
            default='both',
            help='Which model(s) to train'
        )
        
        # Data parameters
        parser.add_argument(
            '--symbols',
            nargs='+',
            default=['SPY', 'QQQ', 'IWM', 'DIA'],
            help='Symbols to use for training'
        )
        
        parser.add_argument(
            '--start',
            type=str,
            default=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
            help='Start date for training data (YYYY-MM-DD)'
        )
        
        parser.add_argument(
            '--end',
            type=str,
            default=datetime.now().strftime('%Y-%m-%d'),
            help='End date for training data (YYYY-MM-DD)'
        )
        
        # Training parameters
        parser.add_argument(
            '--optimize-hyperparams',
            action='store_true',
            help='Optimize hyperparameters using Optuna'
        )
        
        parser.add_argument(
            '--n-trials',
            type=int,
            default=50,
            help='Number of Optuna trials for hyperparameter optimization'
        )
        
        parser.add_argument(
            '--validation-split',
            type=float,
            default=0.2,
            help='Validation data split ratio'
        )
        
        # Output parameters
        parser.add_argument(
            '--save-path',
            type=Path,
            default=Path('models'),
            help='Directory to save trained models'
        )
        
        parser.add_argument(
            '--model-name',
            type=str,
            help='Custom name for saved model'
        )
        
        parser.add_argument(
            '--evaluate',
            action='store_true',
            help='Evaluate model performance after training'
        )
        
        return parser
    
    def execute(self, args):
        """Execute ML training"""
        self.logger.info(f"Starting ML training for {args.model} model(s)")
        
        # Create save directory
        args.save_path.mkdir(parents=True, exist_ok=True)
        
        # Fetch training data
        self.logger.info(f"Fetching data for {args.symbols} from {args.start} to {args.end}")
        data = self._fetch_training_data(args.symbols, args.start, args.end)
        
        if data.empty:
            self.logger.error("No data fetched for training")
            return 1
        
        results = {}
        
        # Train regime detector
        if args.model in ['regime', 'both']:
            self.logger.info("Training regime detector...")
            regime_results = self._train_regime_detector(
                data, args.save_path, args.optimize_hyperparams, args.n_trials
            )
            results['regime'] = regime_results
            
            if args.evaluate:
                self._evaluate_regime_detector(regime_results['model_path'])
        
        # Train strategy selector
        if args.model in ['strategy', 'both']:
            self.logger.info("Training strategy selector...")
            strategy_results = self._train_strategy_selector(
                data, args.save_path, args.optimize_hyperparams, 
                args.n_trials, args.validation_split
            )
            results['strategy'] = strategy_results
            
            if args.evaluate:
                self._evaluate_strategy_selector(strategy_results['model_path'])
        
        # Save training summary
        summary_path = args.save_path / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Training complete! Models saved to {args.save_path}")
        self.logger.info(f"Training summary: {summary_path}")
        
        return 0
    
    def _fetch_training_data(self, symbols, start, end):
        """Fetch training data from Yahoo Finance"""
        all_data = {}
        
        for symbol in symbols:
            self.logger.info(f"Fetching {symbol}...")
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start, end=end)
            
            if not hist.empty:
                hist.columns = hist.columns.str.lower()
                all_data[symbol] = hist
                self.logger.info(f"  ✓ {symbol}: {len(hist)} days")
            else:
                self.logger.warning(f"  ✗ {symbol}: No data")
        
        # Combine into single DataFrame for training
        if all_data:
            # Use first symbol as primary
            primary = list(all_data.values())[0]
            return primary
        
        return pd.DataFrame()
    
    def _train_regime_detector(self, data, save_path, optimize_hyperparams, n_trials):
        """Train HMM regime detector"""
        try:
            from ..ml.models import MarketRegimeDetector
            from ..ml.features import FeatureEngineeringPipeline
            
            # Generate features
            feature_pipeline = FeatureEngineeringPipeline()
            features = feature_pipeline.generate_features(data, store_features=False)
            
            # Initialize and train model
            detector = MarketRegimeDetector()
            
            # Select key features for regime detection
            regime_features = ['returns_1d', 'returns_5d', 'volatility_20d', 
                              'volume_ratio', 'rsi_14', 'macd_signal']
            
            # Filter available features
            available_features = [f for f in regime_features if f in features.columns]
            train_features = features[available_features].dropna()
            
            # Train model
            metrics = detector.train(train_features)
            
            # Save model
            model_name = f"regime_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = save_path / f"{model_name}.joblib"
            
            # Save using joblib
            import joblib
            joblib.dump(detector, model_path)
            
            self.logger.info(f"Regime detector saved to {model_path}")
            
            return {
                'model_path': str(model_path),
                'metrics': metrics,
                'n_samples': len(train_features),
                'n_features': len(available_features)
            }
            
        except Exception as e:
            self.logger.error(f"Error training regime detector: {e}")
            return {}
    
    def _train_strategy_selector(self, data, save_path, optimize_hyperparams, 
                                n_trials, validation_split):
        """Train XGBoost strategy selector"""
        try:
            from ..ml.models import StrategyMetaSelector
            from ..ml.features import FeatureEngineeringPipeline
            import numpy as np
            
            # Generate features
            feature_pipeline = FeatureEngineeringPipeline()
            features = feature_pipeline.generate_features(data, store_features=False)
            
            # Create synthetic strategy labels based on market conditions
            # In practice, this would use historical strategy performance
            labels = self._generate_strategy_labels(features)
            
            # Remove NaN
            valid_idx = ~(features.isna().any(axis=1) | labels.isna())
            X = features[valid_idx]
            y = labels[valid_idx]
            
            # Initialize model
            selector = StrategyMetaSelector()
            
            # Train model
            metrics = selector.train(X, y, optimize_hyperparams=optimize_hyperparams)
            
            # Save model
            model_name = f"strategy_selector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = save_path / f"{model_name}.joblib"
            
            # Save using joblib
            import joblib
            joblib.dump(selector, model_path)
            
            self.logger.info(f"Strategy selector saved to {model_path}")
            
            return {
                'model_path': str(model_path),
                'metrics': metrics,
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            
        except Exception as e:
            self.logger.error(f"Error training strategy selector: {e}")
            return {}
    
    def _generate_strategy_labels(self, features):
        """Generate synthetic strategy labels for training"""
        # This is a simplified version - in practice, you'd use actual strategy performance
        labels = pd.Series(index=features.index, dtype=str)
        
        for i in range(len(features)):
            row = features.iloc[i]
            
            # Simple rules for demonstration
            if 'rsi_14' in features.columns and 'trend_strength' in features.columns:
                if row.get('trend_strength', 0) > 0.3:
                    labels.iloc[i] = 'trend_following'
                elif row.get('rsi_14', 50) < 30 or row.get('rsi_14', 50) > 70:
                    labels.iloc[i] = 'mean_reversion'
                elif row.get('volatility_20d', 0) > 0.2:
                    labels.iloc[i] = 'breakout'
                else:
                    labels.iloc[i] = 'momentum'
            else:
                labels.iloc[i] = 'momentum'
        
        return labels
    
    def _evaluate_regime_detector(self, model_path):
        """Evaluate regime detector performance"""
        self.logger.info("Evaluating regime detector...")
        # Implementation would include backtesting and performance metrics
        pass
    
    def _evaluate_strategy_selector(self, model_path):
        """Evaluate strategy selector performance"""
        self.logger.info("Evaluating strategy selector...")
        # Implementation would include accuracy, confusion matrix, etc.
        pass


class AutoTradeCommand(BaseCommand):
    """Automated trading with ML models"""
    
    name = "auto-trade"
    help = "Run automated trading with ML-powered portfolio management"
    
    @classmethod
    def add_parser(cls, subparsers):
        """Add auto-trade subcommand parser"""
        parser = subparsers.add_parser(
            cls.name,
            help=cls.help,
            description="Run autonomous trading with ML models and portfolio optimization"
        )
        
        # Trading mode
        parser.add_argument(
            '--mode',
            choices=['paper', 'live'],
            default='paper',
            help='Trading mode (paper or live)'
        )
        
        # Portfolio parameters
        parser.add_argument(
            '--capital',
            type=float,
            default=100000,
            help='Starting capital'
        )
        
        parser.add_argument(
            '--universe',
            nargs='+',
            default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            help='Trading universe (list of symbols)'
        )
        
        # Model paths
        parser.add_argument(
            '--regime-model',
            type=Path,
            help='Path to trained regime detector model'
        )
        
        parser.add_argument(
            '--strategy-model',
            type=Path,
            help='Path to trained strategy selector model'
        )
        
        # Rebalancing parameters
        parser.add_argument(
            '--rebalance-frequency',
            type=int,
            default=30,
            help='Rebalancing frequency in days'
        )
        
        parser.add_argument(
            '--weight-tolerance',
            type=float,
            default=0.05,
            help='Weight deviation tolerance for rebalancing'
        )
        
        parser.add_argument(
            '--max-turnover',
            type=float,
            default=0.5,
            help='Maximum portfolio turnover per rebalance'
        )
        
        # Risk parameters
        parser.add_argument(
            '--max-position',
            type=float,
            default=0.3,
            help='Maximum position size (as fraction of portfolio)'
        )
        
        parser.add_argument(
            '--stop-loss',
            type=float,
            default=0.1,
            help='Stop loss threshold'
        )
        
        parser.add_argument(
            '--max-drawdown',
            type=float,
            default=0.2,
            help='Maximum drawdown before halting'
        )
        
        # Execution parameters
        parser.add_argument(
            '--check-interval',
            type=int,
            default=60,
            help='Market check interval in seconds'
        )
        
        parser.add_argument(
            '--trading-hours-only',
            action='store_true',
            help='Trade only during market hours'
        )
        
        # Output parameters
        parser.add_argument(
            '--log-trades',
            action='store_true',
            help='Log all trades to file'
        )
        
        parser.add_argument(
            '--dashboard',
            action='store_true',
            help='Launch monitoring dashboard'
        )
        
        return parser
    
    def execute(self, args):
        """Execute automated trading"""
        self.logger.info(f"Starting auto-trade in {args.mode} mode")
        
        # Validate models
        if not self._validate_models(args):
            return 1
        
        # Initialize trading components
        components = self._initialize_components(args)
        
        if not components:
            self.logger.error("Failed to initialize trading components")
            return 1
        
        # Start trading loop
        try:
            self._run_trading_loop(args, components)
        except KeyboardInterrupt:
            self.logger.info("Trading interrupted by user")
        except Exception as e:
            self.logger.error(f"Trading error: {e}")
            return 1
        
        # Generate final report
        self._generate_report(components)
        
        return 0
    
    def _validate_models(self, args):
        """Validate ML models exist and are loadable"""
        if args.regime_model and not args.regime_model.exists():
            self.logger.error(f"Regime model not found: {args.regime_model}")
            return False
        
        if args.strategy_model and not args.strategy_model.exists():
            self.logger.error(f"Strategy model not found: {args.strategy_model}")
            return False
        
        return True
    
    def _initialize_components(self, args):
        """Initialize all trading components"""
        try:
            components = {}
            
            # Load ML models
            if args.regime_model:
                import joblib
                components['regime_detector'] = joblib.load(args.regime_model)
                self.logger.info(f"Loaded regime detector from {args.regime_model}")
            
            if args.strategy_model:
                import joblib
                components['strategy_selector'] = joblib.load(args.strategy_model)
                self.logger.info(f"Loaded strategy selector from {args.strategy_model}")
            
            # Initialize portfolio optimizer
            from ..ml.portfolio import MarkowitzOptimizer
            components['optimizer'] = MarkowitzOptimizer()
            
            # Initialize allocator
            from ..ml.portfolio import MLEnhancedAllocator
            components['allocator'] = MLEnhancedAllocator(
                optimizer=components['optimizer'],
                regime_detector=components.get('regime_detector'),
                strategy_selector=components.get('strategy_selector')
            )
            
            # Initialize rebalancing engine
            from ..rebalancing import RebalancingEngine, RebalancingConfig
            config = RebalancingConfig(
                weight_tolerance=args.weight_tolerance,
                time_interval_days=args.rebalance_frequency,
                max_turnover=args.max_turnover
            )
            components['rebalancer'] = RebalancingEngine(
                config=config,
                allocator=components['allocator']
            )
            
            # Initialize execution broker (paper or live)
            if args.mode == 'paper':
                self.logger.info("Initializing paper trading broker")
                # Would initialize paper broker here
            else:
                self.logger.info("Initializing live trading broker")
                # Would initialize live broker here
            
            return components
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return None
    
    def _run_trading_loop(self, args, components):
        """Main trading loop"""
        import time
        
        self.logger.info("Starting trading loop...")
        self.logger.info(f"Universe: {args.universe}")
        self.logger.info(f"Capital: ${args.capital:,.2f}")
        
        iteration = 0
        current_positions = {symbol: 0.0 for symbol in args.universe}
        portfolio_value = args.capital
        
        while True:
            iteration += 1
            self.logger.info(f"\n--- Iteration {iteration} ---")
            
            # Fetch current market data
            market_data = self._fetch_market_data(args.universe)
            
            if not market_data:
                self.logger.warning("No market data available, skipping iteration")
                time.sleep(args.check_interval)
                continue
            
            # Get current prices
            current_prices = {
                symbol: data['close'].iloc[-1] 
                for symbol, data in market_data.items()
            }
            
            # Check if rebalancing needed
            needs_rebalancing, reason, details = components['rebalancer'].check_rebalancing_needed(
                current_positions, market_data, current_prices
            )
            
            if needs_rebalancing:
                self.logger.info(f"Rebalancing triggered: {reason}")
                
                # Execute rebalancing
                result = components['rebalancer'].execute_rebalancing(
                    current_positions,
                    details.get('target_positions', {}),
                    current_prices
                )
                
                if result['success']:
                    self.logger.info(f"Rebalancing complete: {result['n_trades']} trades")
                    current_positions = components['rebalancer'].current_positions
                
                if args.log_trades:
                    self._log_trades(result)
            else:
                self.logger.info(f"No rebalancing needed: {reason}")
            
            # Update portfolio value
            portfolio_value = sum(current_positions.values())
            self.logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
            
            # Check risk limits
            if self._check_risk_limits(portfolio_value, args):
                self.logger.warning("Risk limits breached, halting trading")
                break
            
            # Sleep until next check
            time.sleep(args.check_interval)
    
    def _fetch_market_data(self, symbols):
        """Fetch current market data"""
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1mo')
                if not hist.empty:
                    hist.columns = hist.columns.str.lower()
                    market_data[symbol] = hist
            except Exception as e:
                self.logger.error(f"Error fetching {symbol}: {e}")
        
        return market_data
    
    def _check_risk_limits(self, portfolio_value, args):
        """Check if risk limits are breached"""
        initial_value = args.capital
        drawdown = (initial_value - portfolio_value) / initial_value
        
        if drawdown > args.max_drawdown:
            self.logger.error(f"Max drawdown breached: {drawdown:.1%}")
            return True
        
        return False
    
    def _log_trades(self, result):
        """Log trades to file"""
        # Implementation would log to CSV or database
        pass
    
    def _generate_report(self, components):
        """Generate final trading report"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Trading Session Complete")
        self.logger.info("=" * 60)
        
        if 'rebalancer' in components:
            analysis = components['rebalancer'].analyze_rebalancing_history()
            
            self.logger.info(f"Total rebalances: {analysis.get('total_rebalances', 0)}")
            self.logger.info(f"Total cost: ${analysis.get('total_cost', 0):.2f}")
            self.logger.info(f"Average turnover: {analysis.get('avg_turnover_ratio', 0):.1%}")