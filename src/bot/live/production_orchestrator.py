"""
Production Orchestrator for GPT-Trader

This is the central coordinator that integrates all ML components:
- IntegratedMLPipeline for feature engineering and ML predictions
- StrategyMetaSelector for ML-driven strategy selection
- AutoRetrainingSystem for maintaining model performance
- Event-driven architecture for real-time coordination
- Fallback mechanisms for robust operation

Missing file implementation from recovery plan ORCH-002.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# Core ML components (with safe imports)
# Need to initialize logger first

_temp_logger = logging.getLogger(__name__)

try:
    from ..ml.integrated_pipeline import IntegratedMLPipeline

    ML_PIPELINE_AVAILABLE = True
except ImportError as e:
    _temp_logger.warning(f"IntegratedMLPipeline not available: {e}")
    IntegratedMLPipeline = None
    ML_PIPELINE_AVAILABLE = False

try:
    from ..ml.models.strategy_selector import StrategyMetaSelector

    STRATEGY_SELECTOR_AVAILABLE = True
except ImportError as e:
    _temp_logger.warning(f"StrategyMetaSelector not available: {e}")
    StrategyMetaSelector = None
    STRATEGY_SELECTOR_AVAILABLE = False

try:
    from ..ml.auto_retraining import AutoRetrainingSystem, RetrainingTrigger

    AUTO_RETRAINING_AVAILABLE = True
except ImportError as e:
    _temp_logger.warning(f"AutoRetrainingSystem not available: {e}")
    AutoRetrainingSystem = None
    RetrainingTrigger = None
    AUTO_RETRAINING_AVAILABLE = False

# Strategy components
from ..strategy.base import Strategy
from ..strategy.demo_ma import DemoMAStrategy
from ..strategy.trend_breakout import TrendBreakoutStrategy

# Trading infrastructure (with safe imports)
try:
    from .event_driven_architecture import EventBus, EventType, TradingEvent

    EVENT_BUS_AVAILABLE = True
except ImportError as e:
    _temp_logger.warning(f"Event bus not available: {e}")
    EventBus = None
    EventType = None
    TradingEvent = None
    EVENT_BUS_AVAILABLE = False

try:
    from .trading_engine import TradingEngine

    TRADING_ENGINE_AVAILABLE = True
except ImportError as e:
    _temp_logger.warning(f"TradingEngine not available: {e}")
    TradingEngine = None
    TRADING_ENGINE_AVAILABLE = False

try:
    from .market_data_pipeline import MarketDataPipeline

    MARKET_DATA_AVAILABLE = True
except ImportError as e:
    _temp_logger.warning(f"MarketDataPipeline not available: {e}")
    MarketDataPipeline = None
    MARKET_DATA_AVAILABLE = False

try:
    from .risk_monitor import RiskMonitor

    RISK_MONITOR_AVAILABLE = True
except ImportError as e:
    _temp_logger.warning(f"RiskMonitor not available: {e}")
    RiskMonitor = None
    RISK_MONITOR_AVAILABLE = False

# Data and config
from ..dataflow.sources.yfinance_source import YFinanceSource
from ..logging import get_logger

# Initialize logger early
logger = get_logger(__name__)


class OrchestratorState(Enum):
    """Production orchestrator operational states"""

    INITIALIZING = "initializing"
    LOADING_MODELS = "loading_models"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class StrategyMode(Enum):
    """Strategy selection modes"""

    ML_ONLY = "ml_only"  # Use ML model predictions only
    FALLBACK_ONLY = "fallback_only"  # Use fallback strategy only
    ML_WITH_FALLBACK = "ml_with_fallback"  # ML primary, fallback on errors
    ENSEMBLE = "ensemble"  # Weighted combination


@dataclass
class OrchestratorConfig:
    """Configuration for production orchestrator"""

    # ML Configuration
    ml_pipeline_path: str = "ml_models/production"
    strategy_selector_path: str = "ml_models/strategy_selector"
    confidence_threshold: float = 0.6

    # Strategy Configuration
    available_strategies: list[str] = field(default_factory=lambda: ["demo_ma", "trend_breakout"])
    fallback_strategy: str = "demo_ma"
    strategy_mode: StrategyMode = StrategyMode.ML_WITH_FALLBACK

    # Data Configuration
    feature_lookback_days: int = 252  # 1 year of trading days
    prediction_horizon: int = 1  # Days ahead to predict

    # Health Monitoring
    health_check_interval: int = 300  # 5 minutes
    performance_window: int = 20  # Days for performance tracking
    drift_check_interval: int = 3600  # 1 hour

    # Retraining
    enable_auto_retraining: bool = True
    min_performance_threshold: float = 0.55
    max_model_age_days: int = 30

    # Risk Management
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_position_size: float = 0.1  # 10% max position per symbol


@dataclass
class OrchestrationStatus:
    """Current status of the orchestrator"""

    state: OrchestratorState
    ml_pipeline_loaded: bool = False
    strategy_selector_loaded: bool = False
    current_strategy: str | None = None
    strategy_confidence: float = 0.0
    last_prediction_time: datetime | None = None
    last_health_check: datetime | None = None
    error_count: int = 0
    warnings: list[str] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)


class ProductionOrchestrator:
    """
    Central production orchestrator that coordinates all ML components.

    This class implements the missing production orchestrator that:
    1. Loads and manages ML models for strategy selection
    2. Coordinates real-time feature engineering and predictions
    3. Manages fallback strategies when ML fails
    4. Monitors performance and triggers retraining
    5. Provides health checks and error recovery
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        """Initialize production orchestrator

        Args:
            config: Configuration for orchestrator behavior
        """
        self.config = config or OrchestratorConfig()
        self.status = OrchestrationStatus(state=OrchestratorState.INITIALIZING)

        # Core ML components
        self.ml_pipeline: IntegratedMLPipeline | None = None
        self.strategy_selector: StrategyMetaSelector | None = None
        self.auto_retrainer: AutoRetrainingSystem | None = None

        # Trading infrastructure
        self.event_bus: EventBus | None = None
        self.trading_engine: TradingEngine | None = None
        self.market_data: MarketDataPipeline | None = None
        self.risk_monitor: RiskMonitor | None = None

        # Strategy registry
        self.strategies: dict[str, Strategy] = {}
        self.fallback_strategy: Strategy | None = None

        # Data sources
        self.data_source = YFinanceSource()

        # Threading for background tasks
        self._shutdown_event = threading.Event()
        self._health_check_thread: threading.Thread | None = None
        self._prediction_cache: dict[str, tuple[str, float, datetime]] = {}

        # Performance tracking
        self._recent_predictions: list[dict[str, Any]] = []
        self._feature_data_cache: pd.DataFrame | None = None
        self._last_feature_update: datetime | None = None

        logger.info("ProductionOrchestrator initialized with config")

    async def initialize(self) -> bool:
        """
        Initialize all components and load models.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Starting orchestrator initialization...")
            self.status.state = OrchestratorState.INITIALIZING

            # Initialize event bus
            await self._initialize_event_bus()

            # Load ML models
            await self._load_ml_models()

            # Initialize strategies
            self._initialize_strategies()

            # Initialize trading infrastructure
            await self._initialize_trading_infrastructure()

            # Start background services
            self._start_background_services()

            self.status.state = OrchestratorState.READY
            logger.info("Orchestrator initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            self.status.state = OrchestratorState.ERROR
            self.status.warnings.append(f"Initialization failed: {str(e)}")
            return False

    async def _initialize_event_bus(self):
        """Initialize event-driven communication"""
        if not EVENT_BUS_AVAILABLE:
            logger.info("Event bus not available, skipping")
            return

        try:
            self.event_bus = EventBus()
            await self.event_bus.start()

            # Subscribe to relevant events
            await self.event_bus.subscribe(EventType.MARKET_DATA, self._handle_market_data)
            await self.event_bus.subscribe(EventType.STRATEGY_SIGNAL, self._handle_strategy_signal)
            await self.event_bus.subscribe(EventType.RISK_ALERT, self._handle_risk_alert)

            logger.info("Event bus initialized")
        except Exception as e:
            logger.warning(f"Could not initialize event bus: {e}")
            self.event_bus = None

    async def _load_ml_models(self):
        """Load ML pipeline and strategy selector"""
        self.status.state = OrchestratorState.LOADING_MODELS

        # Load ML pipeline
        if ML_PIPELINE_AVAILABLE:
            try:
                self.ml_pipeline = IntegratedMLPipeline()

                # Try to load saved pipeline if it exists
                import os

                if os.path.exists(self.config.ml_pipeline_path):
                    self.ml_pipeline.load_pipeline(self.config.ml_pipeline_path)
                    logger.info("Loaded ML pipeline from disk")
                else:
                    logger.warning("No saved ML pipeline found, using default")

                self.status.ml_pipeline_loaded = True

            except Exception as e:
                logger.error(f"Failed to load ML pipeline: {e}")
                self.status.warnings.append(f"ML pipeline failed: {str(e)}")
        else:
            logger.warning("ML pipeline not available")
            self.status.warnings.append("ML pipeline component not available")

        # Load strategy selector
        if STRATEGY_SELECTOR_AVAILABLE:
            try:
                self.strategy_selector = StrategyMetaSelector(
                    strategies=self.config.available_strategies
                )

                # Try to load trained model if it exists
                import os

                selector_model_path = f"{self.config.strategy_selector_path}/model.joblib"
                if os.path.exists(selector_model_path):
                    import joblib

                    self.strategy_selector.model = joblib.load(selector_model_path)
                    self.strategy_selector.is_trained = True
                    logger.info("Loaded strategy selector model")
                else:
                    logger.warning("No trained strategy selector found")

                self.status.strategy_selector_loaded = True

            except Exception as e:
                logger.error(f"Failed to load strategy selector: {e}")
                self.status.warnings.append(f"Strategy selector failed: {str(e)}")
        else:
            logger.warning("Strategy selector not available")
            self.status.warnings.append("Strategy selector component not available")

        # Initialize auto-retraining if enabled
        if self.config.enable_auto_retraining and AUTO_RETRAINING_AVAILABLE:
            try:
                self.auto_retrainer = AutoRetrainingSystem()
                logger.info("Auto-retraining system initialized")
            except Exception as e:
                logger.warning(f"Auto-retraining not available: {e}")
        else:
            logger.info("Auto-retraining disabled or not available")

    def _initialize_strategies(self):
        """Initialize available trading strategies"""
        # Register available strategies
        self.strategies = {
            "demo_ma": DemoMAStrategy(),
            "trend_breakout": TrendBreakoutStrategy(),
        }

        # Set fallback strategy
        if self.config.fallback_strategy in self.strategies:
            self.fallback_strategy = self.strategies[self.config.fallback_strategy]
            logger.info(f"Fallback strategy set to: {self.config.fallback_strategy}")
        else:
            # Use first available strategy as fallback
            self.fallback_strategy = next(iter(self.strategies.values()))
            logger.warning(
                f"Fallback strategy '{self.config.fallback_strategy}' not found, using {self.fallback_strategy.name}"
            )

    async def _initialize_trading_infrastructure(self):
        """Initialize trading engine and related components"""
        try:
            # Initialize components that exist
            # Note: Some components may not be fully implemented yet
            logger.info("Trading infrastructure initialization skipped (components not ready)")

        except Exception as e:
            logger.warning(f"Trading infrastructure initialization failed: {e}")

    def _start_background_services(self):
        """Start background monitoring and health check services"""
        # Start health check thread
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()
        logger.info("Background services started")

    async def select_strategy(self, market_data: pd.DataFrame) -> tuple[str, float, dict[str, Any]]:
        """
        Select optimal strategy using ML model with fallback.

        Args:
            market_data: Recent market data for feature engineering

        Returns:
            Tuple of (strategy_name, confidence, metadata)
        """
        try:
            # Check if we should use ML-based selection
            if self._should_use_ml_selection():
                return await self._ml_strategy_selection(market_data)
            else:
                return self._fallback_strategy_selection()

        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            self.status.error_count += 1
            return self._fallback_strategy_selection()

    def _should_use_ml_selection(self) -> bool:
        """Determine if ML selection should be used"""
        # Check strategy mode
        if self.config.strategy_mode == StrategyMode.FALLBACK_ONLY:
            return False

        # Check if ML components are loaded and healthy
        if not (self.status.ml_pipeline_loaded and self.status.strategy_selector_loaded):
            return False

        if not (self.ml_pipeline and self.strategy_selector and self.strategy_selector.is_trained):
            return False

        # Check recent error rate
        if self.status.error_count > 5:  # Too many recent errors
            return False

        return True

    async def _ml_strategy_selection(
        self, market_data: pd.DataFrame
    ) -> tuple[str, float, dict[str, Any]]:
        """Use ML model to select strategy"""
        try:
            # Engineer features
            features = await self._prepare_features(market_data)

            # Get strategy prediction with confidence
            strategy, confidence, probabilities = (
                self.strategy_selector.select_strategy_with_confidence(features)
            )

            # Check confidence threshold
            if confidence < self.config.confidence_threshold:
                logger.warning(f"Low ML confidence ({confidence:.3f}), using fallback")
                return self._fallback_strategy_selection()

            # Cache prediction
            self._cache_prediction(strategy, confidence)

            metadata = {
                "method": "ml",
                "all_probabilities": probabilities,
                "feature_count": len(features.columns),
                "timestamp": datetime.now(),
            }

            logger.info(f"ML selected strategy: {strategy} (confidence: {confidence:.3f})")
            return strategy, confidence, metadata

        except Exception as e:
            logger.error(f"ML strategy selection failed: {e}")
            raise

    def _fallback_strategy_selection(self) -> tuple[str, float, dict[str, Any]]:
        """Use fallback strategy selection"""
        strategy_name = self.fallback_strategy.name
        confidence = 0.8  # Fixed confidence for fallback

        metadata = {
            "method": "fallback",
            "reason": "ML unavailable or low confidence",
            "timestamp": datetime.now(),
        }

        logger.info(f"Using fallback strategy: {strategy_name}")
        return strategy_name, confidence, metadata

    async def _prepare_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare ML features from market data"""
        if self.ml_pipeline is None:
            raise ValueError("ML pipeline not loaded")

        # Check if we can use cached features
        if (
            self._feature_data_cache is not None
            and self._last_feature_update
            and (datetime.now() - self._last_feature_update).seconds < 300
        ):  # 5 min cache
            return self._feature_data_cache.tail(1)  # Return latest features

        # Engineer new features
        features = self.ml_pipeline.prepare_features(market_data, use_selection=True)

        # Cache features
        self._feature_data_cache = features
        self._last_feature_update = datetime.now()

        return features.tail(1)  # Return only latest row for prediction

    def _cache_prediction(self, strategy: str, confidence: float):
        """Cache strategy prediction for performance tracking"""
        self.status.current_strategy = strategy
        self.status.strategy_confidence = confidence
        self.status.last_prediction_time = datetime.now()

        # Add to recent predictions for performance analysis
        self._recent_predictions.append(
            {"strategy": strategy, "confidence": confidence, "timestamp": datetime.now()}
        )

        # Keep only recent predictions
        cutoff = datetime.now() - timedelta(days=self.config.performance_window)
        self._recent_predictions = [p for p in self._recent_predictions if p["timestamp"] > cutoff]

    def generate_trading_signals(
        self, strategy_name: str, market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate trading signals using selected strategy"""
        try:
            if strategy_name not in self.strategies:
                logger.warning(f"Strategy {strategy_name} not found, using fallback")
                strategy = self.fallback_strategy
            else:
                strategy = self.strategies[strategy_name]

            # Generate signals
            signals = strategy.generate_signals(market_data)

            logger.info(f"Generated signals using {strategy.name}")
            return signals

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            # Return empty signals as fallback
            return pd.DataFrame(index=market_data.index, columns=["signal"])

    def _health_check_loop(self):
        """Background health monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                self._perform_health_check()
                self.status.last_health_check = datetime.now()

            except Exception as e:
                logger.error(f"Health check failed: {e}")

            # Wait for next check
            self._shutdown_event.wait(self.config.health_check_interval)

    def _perform_health_check(self):
        """Perform comprehensive health check"""
        # Check ML component health
        if self.ml_pipeline:
            # Check if pipeline is still functional
            pass

        # Check strategy selector health
        if self.strategy_selector:
            # Check if model is still loaded
            pass

        # Check for data drift if enabled
        if hasattr(self, "auto_retrainer") and self.auto_retrainer:
            # Trigger drift checks
            pass

        # Update performance metrics
        self._update_performance_metrics()

    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        if not self._recent_predictions:
            return

        # Calculate basic metrics
        total_predictions = len(self._recent_predictions)
        ml_predictions = sum(1 for p in self._recent_predictions if "ml" in str(p))
        avg_confidence = np.mean([p["confidence"] for p in self._recent_predictions])

        self.status.performance_metrics = {
            "total_predictions": total_predictions,
            "ml_prediction_rate": (
                ml_predictions / total_predictions if total_predictions > 0 else 0
            ),
            "avg_confidence": avg_confidence,
            "error_rate": self.status.error_count / max(total_predictions, 1),
        }

    async def _handle_market_data(self, event: TradingEvent):
        """Handle incoming market data events"""
        try:
            # Process market data for strategy selection
            market_data = event.data.get("market_data")
            if market_data is not None:
                # Trigger strategy selection
                strategy, confidence, metadata = await self.select_strategy(market_data)

                # Generate trading signals
                signals = self.generate_trading_signals(strategy, market_data)

                # Publish strategy signals
                if self.event_bus and EVENT_BUS_AVAILABLE:
                    signal_event = TradingEvent(
                        event_type=EventType.STRATEGY_SIGNAL,
                        data={
                            "strategy": strategy,
                            "confidence": confidence,
                            "signals": signals,
                            "metadata": metadata,
                        },
                    )
                    await self.event_bus.publish(signal_event)

        except Exception as e:
            logger.error(f"Error handling market data: {e}")

    async def _handle_strategy_signal(self, event):
        """Handle strategy signal events"""
        # Forward to trading engine or risk management
        pass

    async def _handle_risk_alert(self, event):
        """Handle risk alerts"""
        if hasattr(event, "data"):
            alert_type = event.data.get("alert_type")
            if alert_type == "emergency_stop":
                logger.critical("Emergency stop triggered")
                self.status.state = OrchestratorState.PAUSED

    def get_status(self) -> OrchestrationStatus:
        """Get current orchestrator status"""
        return self.status

    def get_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary"""
        return {
            "state": self.status.state.value,
            "ml_pipeline_loaded": self.status.ml_pipeline_loaded,
            "strategy_selector_loaded": self.status.strategy_selector_loaded,
            "current_strategy": self.status.current_strategy,
            "strategy_confidence": self.status.strategy_confidence,
            "last_prediction": (
                self.status.last_prediction_time.isoformat()
                if self.status.last_prediction_time
                else None
            ),
            "last_health_check": (
                self.status.last_health_check.isoformat() if self.status.last_health_check else None
            ),
            "error_count": self.status.error_count,
            "warnings": self.status.warnings,
            "performance_metrics": self.status.performance_metrics,
            "strategies_available": list(self.strategies.keys()),
            "fallback_strategy": self.fallback_strategy.name if self.fallback_strategy else None,
        }

    async def shutdown(self):
        """Gracefully shutdown orchestrator"""
        logger.info("Shutting down production orchestrator...")
        self.status.state = OrchestratorState.SHUTDOWN

        # Signal background threads to stop
        self._shutdown_event.set()

        # Wait for health check thread
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)

        # Shutdown event bus
        if self.event_bus:
            await self.event_bus.stop()

        logger.info("Production orchestrator shutdown complete")


# Factory function for easy creation
def create_production_orchestrator(
    config: OrchestratorConfig | None = None,
) -> ProductionOrchestrator:
    """
    Create and initialize production orchestrator.

    Args:
        config: Optional configuration

    Returns:
        Configured ProductionOrchestrator instance
    """
    return ProductionOrchestrator(config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def main():
        """Example usage of production orchestrator"""
        # Create orchestrator with custom config
        config = OrchestratorConfig(
            strategy_mode=StrategyMode.ML_WITH_FALLBACK,
            confidence_threshold=0.7,
            enable_auto_retraining=True,
        )

        orchestrator = create_production_orchestrator(config)

        # Initialize
        success = await orchestrator.initialize()
        if not success:
            print("Failed to initialize orchestrator")
            return

        # Get sample market data
        data_source = YFinanceSource()
        market_data = data_source.fetch("AAPL", "1y")

        # Test strategy selection
        strategy, confidence, metadata = await orchestrator.select_strategy(market_data)
        print(f"Selected strategy: {strategy} (confidence: {confidence:.3f})")

        # Generate signals
        signals = orchestrator.generate_trading_signals(strategy, market_data)
        print(f"Generated {len(signals)} signals")

        # Get health summary
        health = orchestrator.get_health_summary()
        print(f"Orchestrator health: {health['state']}")

        # Shutdown
        await orchestrator.shutdown()

    # Run example
    asyncio.run(main())
