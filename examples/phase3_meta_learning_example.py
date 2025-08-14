"""
Phase 3 Meta-Learning Example
Demonstrates comprehensive meta-learning capabilities including:
- Market regime detection and switching
- Temporal strategy adaptation
- Continuous learning with concept drift detection
"""

import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.meta_learning.regime_detection import RegimeDetector, MarketRegime
from bot.meta_learning.temporal_adaptation import TemporalAdaptationEngine
from bot.meta_learning.continuous_learning import ContinuousLearningPipeline
from bot.knowledge.strategy_knowledge_base import (
    StrategyKnowledgeBase,
    StrategyContext,
    StrategyPerformance,
    StrategyMetadata,
)
from bot.meta_learning.strategy_transfer import StrategyTransferEngine, AssetCharacteristics
from bot.dataflow.sources.enhanced_yfinance_source import EnhancedYFinanceSource
from bot.backtest.engine_portfolio import run_backtest
from bot.strategy.enhanced_trend_breakout import EnhancedTrendBreakoutStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_strategies(knowledge_base: StrategyKnowledgeBase) -> None:
    """Create sample strategies in the knowledge base for demonstration."""
    logger.info("Creating sample strategies in knowledge base...")

    # Sample strategy 1: Trending market strategy
    strategy1 = StrategyMetadata(
        strategy_id="trend_strategy_001",
        name="Enhanced Trend Breakout - Trending",
        description="Optimized for trending markets with high momentum",
        strategy_type="trend_following",
        parameters={
            "donchian_lookback": 55,
            "atr_period": 20,
            "atr_k": 2.0,
            "entry_confirmation_periods": 1,
            "exit_confirmation_periods": 1,
            "max_risk_per_trade": 0.02,
            "use_volume_filter": True,
            "volume_threshold": 1.5,
            "use_correlation_filter": False,
        },
        context=StrategyContext(
            market_regime="trending",
            time_period="bull_market",
            asset_class="equity",
            risk_profile="moderate",
            volatility_regime="medium",
            correlation_regime="medium",
        ),
        performance=StrategyPerformance(
            sharpe_ratio=1.85,
            cagr=0.45,
            max_drawdown=0.12,
            win_rate=0.62,
            consistency_score=0.78,
            n_trades=156,
            avg_trade_duration=5.2,
            profit_factor=1.8,
            calmar_ratio=3.8,
            sortino_ratio=2.1,
            information_ratio=1.2,
            beta=0.8,
            alpha=0.15,
        ),
        discovery_date=datetime.now() - timedelta(days=30),
        last_updated=datetime.now(),
        usage_count=15,
        success_rate=0.82,
        tags=["trending", "momentum", "breakout"],
        notes="Performs well in trending markets with clear directional movement",
    )

    # Sample strategy 2: Volatile market strategy
    strategy2 = StrategyMetadata(
        strategy_id="volatile_strategy_001",
        name="Enhanced Trend Breakout - Volatile",
        description="Adapted for volatile markets with tight risk management",
        strategy_type="trend_following",
        parameters={
            "donchian_lookback": 40,
            "atr_period": 14,
            "atr_k": 1.5,
            "entry_confirmation_periods": 2,
            "exit_confirmation_periods": 1,
            "max_risk_per_trade": 0.015,
            "use_volume_filter": True,
            "volume_threshold": 2.0,
            "use_correlation_filter": True,
        },
        context=StrategyContext(
            market_regime="volatile",
            time_period="sideways_market",
            asset_class="equity",
            risk_profile="conservative",
            volatility_regime="high",
            correlation_regime="high",
        ),
        performance=StrategyPerformance(
            sharpe_ratio=1.45,
            cagr=0.28,
            max_drawdown=0.08,
            win_rate=0.58,
            consistency_score=0.72,
            n_trades=203,
            avg_trade_duration=4.8,
            profit_factor=1.6,
            calmar_ratio=4.2,
            sortino_ratio=1.9,
            information_ratio=1.0,
            beta=0.9,
            alpha=0.12,
        ),
        discovery_date=datetime.now() - timedelta(days=25),
        last_updated=datetime.now(),
        usage_count=12,
        success_rate=0.75,
        tags=["volatile", "conservative", "risk_managed"],
        notes="Conservative approach for volatile market conditions",
    )

    # Sample strategy 3: Crisis market strategy
    strategy3 = StrategyMetadata(
        strategy_id="crisis_strategy_001",
        name="Enhanced Trend Breakout - Crisis",
        description="Ultra-conservative strategy for crisis market conditions",
        strategy_type="trend_following",
        parameters={
            "donchian_lookback": 30,
            "atr_period": 10,
            "atr_k": 1.0,
            "entry_confirmation_periods": 3,
            "exit_confirmation_periods": 1,
            "max_risk_per_trade": 0.01,
            "use_volume_filter": True,
            "volume_threshold": 2.5,
            "use_correlation_filter": True,
        },
        context=StrategyContext(
            market_regime="crisis",
            time_period="bear_market",
            asset_class="equity",
            risk_profile="conservative",
            volatility_regime="high",
            correlation_regime="high",
        ),
        performance=StrategyPerformance(
            sharpe_ratio=0.95,
            cagr=0.12,
            max_drawdown=0.05,
            win_rate=0.52,
            consistency_score=0.68,
            n_trades=89,
            avg_trade_duration=6.1,
            profit_factor=1.3,
            calmar_ratio=2.8,
            sortino_ratio=1.4,
            information_ratio=0.8,
            beta=0.7,
            alpha=0.08,
        ),
        discovery_date=datetime.now() - timedelta(days=20),
        last_updated=datetime.now(),
        usage_count=8,
        success_rate=0.65,
        tags=["crisis", "conservative", "defensive"],
        notes="Defensive strategy for crisis market conditions",
    )

    # Add strategies to knowledge base
    knowledge_base.add_strategy(strategy1)
    knowledge_base.add_strategy(strategy2)
    knowledge_base.add_strategy(strategy3)

    logger.info(f"Created {3} sample strategies in knowledge base")


def demonstrate_regime_detection(
    regime_detector: RegimeDetector,
    market_data: pd.DataFrame,
    knowledge_base: StrategyKnowledgeBase,
) -> None:
    """Demonstrate market regime detection capabilities."""
    print("\n" + "=" * 60)
    print("MARKET REGIME DETECTION DEMONSTRATION")
    print("=" * 60)

    # Detect current regime
    current_regime = regime_detector.detect_regime(market_data)

    print(f"Current Market Regime: {current_regime.regime.value}")
    print(f"Confidence: {current_regime.confidence:.3f}")
    print(f"Duration: {current_regime.duration_days} days")
    print(f"Volatility: {current_regime.volatility:.3f}")
    print(f"Trend Strength: {current_regime.trend_strength:.3f}")
    print(f"Momentum Score: {current_regime.momentum_score:.3f}")
    print(f"Volume Profile: {current_regime.volume_profile}")

    # Get regime recommendations
    recommendations = regime_detector.get_regime_recommendations(current_regime, knowledge_base)

    print(f"\n--- Strategy Recommendations for {current_regime.regime.value} Regime ---")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"{i}. {rec['name']}")
        print(f"   Sharpe Ratio: {rec['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown: {rec['max_drawdown']:.3f}")
        print(f"   Context Match: {rec['context_match']:.3f}")
        print()


def demonstrate_temporal_adaptation(
    temporal_adaptation: TemporalAdaptationEngine,
    knowledge_base: StrategyKnowledgeBase,
    market_data: pd.DataFrame,
    current_regime,
) -> None:
    """Demonstrate temporal strategy adaptation capabilities."""
    print("\n" + "=" * 60)
    print("TEMPORAL STRATEGY ADAPTATION DEMONSTRATION")
    print("=" * 60)

    # Get a strategy to adapt
    strategies = knowledge_base.find_strategies(limit=1)
    if not strategies:
        print("No strategies found for adaptation")
        return

    strategy = strategies[0]
    print(f"Adapting strategy: {strategy.name}")
    print(f"Original Sharpe Ratio: {strategy.performance.sharpe_ratio:.3f}")
    print(f"Original Context: {strategy.context.market_regime}")

    # Adapt strategy
    adaptation_result = temporal_adaptation.adapt_strategy(strategy, market_data, current_regime)

    if adaptation_result["adapted"]:
        print(f"\n--- Adaptation Results ---")
        print(f"Adaptation Reason: {adaptation_result['reason']}")
        print(f"Confidence: {adaptation_result['confidence']:.3f}")

        # Show parameter changes
        original_params = strategy.parameters
        adapted_params = adaptation_result["adapted_parameters"]

        print(f"\n--- Parameter Changes ---")
        for param, value in adapted_params.items():
            if param in original_params and original_params[param] != value:
                print(f"  {param}: {original_params[param]:.3f} → {value:.3f}")

        # Show adaptation rules
        adaptation_rules = adaptation_result.get("adaptation_rules", [])
        if adaptation_rules:
            print(f"\n--- Applied Adaptation Rules ---")
            for rule in adaptation_rules:
                print(f"  {rule['reason']} (confidence: {rule['confidence']:.2f})")
    else:
        print(f"No adaptation needed: {adaptation_result['reason']}")


def demonstrate_continuous_learning(
    continuous_learning: ContinuousLearningPipeline,
    market_data: pd.DataFrame,
    knowledge_base: StrategyKnowledgeBase,
) -> None:
    """Demonstrate continuous learning capabilities."""
    print("\n" + "=" * 60)
    print("CONTINUOUS LEARNING DEMONSTRATION")
    print("=" * 60)

    # Simulate strategy performance data
    strategy_performance = {
        "strategy_001": {"sharpe": 1.85, "max_drawdown": 0.12, "win_rate": 0.62},
        "strategy_002": {"sharpe": 1.45, "max_drawdown": 0.08, "win_rate": 0.58},
        "strategy_003": {"sharpe": 0.95, "max_drawdown": 0.05, "win_rate": 0.52},
    }

    # Process new data
    learning_result = continuous_learning.process_new_data(market_data, strategy_performance)

    print(f"Regime Detected: {learning_result['regime_detected']}")
    print(f"Drift Detected: {learning_result['drift_detected']}")

    if learning_result["drift_detected"]:
        drift_event = learning_result["drift_event"]
        print(f"Drift Type: {drift_event.drift_type}")
        print(f"Drift Magnitude: {drift_event.drift_magnitude:.3f}")
        print(f"Affected Components: {drift_event.affected_components}")

    # Get learning analytics
    analytics = continuous_learning.get_learning_analytics()

    print(f"\n--- Learning Analytics ---")
    print(f"Total Drift Events: {analytics['total_drift_events']}")
    print(f"Total Learning Updates: {analytics['total_learning_updates']}")
    print(f"Active Models: {len(analytics['active_models'])}")

    # Show recent drift events
    if analytics["recent_drift_events"]:
        print(f"\n--- Recent Drift Events ---")
        for event in analytics["recent_drift_events"][:3]:
            print(f"  {event['date']}: {event['type']} drift (magnitude: {event['magnitude']:.3f})")


def demonstrate_strategy_transfer(
    transfer_engine: StrategyTransferEngine, knowledge_base: StrategyKnowledgeBase
) -> None:
    """Demonstrate cross-asset strategy transfer capabilities."""
    print("\n" + "=" * 60)
    print("CROSS-ASSET STRATEGY TRANSFER DEMONSTRATION")
    print("=" * 60)

    # Get a source strategy
    source_strategies = knowledge_base.find_strategies(min_sharpe=1.5, limit=1)
    if not source_strategies:
        print("No source strategies found")
        return

    source_strategy = source_strategies[0]
    print(f"Source Strategy: {source_strategy.name}")
    print(f"Original Sharpe: {source_strategy.performance.sharpe_ratio:.3f}")
    print(f"Original Context: {source_strategy.context.market_regime}")

    # Define target context and asset
    target_context = StrategyContext(
        market_regime="crisis",
        time_period="bear_market",
        asset_class="equity",
        risk_profile="conservative",
        volatility_regime="high",
        correlation_regime="high",
    )

    target_asset = AssetCharacteristics(
        volatility=0.35,  # High volatility
        correlation=0.8,  # High correlation
        volume_profile="high",
        price_range=0.05,
        liquidity="medium",
        market_cap=1000000000.0,
        sector="technology",
    )

    # Transfer strategy
    transfer_result = transfer_engine.transfer_strategy(
        source_strategy, target_context, target_asset
    )

    print(f"\n--- Transfer Results ---")
    print(f"Confidence Score: {transfer_result['confidence_score']:.3f}")
    print(f"Adaptation Notes: {transfer_result['adaptation_notes']}")

    # Show adapted parameters
    print(f"\n--- Adapted Parameters ---")
    adapted_params = transfer_result["adapted_parameters"]
    for param, value in adapted_params.items():
        if param in source_strategy.parameters:
            original = source_strategy.parameters[param]
            if original != value:
                print(f"  {param}: {original:.3f} → {value:.3f}")


def demonstrate_regime_switching(
    regime_detector: RegimeDetector,
    knowledge_base: StrategyKnowledgeBase,
    market_data: pd.DataFrame,
) -> None:
    """Demonstrate automatic regime switching capabilities."""
    print("\n" + "=" * 60)
    print("AUTOMATIC REGIME SWITCHING DEMONSTRATION")
    print("=" * 60)

    # Simulate regime changes over time
    regimes_to_test = [
        MarketRegime.TRENDING_UP,
        MarketRegime.VOLATILE,
        MarketRegime.CRISIS,
        MarketRegime.SIDEWAYS,
        MarketRegime.RECOVERY,
    ]

    print("Testing regime switching for different market conditions:")

    for regime in regimes_to_test:
        # Create synthetic market data for this regime
        synthetic_data = create_synthetic_market_data(regime)

        # Detect regime
        detected_regime = regime_detector.detect_regime(synthetic_data)

        # Get recommendations
        recommendations = regime_detector.get_regime_recommendations(
            detected_regime, knowledge_base
        )

        print(f"\n{regime.value.upper()} Market:")
        print(f"  Detected Regime: {detected_regime.regime.value}")
        print(f"  Confidence: {detected_regime.confidence:.3f}")
        print(f"  Top Strategy: {recommendations[0]['name'] if recommendations else 'None'}")
        print(
            f"  Strategy Sharpe: {recommendations[0]['sharpe_ratio']:.3f}"
            if recommendations
            else "  Strategy Sharpe: 0.000"
        )


def create_synthetic_market_data(regime: MarketRegime) -> pd.DataFrame:
    """Create synthetic market data for different regimes."""
    # This is a simplified synthetic data generator
    # In practice, you would use real market data

    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")

    if regime == MarketRegime.TRENDING_UP:
        # Trending up: steady upward movement
        prices = [100 + i * 0.5 + np.random.normal(0, 2) for i in range(252)]
    elif regime == MarketRegime.VOLATILE:
        # Volatile: high volatility, choppy movement
        prices = [100 + np.random.normal(0, 10) for _ in range(252)]
    elif regime == MarketRegime.CRISIS:
        # Crisis: sharp decline with high volatility
        prices = [100 - i * 0.3 + np.random.normal(0, 15) for i in range(252)]
    elif regime == MarketRegime.SIDEWAYS:
        # Sideways: low volatility, range-bound
        prices = [100 + np.random.normal(0, 3) for _ in range(252)]
    else:  # RECOVERY
        # Recovery: gradual upward movement
        prices = [80 + i * 0.3 + np.random.normal(0, 5) for i in range(252)]

    # Create DataFrame
    data = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": [p + abs(np.random.normal(0, 2)) for p in prices],
            "low": [p - abs(np.random.normal(0, 2)) for p in prices],
            "close": prices,
            "volume": [np.random.randint(1000000, 10000000) for _ in range(252)],
        }
    )

    return data


def main():
    """Run comprehensive Phase 3 meta-learning demonstration."""
    logger.info("Starting Phase 3 Meta-Learning Demonstration")

    # Initialize components
    knowledge_base = StrategyKnowledgeBase()
    regime_detector = RegimeDetector()
    temporal_adaptation = TemporalAdaptationEngine(regime_detector)
    continuous_learning = ContinuousLearningPipeline(
        knowledge_base, regime_detector, temporal_adaptation
    )
    transfer_engine = StrategyTransferEngine(knowledge_base)

    # Create sample strategies
    create_sample_strategies(knowledge_base)

    # Get real market data for demonstration
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Use first symbol for demonstration
    data_source = EnhancedYFinanceSource()
    market_data = data_source.get_daily_bars(
        symbols[0], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )

    if market_data is None or market_data.empty:
        logger.warning("Could not fetch real market data, using synthetic data")
        market_data = create_synthetic_market_data(MarketRegime.TRENDING_UP)

    print("PHASE 3 META-LEARNING SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This demonstration showcases the comprehensive meta-learning capabilities")
    print("implemented in Phase 3, including:")
    print("• Market regime detection and automatic switching")
    print("• Temporal strategy adaptation")
    print("• Continuous learning with concept drift detection")
    print("• Cross-asset strategy transfer")
    print("=" * 80)

    # Run demonstrations
    demonstrate_regime_detection(regime_detector, market_data, knowledge_base)
    demonstrate_temporal_adaptation(
        temporal_adaptation, knowledge_base, market_data, regime_detector.detect_regime(market_data)
    )
    demonstrate_continuous_learning(continuous_learning, market_data, knowledge_base)
    demonstrate_strategy_transfer(transfer_engine, knowledge_base)
    demonstrate_regime_switching(regime_detector, knowledge_base, market_data)

    print("\n" + "=" * 80)
    print("PHASE 3 META-LEARNING DEMONSTRATION COMPLETED")
    print("=" * 80)
    print("Key Achievements:")
    print("✅ Market regime detection with confidence scoring")
    print("✅ Automatic strategy recommendations based on regime")
    print("✅ Temporal adaptation with performance tracking")
    print("✅ Concept drift detection and automatic retraining")
    print("✅ Cross-asset strategy transfer with validation")
    print("✅ Continuous learning pipeline with analytics")
    print("=" * 80)

    logger.info("Phase 3 meta-learning demonstration completed successfully")


if __name__ == "__main__":
    main()
