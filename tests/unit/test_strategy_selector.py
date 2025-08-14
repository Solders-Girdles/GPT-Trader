"""
Unit tests for the Real-Time Strategy Selector component.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
from bot.knowledge.strategy_knowledge_base import (
    StrategyContext,
    StrategyKnowledgeBase,
    StrategyMetadata,
    StrategyPerformance,
)
from bot.live.strategy_selector import (
    RealTimeStrategySelector,
    SelectionConfig,
    SelectionMethod,
    StrategyScore,
)
from bot.meta_learning.regime_detection import MarketRegime, RegimeCharacteristics, RegimeDetector


class TestStrategySelector:
    """Test cases for the RealTimeStrategySelector class."""

    @pytest.fixture
    def mock_knowledge_base(self):
        """Create a mock knowledge base with sample strategies."""
        knowledge_base = Mock(spec=StrategyKnowledgeBase)

        # Create sample strategies
        strategies = []
        for i in range(5):
            strategy = StrategyMetadata(
                strategy_id=f"strategy_{i}",
                name=f"Test Strategy {i}",
                description=f"Test strategy {i}",
                strategy_type="trend_following",
                parameters={"param1": i, "param2": i * 2},
                context=StrategyContext(
                    market_regime="trending",
                    time_period="bull_market",
                    asset_class="equity",
                    risk_profile="moderate",
                    volatility_regime="medium",
                    correlation_regime="low",
                ),
                performance=StrategyPerformance(
                    sharpe_ratio=1.0 + i * 0.2,
                    cagr=0.1 + i * 0.02,
                    max_drawdown=0.1 - i * 0.01,
                    win_rate=0.6 + i * 0.02,
                    consistency_score=0.7 + i * 0.02,
                    n_trades=50 + i * 10,
                    avg_trade_duration=5.0,
                    profit_factor=1.3 + i * 0.1,
                    calmar_ratio=1.2 + i * 0.1,
                    sortino_ratio=1.5 + i * 0.1,
                    information_ratio=1.0 + i * 0.1,
                    beta=0.8 + i * 0.05,
                    alpha=0.05 + i * 0.01,
                ),
                discovery_date=datetime.now() - timedelta(days=30),
                last_updated=datetime.now() - timedelta(days=5),
                usage_count=10 + i * 5,
                success_rate=0.7 + i * 0.02,
            )
            strategies.append(strategy)

        knowledge_base.find_strategies.return_value = strategies
        return knowledge_base

    @pytest.fixture
    def mock_regime_detector(self):
        """Create a mock regime detector."""
        detector = Mock(spec=RegimeDetector)

        # Mock regime characteristics
        regime_char = RegimeCharacteristics(
            regime=MarketRegime.TRENDING_UP,
            confidence=0.85,
            duration_days=15,
            volatility=0.18,
            trend_strength=0.25,
            correlation_level=0.3,
            volume_profile="normal",
            momentum_score=0.7,
            regime_features={"feature1": 0.5, "feature2": 0.3},
        )

        detector.detect_regime.return_value = regime_char
        detector._regime_to_context.return_value = StrategyContext(
            market_regime="trending",
            time_period="bull_market",
            asset_class="equity",
            risk_profile="moderate",
            volatility_regime="medium",
            correlation_regime="low",
        )
        detector._calculate_context_match.return_value = 0.8

        return detector

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SelectionConfig(
            selection_method=SelectionMethod.HYBRID,
            max_strategies=3,
            min_confidence=0.6,
            min_sharpe=0.5,
            max_drawdown=0.15,
            regime_weight=0.3,
            performance_weight=0.4,
            confidence_weight=0.2,
            risk_weight=0.1,
            adaptation_weight=0.2,
        )

    @pytest.fixture
    def strategy_selector(self, mock_knowledge_base, mock_regime_detector, config):
        """Create a strategy selector instance for testing."""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        with (
            patch("bot.live.strategy_selector.TemporalAdaptationEngine") as mock_adaptation,
            patch("bot.live.strategy_selector.StrategyDecompositionAnalyzer") as mock_decomp,
            patch("bot.live.strategy_selector.PerformanceAttributionAnalyzer") as mock_attr,
        ):

            mock_adaptation.return_value.check_adaptation_needed.return_value = False
            mock_adaptation.return_value.calculate_adaptation_quality.return_value = 0.9

            selector = RealTimeStrategySelector(
                knowledge_base=mock_knowledge_base,
                regime_detector=mock_regime_detector,
                config=config,
                symbols=symbols,
            )

            return selector

    def test_initialization(self, strategy_selector):
        """Test strategy selector initialization."""
        assert strategy_selector is not None
        assert len(strategy_selector.current_selection) == 0
        assert strategy_selector.last_selection_time is None
        assert len(strategy_selector.selection_history) == 0

    def test_get_candidate_strategies(self, strategy_selector, mock_knowledge_base):
        """Test getting candidate strategies from knowledge base."""
        # Create a mock regime
        regime_char = RegimeCharacteristics(
            regime=MarketRegime.TRENDING_UP,
            confidence=0.85,
            duration_days=15,
            volatility=0.18,
            trend_strength=0.25,
            correlation_level=0.3,
            volume_profile="normal",
            momentum_score=0.7,
            regime_features={"feature1": 0.5, "feature2": 0.3},
        )

        candidates = strategy_selector._get_candidate_strategies(regime_char)

        # Verify knowledge base was called
        mock_knowledge_base.find_strategies.assert_called_once()
        call_args = mock_knowledge_base.find_strategies.call_args

        # Verify the call arguments
        assert call_args[1]["min_sharpe"] == 0.5
        assert call_args[1]["max_drawdown"] == 0.15
        assert call_args[1]["limit"] == 50

        # Verify we got strategies back
        assert len(candidates) == 5

    def test_calculate_regime_match(self, strategy_selector):
        """Test regime match calculation."""
        # Create a test strategy
        strategy = StrategyMetadata(
            strategy_id="test_strategy",
            name="Test Strategy",
            description="Test strategy",
            strategy_type="trend_following",
            parameters={"param1": 1},
            context=StrategyContext(
                market_regime="trending",
                time_period="bull_market",
                asset_class="equity",
                risk_profile="moderate",
                volatility_regime="medium",
                correlation_regime="low",
            ),
            performance=StrategyPerformance(
                sharpe_ratio=1.5,
                cagr=0.15,
                max_drawdown=0.1,
                win_rate=0.65,
                consistency_score=0.8,
                n_trades=50,
                avg_trade_duration=5.0,
                profit_factor=1.4,
                calmar_ratio=1.5,
                sortino_ratio=1.8,
                information_ratio=1.2,
                beta=0.9,
                alpha=0.08,
            ),
            discovery_date=datetime.now() - timedelta(days=30),
            last_updated=datetime.now() - timedelta(days=5),
            usage_count=20,
            success_rate=0.75,
        )

        # Create a test regime
        regime_char = RegimeCharacteristics(
            regime=MarketRegime.TRENDING_UP,
            confidence=0.85,
            duration_days=15,
            volatility=0.18,
            trend_strength=0.25,
            correlation_level=0.3,
            volume_profile="normal",
            momentum_score=0.7,
            regime_features={"feature1": 0.5, "feature2": 0.3},
        )

        # Calculate regime match
        match_score = strategy_selector._calculate_regime_match(strategy, regime_char)

        # Verify the score is reasonable
        assert 0.0 <= match_score <= 1.0
        assert match_score > 0.0  # Should have some match

    def test_calculate_performance_score(self, strategy_selector):
        """Test performance score calculation."""
        # Create a test strategy with good performance
        strategy = StrategyMetadata(
            strategy_id="test_strategy",
            name="Test Strategy",
            description="Test strategy",
            strategy_type="trend_following",
            parameters={"param1": 1},
            context=StrategyContext(
                market_regime="trending",
                time_period="bull_market",
                asset_class="equity",
                risk_profile="moderate",
                volatility_regime="medium",
                correlation_regime="low",
            ),
            performance=StrategyPerformance(
                sharpe_ratio=2.0,  # Good Sharpe ratio
                cagr=0.25,  # Good CAGR
                max_drawdown=0.08,  # Low drawdown
                win_rate=0.7,
                consistency_score=0.85,
                n_trades=50,
                avg_trade_duration=5.0,
                profit_factor=1.6,
                calmar_ratio=3.1,
                sortino_ratio=2.5,
                information_ratio=1.5,
                beta=0.8,
                alpha=0.12,
            ),
            discovery_date=datetime.now() - timedelta(days=30),
            last_updated=datetime.now() - timedelta(days=5),
            usage_count=20,
            success_rate=0.75,
        )

        # Calculate performance score
        perf_score = strategy_selector._calculate_performance_score(strategy)

        # Verify the score is reasonable
        assert 0.0 <= perf_score <= 1.0
        assert perf_score > 0.5  # Should be good performance

    def test_calculate_confidence_score(self, strategy_selector):
        """Test confidence score calculation."""
        # Create a test strategy with good confidence indicators
        strategy = StrategyMetadata(
            strategy_id="test_strategy",
            name="Test Strategy",
            description="Test strategy",
            strategy_type="trend_following",
            parameters={"param1": 1},
            context=StrategyContext(
                market_regime="trending",
                time_period="bull_market",
                asset_class="equity",
                risk_profile="moderate",
                volatility_regime="medium",
                correlation_regime="low",
            ),
            performance=StrategyPerformance(
                sharpe_ratio=1.5,
                cagr=0.15,
                max_drawdown=0.1,
                win_rate=0.65,
                consistency_score=0.8,
                n_trades=50,
                avg_trade_duration=5.0,
                profit_factor=1.4,
                calmar_ratio=1.5,
                sortino_ratio=1.8,
                information_ratio=1.2,
                beta=0.9,
                alpha=0.08,
            ),
            discovery_date=datetime.now() - timedelta(days=30),
            last_updated=datetime.now() - timedelta(days=1),  # Recently updated
            usage_count=100,  # High usage
            success_rate=0.85,  # High success rate
        )

        # Calculate confidence score
        conf_score = strategy_selector._calculate_confidence_score(strategy)

        # Verify the score is reasonable
        assert 0.0 <= conf_score <= 1.0
        assert conf_score > 0.5  # Should be good confidence

    def test_calculate_risk_score(self, strategy_selector):
        """Test risk score calculation."""
        # Create a test strategy with low risk
        strategy = StrategyMetadata(
            strategy_id="test_strategy",
            name="Test Strategy",
            description="Test strategy",
            strategy_type="trend_following",
            parameters={"param1": 1},
            context=StrategyContext(
                market_regime="trending",
                time_period="bull_market",
                asset_class="equity",
                risk_profile="moderate",
                volatility_regime="medium",
                correlation_regime="low",
            ),
            performance=StrategyPerformance(
                sharpe_ratio=1.5,
                cagr=0.15,
                max_drawdown=0.05,  # Low drawdown
                win_rate=0.65,
                consistency_score=0.8,
                n_trades=50,
                avg_trade_duration=5.0,
                profit_factor=1.4,
                calmar_ratio=1.5,
                sortino_ratio=1.8,
                information_ratio=1.2,
                beta=0.5,  # Low beta
                alpha=0.08,
            ),
            discovery_date=datetime.now() - timedelta(days=30),
            last_updated=datetime.now() - timedelta(days=5),
            usage_count=20,
            success_rate=0.75,
        )

        # Calculate risk score
        risk_score = strategy_selector._calculate_risk_score(strategy)

        # Verify the score is reasonable
        assert 0.0 <= risk_score <= 1.0
        assert risk_score > 0.5  # Should be low risk (high score)

    def test_determine_selection_reason(self, strategy_selector):
        """Test selection reason determination."""
        # Test with high regime match
        reason = strategy_selector._determine_selection_reason(
            regime_match=0.9, performance=0.6, confidence=0.5, risk=0.7, adaptation=0.6
        )
        assert reason == "strong_regime_match"

        # Test with high performance
        reason = strategy_selector._determine_selection_reason(
            regime_match=0.5, performance=0.9, confidence=0.5, risk=0.7, adaptation=0.6
        )
        assert reason == "strong_performance"

        # Test with balanced scores
        reason = strategy_selector._determine_selection_reason(
            regime_match=0.4, performance=0.4, confidence=0.4, risk=0.4, adaptation=0.4
        )
        assert reason == "balanced_selection"

    def test_get_selection_summary(self, strategy_selector):
        """Test getting selection summary."""
        # Test with no selection
        summary = strategy_selector.get_selection_summary()
        assert summary["status"] == "no_selection"

        # Test with selection
        strategy_selector.current_selection = [
            StrategyScore(
                strategy_id="test_1",
                strategy=Mock(),
                overall_score=0.8,
                regime_match_score=0.7,
                performance_score=0.8,
                confidence_score=0.6,
                risk_score=0.7,
                adaptation_score=0.8,
                selection_reason="strong_performance",
            )
        ]
        strategy_selector.last_selection_time = datetime.now()

        summary = strategy_selector.get_selection_summary()
        assert summary["n_strategies"] == 1
        assert summary["avg_score"] == 0.8
        assert len(summary["strategies"]) == 1
        assert summary["strategies"][0]["strategy_id"] == "test_1"

    def test_get_selection_history(self, strategy_selector):
        """Test getting selection history."""
        # Initially empty
        history = strategy_selector.get_selection_history()
        assert len(history) == 0

        # Add some history
        strategy_selector.selection_history = [
            {"timestamp": datetime.now(), "regime": "trending", "strategies": []}
        ]

        history = strategy_selector.get_selection_history()
        assert len(history) == 1
        assert history[0]["regime"] == "trending"


class TestSelectionConfig:
    """Test cases for the SelectionConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SelectionConfig()

        assert config.selection_method == SelectionMethod.HYBRID
        assert config.max_strategies == 5
        assert config.min_confidence == 0.7
        assert config.min_sharpe == 0.5
        assert config.max_drawdown == 0.15
        assert config.regime_weight == 0.3
        assert config.performance_weight == 0.4
        assert config.confidence_weight == 0.2
        assert config.risk_weight == 0.1
        assert config.adaptation_weight == 0.2
        assert config.rebalance_interval == 3600
        assert config.lookback_days == 30

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SelectionConfig(
            selection_method=SelectionMethod.REGIME_BASED,
            max_strategies=10,
            min_confidence=0.8,
            min_sharpe=0.7,
            max_drawdown=0.1,
            regime_weight=0.5,
            performance_weight=0.3,
            confidence_weight=0.1,
            risk_weight=0.05,
            adaptation_weight=0.05,
            rebalance_interval=1800,
            lookback_days=60,
        )

        assert config.selection_method == SelectionMethod.REGIME_BASED
        assert config.max_strategies == 10
        assert config.min_confidence == 0.8
        assert config.min_sharpe == 0.7
        assert config.max_drawdown == 0.1
        assert config.regime_weight == 0.5
        assert config.performance_weight == 0.3
        assert config.confidence_weight == 0.1
        assert config.risk_weight == 0.05
        assert config.adaptation_weight == 0.05
        assert config.rebalance_interval == 1800
        assert config.lookback_days == 60


class TestStrategyScore:
    """Test cases for the StrategyScore class."""

    def test_strategy_score_creation(self):
        """Test creating a StrategyScore instance."""
        strategy = Mock()
        score = StrategyScore(
            strategy_id="test_strategy",
            strategy=strategy,
            overall_score=0.85,
            regime_match_score=0.8,
            performance_score=0.9,
            confidence_score=0.7,
            risk_score=0.8,
            adaptation_score=0.75,
            selection_reason="strong_performance",
        )

        assert score.strategy_id == "test_strategy"
        assert score.strategy == strategy
        assert score.overall_score == 0.85
        assert score.regime_match_score == 0.8
        assert score.performance_score == 0.9
        assert score.confidence_score == 0.7
        assert score.risk_score == 0.8
        assert score.adaptation_score == 0.75
        assert score.selection_reason == "strong_performance"

    def test_strategy_score_comparison(self):
        """Test comparing StrategyScore instances."""
        strategy1 = Mock()
        strategy2 = Mock()

        score1 = StrategyScore(
            strategy_id="strategy_1",
            strategy=strategy1,
            overall_score=0.8,
            regime_match_score=0.7,
            performance_score=0.8,
            confidence_score=0.6,
            risk_score=0.7,
            adaptation_score=0.8,
            selection_reason="strong_performance",
        )

        score2 = StrategyScore(
            strategy_id="strategy_2",
            strategy=strategy2,
            overall_score=0.9,
            regime_match_score=0.8,
            performance_score=0.9,
            confidence_score=0.7,
            risk_score=0.8,
            adaptation_score=0.9,
            selection_reason="strong_performance",
        )

        # Test sorting by overall score
        scores = [score1, score2]
        sorted_scores = sorted(scores, key=lambda x: x.overall_score, reverse=True)

        assert sorted_scores[0].strategy_id == "strategy_2"
        assert sorted_scores[1].strategy_id == "strategy_1"
