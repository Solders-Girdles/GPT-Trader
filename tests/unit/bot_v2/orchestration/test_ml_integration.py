"""
Integration tests for ML pipeline connection
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from bot_v2.orchestration.ml_integration import (
    MLPipelineIntegrator, MLDecision, create_ml_integrator
)
from bot_v2.orchestration.enhanced_orchestrator import (
    EnhancedTradingOrchestrator, create_enhanced_orchestrator
)
from bot_v2.orchestration.types import TradingMode, OrchestratorConfig


class TestMLPipelineIntegrator:
    """Test ML pipeline integration"""
    
    def test_ml_integrator_initialization(self):
        """Test ML integrator initializes correctly"""
        config = {
            'min_confidence': 0.7,
            'max_position_size': 0.15,
            'enable_caching': True
        }
        
        integrator = MLPipelineIntegrator(config)
        
        assert integrator.min_confidence == 0.7
        assert integrator.max_position_size == 0.15
        assert integrator.enable_caching == True
    
    @patch('bot_v2.features.ml_strategy.ml_strategy')
    @patch('bot_v2.features.market_regime.market_regime')
    def test_make_trading_decision(self, mock_regime, mock_strategy):
        """Test complete ML trading decision"""
        
        # Mock regime detection
        mock_regime_analysis = Mock()
        mock_regime_analysis.current_regime.value = 'BULL_QUIET'
        mock_regime_analysis.confidence = 0.8
        mock_regime.detect_regime.return_value = mock_regime_analysis
        
        # Mock strategy prediction
        mock_prediction = Mock()
        mock_prediction.strategy.value = 'momentum'
        mock_prediction.confidence = 0.75
        mock_prediction.expected_return = 12.5
        mock_strategy.predict_best_strategy.return_value = [mock_prediction]
        
        # Create integrator
        integrator = MLPipelineIntegrator({'min_confidence': 0.6})
        integrator.ml_strategy = mock_strategy
        integrator.market_regime = mock_regime
        
        # Make decision
        decision = integrator.make_trading_decision(
            symbol='AAPL',
            portfolio_value=10000,
            current_positions={}
        )
        
        assert decision.symbol == 'AAPL'
        assert decision.strategy == 'momentum'
        assert decision.confidence == 0.75
        assert decision.regime == 'BULL_QUIET'
        assert decision.decision in ['buy', 'sell', 'hold']
        assert len(decision.reasoning) > 0
    
    def test_confidence_filtering(self):
        """Test that low confidence results in hold decision"""
        integrator = MLPipelineIntegrator({'min_confidence': 0.7})
        
        # Mock low confidence prediction
        with patch.object(integrator, '_get_strategy_prediction') as mock_pred:
            mock_pred.return_value = ('momentum', 0.5, 0.1)  # Low confidence
            
            with patch.object(integrator, '_get_market_regime') as mock_regime:
                mock_regime.return_value = ('BULL_QUIET', 0.8)
                
                decision = integrator.make_trading_decision(
                    'AAPL', 10000, {}
                )
                
                assert decision.decision == 'hold'
                assert 'below threshold' in ' '.join(decision.reasoning)
    
    def test_position_sizing_with_regime(self):
        """Test position sizing adjusts based on regime"""
        integrator = MLPipelineIntegrator({'max_position_size': 0.2})
        
        # Test bull regime
        bull_size = integrator._calculate_ml_position_size(
            confidence=0.8,
            expected_return=0.15,
            regime='BULL_QUIET',
            portfolio_value=10000
        )
        
        # Test crisis regime
        crisis_size = integrator._calculate_ml_position_size(
            confidence=0.8,
            expected_return=0.15,
            regime='CRISIS',
            portfolio_value=10000
        )
        
        assert bull_size > crisis_size
        assert bull_size <= 0.2  # Max position size
    
    def test_caching_functionality(self):
        """Test ML prediction caching"""
        integrator = MLPipelineIntegrator({
            'enable_caching': True,
            'cache_ttl_minutes': 5
        })
        
        # Cache a prediction
        integrator._cache_prediction('test_key', ('strategy', 0.8, 0.1))
        
        # Check cache is valid
        assert integrator._is_cache_valid('test_key') == True
        assert integrator._prediction_cache['test_key'] == ('strategy', 0.8, 0.1)
        
        # Clear cache
        integrator.clear_cache()
        assert len(integrator._prediction_cache) == 0


class TestEnhancedOrchestrator:
    """Test enhanced orchestrator with ML integration"""
    
    def test_enhanced_orchestrator_initialization(self):
        """Test enhanced orchestrator initializes with ML"""
        config = OrchestratorConfig(
            mode=TradingMode.BACKTEST,
            capital=10000,
            min_confidence=0.65
        )
        
        orchestrator = EnhancedTradingOrchestrator(config)
        
        assert orchestrator.ml_integrator is not None
        assert orchestrator.ml_integrator.min_confidence == 0.65
        assert len(orchestrator.ml_decisions_history) == 0
    
    @patch('bot_v2.orchestration.enhanced_orchestrator.MLPipelineIntegrator')
    def test_execute_ml_trading_cycle(self, mock_integrator_class):
        """Test ML trading cycle execution"""
        
        # Mock ML decisions
        mock_decision = MLDecision(
            symbol='AAPL',
            strategy='momentum',
            confidence=0.8,
            expected_return=0.12,
            regime='BULL_QUIET',
            regime_confidence=0.75,
            position_size=0.1,
            risk_adjusted_size=0.08,
            decision='buy',
            reasoning=['Test reasoning'],
            timestamp=datetime.now()
        )
        
        mock_integrator = Mock()
        mock_integrator.get_portfolio_ml_decisions.return_value = [mock_decision]
        mock_integrator_class.return_value = mock_integrator
        
        # Create orchestrator
        config = OrchestratorConfig(mode=TradingMode.BACKTEST)
        orchestrator = EnhancedTradingOrchestrator(config)
        orchestrator.ml_integrator = mock_integrator
        
        # Mock backtest
        orchestrator.backtest = Mock()
        orchestrator.backtest.run_backtest.return_value = Mock(metrics={'return': 0.1})
        
        # Execute cycle
        results = orchestrator.execute_ml_trading_cycle(
            symbols=['AAPL'],
            current_positions={}
        )
        
        assert 'AAPL' in results
        assert results['AAPL'].success == True
        assert orchestrator.ml_performance_metrics['total_decisions'] == 1
        assert orchestrator.ml_performance_metrics['buy_signals'] == 1
    
    def test_ml_performance_report(self):
        """Test ML performance reporting"""
        orchestrator = EnhancedTradingOrchestrator()
        
        # Add some test decisions
        test_decision = MLDecision(
            symbol='AAPL',
            strategy='momentum',
            confidence=0.75,
            expected_return=0.1,
            regime='BULL_QUIET',
            regime_confidence=0.8,
            position_size=0.05,
            risk_adjusted_size=0.04,
            decision='buy',
            reasoning=['test'],
            timestamp=datetime.now()
        )
        
        orchestrator.ml_decisions_history.append(test_decision)
        orchestrator._update_ml_metrics(test_decision)
        
        # Get report
        report = orchestrator.get_ml_performance_report()
        
        assert 'metrics' in report
        assert report['metrics']['total_decisions'] == 1
        assert report['metrics']['buy_signals'] == 1
        assert 'recent_decisions' in report
        assert len(report['recent_decisions']) == 1
        assert report['recent_decisions'][0]['symbol'] == 'AAPL'
    
    def test_ml_decision_execution_modes(self):
        """Test ML decision execution in different modes"""
        
        # Test backtest mode
        config = OrchestratorConfig(mode=TradingMode.BACKTEST)
        orchestrator = EnhancedTradingOrchestrator(config)
        
        test_decision = MLDecision(
            symbol='AAPL',
            strategy='momentum',
            confidence=0.8,
            expected_return=0.1,
            regime='BULL_QUIET',
            regime_confidence=0.75,
            position_size=0.1,
            risk_adjusted_size=0.08,
            decision='buy',
            reasoning=['test'],
            timestamp=datetime.now()
        )
        
        # Mock backtest
        orchestrator.backtest = Mock()
        orchestrator.backtest.run_backtest.return_value = Mock(metrics={})
        
        result = orchestrator._execute_ml_decision(test_decision, 10000)
        
        assert result.mode == TradingMode.BACKTEST
        
        # Test paper mode
        config.mode = TradingMode.PAPER
        orchestrator = EnhancedTradingOrchestrator(config)
        orchestrator.paper_trade = Mock()
        orchestrator.paper_trade.execute_paper_trade.return_value = {}
        
        result = orchestrator._execute_ml_decision(test_decision, 10000)
        
        assert result.mode == TradingMode.PAPER
    
    def test_risk_adjustments(self):
        """Test risk adjustments in position sizing"""
        integrator = MLPipelineIntegrator()
        
        # Test with high exposure
        current_positions = {
            'AAPL': 0.3,
            'GOOGL': 0.3,
            'MSFT': 0.25
        }  # 85% invested
        
        adjusted_size = integrator._apply_risk_adjustments(
            base_size=0.1,
            symbol='TSLA',
            regime='BULL_QUIET',
            current_positions=current_positions
        )
        
        # Should reduce size due to high exposure
        assert adjusted_size < 0.1
    
    def test_cache_clearing(self):
        """Test ML cache clearing functionality"""
        orchestrator = EnhancedTradingOrchestrator()
        
        # Add something to cache
        orchestrator.ml_integrator._cache_prediction('test', 'value')
        assert len(orchestrator.ml_integrator._prediction_cache) > 0
        
        # Clear cache
        orchestrator.clear_ml_cache()
        assert len(orchestrator.ml_integrator._prediction_cache) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
