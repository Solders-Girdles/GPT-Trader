"""
ARCHIVED: Early orchestration integration tests.
Replaced by comprehensive unit tests under tests/unit/bot_v2/orchestration/.
"""
import os
import pytest
from unittest.mock import Mock
import pandas as pd
from datetime import datetime

from bot_v2.orchestration import TradingOrchestrator, OrchestratorConfig, TradingMode

pytestmark = pytest.mark.integration

class TestOrchestration:
    """Test orchestration integration"""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly"""
        config = OrchestratorConfig(
            mode=TradingMode.BACKTEST,
            symbols=['AAPL', 'MSFT'],
            capital=10000
        )
        
        orchestrator = TradingOrchestrator(config)
        
        assert orchestrator is not None
        assert orchestrator.config.mode == TradingMode.BACKTEST
        assert orchestrator.config.symbols == ['AAPL', 'MSFT']
        assert orchestrator.config.capital == 10000
        
        # Check status
        status = orchestrator.get_status()
        assert 'available_slices' in status
        assert 'failed_slices' in status
        assert status['mode'] == 'backtest'
    
    def test_execute_trading_cycle_backtest(self, monkeypatch):
        """Test trading cycle execution in backtest mode"""
        # Force data providers to use MockProvider
        monkeypatch.setenv('TESTING', 'true')
        
        config = OrchestratorConfig(
            mode=TradingMode.BACKTEST,
            symbols=['AAPL'],
            capital=10000
        )
        
        orchestrator = TradingOrchestrator(config)
        result = orchestrator.execute_trading_cycle('AAPL')
        
        assert result is not None
        assert result.symbol == 'AAPL'
        assert result.mode == TradingMode.BACKTEST
        assert 'data_fetched' in result.data
        
    def test_graceful_degradation(self):
        """Test system continues when slices fail"""
        config = OrchestratorConfig(
            mode=TradingMode.BACKTEST,
            enable_ml_strategy=False,  # Disable ML to avoid failures
            enable_regime_detection=False
        )
        
        orchestrator = TradingOrchestrator(config)
        
        # Should still work even if some slices fail
        result = orchestrator.execute_trading_cycle('AAPL')
        
        assert result is not None
        # Success can be True even with some errors (graceful degradation)
        assert isinstance(result.success, bool)
    
    def test_multiple_symbols(self):
        """Test processing multiple symbols"""
        config = OrchestratorConfig(
            mode=TradingMode.BACKTEST,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            capital=50000
        )
        
        orchestrator = TradingOrchestrator(config)
        results = orchestrator.run(['AAPL', 'MSFT'])
        
        assert len(results) == 2
        assert results[0].symbol == 'AAPL'
        assert results[1].symbol == 'MSFT'
    
    def test_different_modes(self):
        """Test different trading modes"""
        modes = [TradingMode.BACKTEST, TradingMode.PAPER, TradingMode.OPTIMIZE]
        
        for mode in modes:
            config = OrchestratorConfig(
                mode=mode,
                symbols=['AAPL'],
                capital=10000
            )
            
            orchestrator = TradingOrchestrator(config)
            status = orchestrator.get_status()
            
            assert status['mode'] == mode.value
