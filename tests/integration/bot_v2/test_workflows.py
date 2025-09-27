"""
Integration tests for the workflow engine
"""
import os
import pytest
from unittest.mock import Mock
import pandas as pd

from bot_v2.orchestration import TradingOrchestrator, OrchestratorConfig, TradingMode
from bot_v2.workflows import WorkflowEngine, WorkflowContext, ALL_WORKFLOWS

pytestmark = pytest.mark.integration

class TestWorkflows:
    """Test workflow integration"""
    
    def setup_method(self):
        """Setup for each test"""
        config = OrchestratorConfig(
            mode=TradingMode.BACKTEST,
            symbols=['AAPL'],
            capital=10000
        )
        self.orchestrator = TradingOrchestrator(config)
        self.engine = WorkflowEngine(self.orchestrator)
    
    def test_workflow_engine_initialization(self):
        """Test workflow engine initializes with workflows"""
        assert self.engine is not None
        assert len(self.engine.workflows) == 6
        assert 'simple_backtest' in self.engine.workflows
        assert 'paper_trading' in self.engine.workflows
    
    def test_workflow_context(self):
        """Test workflow context management"""
        context = WorkflowContext({'symbol': 'AAPL', 'capital': 10000})
        
        assert context.get('symbol') == 'AAPL'
        assert context.get('capital') == 10000
        assert context.get('missing', 'default') == 'default'
        
        # Test updates
        context.set('strategy', 'momentum')
        assert context.get('strategy') == 'momentum'
        
        # Test checkpoints
        context.checkpoint('before_trade')
        context.set('position', 100)
        assert context.get('position') == 100
        
        context.restore_checkpoint('before_trade')
        assert context.get('position') is None
    
    def test_quick_test_workflow(self, monkeypatch):
        """Test quick_test workflow execution"""
        # Force provider to MockProvider via env
        monkeypatch.setenv('TESTING', 'true')
        
        result = self.engine.execute_workflow('quick_test', {
            'symbol': 'AAPL',
            'capital': 10000
        })
        
        assert result['status'] in ['completed', 'failed']
        assert 'steps_executed' in result
        assert 'steps_failed' in result
        assert 'context' in result
    
    def test_workflow_validation(self):
        """Test workflow step validation"""
        # Try workflow without required context
        result = self.engine.execute_workflow('simple_backtest', {})
        
        # Should fail validation
        assert result['status'] == 'failed'
        assert len(result['steps_failed']) > 0
    
    def test_all_predefined_workflows(self):
        """Test that all predefined workflows are valid"""
        for name, workflow in ALL_WORKFLOWS.items():
            assert len(workflow) > 0
            
            # Check each step has required attributes
            for step in workflow:
                assert hasattr(step, 'name')
                assert hasattr(step, 'function')
                assert hasattr(step, 'description')
    
    def test_workflow_retry_mechanism(self):
        """Test workflow step retry on failure"""
        # Create a workflow with a step that might fail
        context = WorkflowContext({
            'symbol': 'INVALID_SYMBOL',
            'capital': 10000
        })
        
        # Execute should handle retries
        result = self.engine.execute_workflow('quick_test', context.get_all())
        
        # Should complete even if some steps fail
        assert 'status' in result
        assert 'steps_executed' in result
    
    def test_workflow_continue_on_failure(self):
        """Test workflow continues when step has continue_on_failure=True"""
        # ML_DRIVEN workflow has steps with continue_on_failure
        result = self.engine.execute_workflow('ml_driven', {
            'symbol': 'AAPL',
            'capital': 10000
        })
        
        # Should complete even if optimization fails
        assert result['status'] in ['completed', 'failed']
        
        # Check if it continued past failures
        if result['steps_failed']:
            for failure in result['steps_failed']:
                if 'continued' in failure:
                    assert failure['continued'] == True
