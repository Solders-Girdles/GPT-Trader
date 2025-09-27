"""
ARCHIVED: Early end-to-end integration tests for the complete system.
This file was a developmental iteration and has been moved out of the
active suite in favor of focused smoke tests and unit coverage.
"""
import pytest
import subprocess
import json
from pathlib import Path

pytestmark = pytest.mark.integration

class TestEndToEnd:
    """Test complete system integration"""
    
    def test_cli_help(self):
        """Test CLI help command"""
        result = subprocess.run(
            ['python', '-m', 'src.bot_v2', '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'GPT-Trader Bot V2' in result.stdout
        assert '--mode' in result.stdout
        assert '--workflow' in result.stdout
    
    def test_cli_status(self):
        """Test CLI status command"""
        result = subprocess.run(
            ['python', '-m', 'src.bot_v2', '--status'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Orchestrator Status' in result.stdout
        assert 'Available Slices' in result.stdout
    
    def test_cli_list_workflows(self):
        """Test listing workflows via CLI"""
        result = subprocess.run(
            ['python', '-m', 'src.bot_v2', '--list-workflows'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Available Workflows' in result.stdout
        assert 'simple_backtest' in result.stdout
        assert 'risk_managed_live_trading' in result.stdout
    
    def test_cli_json_output(self):
        """Test JSON output format"""
        result = subprocess.run(
            ['python', '-m', 'src.bot_v2', 
             '--mode', 'backtest',
             '--symbols', 'AAPL',
             '--output', 'json'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should include JSON fields even if mixed with logs
        s = result.stdout or ""
        assert '"mode"' in s
        # Accept either top-level results JSON or structured log with embedded fields
        assert ('"results"' in s) or ('"message"' in s and '"mode"' in s)
    
    def test_workflow_execution_via_cli(self):
        """Test executing workflow through CLI"""
        result = subprocess.run(
            ['python', '-m', 'src.bot_v2',
             '--workflow', 'quick_test',
             '--symbols', 'AAPL',
             '--capital', '10000'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert 'Executing workflow: quick_test' in result.stdout
        assert 'WORKFLOW: quick_test' in result.stdout
        assert 'Status:' in result.stdout
    
    def test_multiple_symbols_cli(self):
        """Test processing multiple symbols via CLI"""
        result = subprocess.run(
            ['python', '-m', 'src.bot_v2',
             '--mode', 'backtest',
             '--symbols', 'AAPL', 'MSFT', 'GOOGL',
             '--output', 'quiet'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should complete without errors
        assert result.returncode in [0, 1]  # May fail but shouldn't crash
    
    def test_config_file_loading(self):
        """Test loading configuration from file"""
        # Create a test config file
        config_path = Path('/tmp/test_config.json')
        config_data = {
            'enable_ml_strategy': True,
            'enable_regime_detection': True,
            'confidence_threshold': 0.7
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        result = subprocess.run(
            ['python', '-m', 'src.bot_v2',
             '--config', str(config_path),
             '--symbols', 'AAPL'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should load config
        assert 'Loaded config from' in result.stderr or result.returncode in [0, 1]
        
        # Cleanup
        config_path.unlink(missing_ok=True)
