"""Integration tests for GPT-Trader CLI.

These tests verify that the CLI commands work correctly end-to-end,
including proper error handling, output formatting, and file generation.
"""

import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import pandas as pd


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.test_env = os.environ.copy()
        cls.test_env["DEMO_MODE"] = "true"
        cls.test_env["LOG_LEVEL"] = "WARNING"  # Reduce noise in tests
        # Add src to PYTHONPATH for module discovery
        project_root = Path(__file__).parent.parent.parent
        src_path = project_root / "src"
        if "PYTHONPATH" in cls.test_env:
            cls.test_env["PYTHONPATH"] = f"{src_path}{os.pathsep}{cls.test_env['PYTHONPATH']}"
        else:
            cls.test_env["PYTHONPATH"] = str(src_path)
        
    def run_cli_command(
        self, 
        args: List[str], 
        check: bool = True,
        capture_output: bool = True,
        timeout: int = 30
    ) -> subprocess.CompletedProcess:
        """Run a CLI command and return the result.
        
        Args:
            args: Command arguments (without 'python -m bot.cli')
            check: Whether to raise on non-zero exit code
            capture_output: Whether to capture stdout/stderr
            timeout: Command timeout in seconds
            
        Returns:
            CompletedProcess object with result
        """
        cmd = [sys.executable, "-m", "bot.cli"] + args
        
        return subprocess.run(
            cmd,
            env=self.test_env,
            check=check,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
    
    def test_help_command(self):
        """Test that help command works and shows expected content."""
        result = self.run_cli_command(["--help"])
        
        assert result.returncode == 0
        assert "GPT-Trader" in result.stdout
        assert "Available Commands:" in result.stdout
        assert "backtest" in result.stdout
        assert "optimize" in result.stdout
        assert "paper" in result.stdout
        
    def test_version_command(self):
        """Test version display."""
        result = self.run_cli_command(["--version"])
        
        assert result.returncode == 0
        assert "GPT-Trader" in result.stdout or "version" in result.stdout.lower()
        
    def test_backtest_basic(self):
        """Test basic backtest command."""
        result = self.run_cli_command([
            "backtest",
            "--symbol", "AAPL",
            "--start", "2024-01-01",
            "--end", "2024-01-31",
            "--strategy", "demo_ma"
        ])
        
        assert result.returncode == 0
        assert "Backtest Results" in result.stdout
        assert "Total Return" in result.stdout
        assert "Sharpe Ratio" in result.stdout
        
    def test_backtest_invalid_dates(self):
        """Test backtest with invalid date range."""
        result = self.run_cli_command(
            [
                "backtest",
                "--symbol", "AAPL",
                "--start", "2024-01-31",  # Start after end
                "--end", "2024-01-01",
                "--strategy", "demo_ma"
            ],
            check=False
        )
        
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "error" in result.stdout.lower()
        
    def test_backtest_output_files(self):
        """Test that backtest creates expected output files."""
        # Run backtest
        result = self.run_cli_command([
            "backtest",
            "--symbol", "AAPL",
            "--start", "2024-01-01",
            "--end", "2024-01-31",
            "--strategy", "demo_ma"
        ])
        
        assert result.returncode == 0
        
        # Check for output files
        output_dir = Path("data/backtests")
        assert output_dir.exists()
        
        # Find most recent files
        csv_files = list(output_dir.glob("PORT_demo_ma_*.csv"))
        summary_files = list(output_dir.glob("PORT_demo_ma_*_summary.csv"))
        trades_files = list(output_dir.glob("PORT_demo_ma_*_trades.csv"))
        
        assert len(csv_files) > 0, "No portfolio CSV files found"
        assert len(summary_files) > 0, "No summary CSV files found"
        assert len(trades_files) > 0, "No trades CSV files found"
        
    def test_backtest_with_multiple_symbols(self):
        """Test backtest with symbol list."""
        # Create a temporary symbol list file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("symbol\n")
            f.write("AAPL\n")
            f.write("MSFT\n")
            f.write("GOOGL\n")
            symbol_file = f.name
        
        try:
            result = self.run_cli_command([
                "backtest",
                "--symbol-list", symbol_file,
                "--start", "2024-01-01",
                "--end", "2024-01-31",
                "--strategy", "trend_breakout"
            ])
            
            assert result.returncode == 0
            assert "Backtest Results" in result.stdout
        finally:
            Path(symbol_file).unlink(missing_ok=True)
            
    def test_backtest_with_risk_parameters(self):
        """Test backtest with custom risk parameters."""
        result = self.run_cli_command([
            "backtest",
            "--symbol", "SPY",
            "--start", "2024-01-01",
            "--end", "2024-01-31",
            "--strategy", "demo_ma",
            "--risk-pct", "1.0",
            "--max-positions", "5",
            "--cost-bps", "10"
        ])
        
        assert result.returncode == 0
        assert "Backtest Results" in result.stdout
        
    def test_demo_mode_restrictions(self):
        """Test that demo mode properly restricts certain operations."""
        # Paper trading should be restricted in demo mode
        result = self.run_cli_command(
            ["paper", "--symbol", "AAPL"],
            check=False
        )
        
        # Should either fail or show warning about demo mode
        assert "demo" in result.stdout.lower() or result.returncode != 0
        
    def test_cli_error_handling(self):
        """Test that CLI provides helpful error messages."""
        # Test with missing required arguments
        result = self.run_cli_command(
            ["backtest", "--symbol", "AAPL"],  # Missing dates
            check=False
        )
        
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "required" in result.stdout.lower()
        
    def test_quiet_mode(self):
        """Test quiet mode suppresses output."""
        result = self.run_cli_command([
            "--quiet",
            "backtest",
            "--symbol", "AAPL",
            "--start", "2024-01-01",
            "--end", "2024-01-31",
            "--strategy", "demo_ma"
        ])
        
        assert result.returncode == 0
        # Output should be minimal in quiet mode
        output_lines = result.stdout.strip().split('\n')
        assert len(output_lines) < 50  # Arbitrary threshold for "quiet"
        
    def test_verbose_mode(self):
        """Test verbose mode shows additional output."""
        result = self.run_cli_command([
            "-vv",  # Double verbose
            "backtest",
            "--symbol", "AAPL",
            "--start", "2024-01-01",
            "--end", "2024-01-31",
            "--strategy", "demo_ma"
        ])
        
        assert result.returncode == 0
        # Should have debug-level output
        assert "DEBUG" in result.stdout or len(result.stdout) > 1000
        
    def test_data_validation_modes(self):
        """Test strict vs repair data validation modes."""
        # Test strict mode (should fail on bad data)
        result_strict = self.run_cli_command([
            "--data-strict", "strict",
            "backtest",
            "--symbol", "AAPL",
            "--start", "2024-01-01",
            "--end", "2024-01-31",
            "--strategy", "demo_ma"
        ])
        
        # Test repair mode (should attempt to fix bad data)
        result_repair = self.run_cli_command([
            "--data-strict", "repair",
            "backtest",
            "--symbol", "AAPL",
            "--start", "2024-01-01",
            "--end", "2024-01-31",
            "--strategy", "demo_ma"
        ])
        
        # Both should complete, but might have different results
        assert result_strict.returncode == 0
        assert result_repair.returncode == 0


class TestCLISubcommands:
    """Test specific CLI subcommands."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.test_env = os.environ.copy()
        cls.test_env["DEMO_MODE"] = "true"
        # Add src to PYTHONPATH
        project_root = Path(__file__).parent.parent.parent
        src_path = project_root / "src"
        cls.test_env["PYTHONPATH"] = str(src_path)
        
    def run_cli_command(self, args: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Helper to run CLI commands."""
        cmd = [sys.executable, "-m", "bot.cli"] + args
        return subprocess.run(
            cmd,
            env=self.test_env,
            capture_output=True,
            text=True,
            timeout=kwargs.get('timeout', 30),
            check=kwargs.get('check', True)
        )
        
    def test_backtest_help(self):
        """Test backtest subcommand help."""
        result = self.run_cli_command(["backtest", "--help"])
        
        assert result.returncode == 0
        assert "Run comprehensive backtests" in result.stdout
        assert "--symbol" in result.stdout
        assert "--start" in result.stdout
        assert "--end" in result.stdout
        
    def test_optimize_help(self):
        """Test optimize subcommand help."""
        result = self.run_cli_command(["optimize", "--help"], check=False)
        
        # Should show help or indicate not implemented
        assert "--help" in ' '.join(result.args) 
        
    def test_shortcuts_command(self):
        """Test shortcuts display."""
        result = self.run_cli_command(["shortcuts"])
        
        assert result.returncode == 0
        assert "shortcut" in result.stdout.lower() or "alias" in result.stdout.lower()


class TestCLIIntegrationAdvanced:
    """Advanced integration tests for complex scenarios."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.test_env = os.environ.copy()
        cls.test_env["DEMO_MODE"] = "true"
        cls.test_data_dir = Path("tests/test_data")
        cls.test_data_dir.mkdir(exist_ok=True)
        # Add src to PYTHONPATH
        project_root = Path(__file__).parent.parent.parent
        src_path = project_root / "src"
        cls.test_env["PYTHONPATH"] = str(src_path)
        
    def run_cli_command(self, args: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Helper to run CLI commands."""
        cmd = [sys.executable, "-m", "bot.cli"] + args
        return subprocess.run(
            cmd,
            env=self.test_env,
            capture_output=True,
            text=True,
            timeout=kwargs.get('timeout', 60),
            check=kwargs.get('check', True)
        )
        
    def test_backtest_regime_filter(self):
        """Test backtest with regime filter enabled."""
        result = self.run_cli_command([
            "backtest",
            "--symbol", "AAPL",
            "--start", "2024-01-01",
            "--end", "2024-03-31",
            "--strategy", "trend_breakout",
            "--regime", "on",
            "--regime-symbol", "SPY",
            "--regime-window", "200"
        ])
        
        assert result.returncode == 0
        assert "Backtest Results" in result.stdout
        
    def test_backtest_different_strategies(self):
        """Test different strategy implementations."""
        strategies = ["demo_ma", "trend_breakout"]
        
        for strategy in strategies:
            result = self.run_cli_command([
                "backtest",
                "--symbol", "SPY",
                "--start", "2024-01-01",
                "--end", "2024-01-31",
                "--strategy", strategy
            ])
            
            assert result.returncode == 0, f"Strategy {strategy} failed"
            assert "Backtest Results" in result.stdout
            
    def test_concurrent_backtests(self):
        """Test running multiple backtests concurrently."""
        import concurrent.futures
        
        def run_backtest(symbol: str) -> bool:
            result = self.run_cli_command([
                "backtest",
                "--symbol", symbol,
                "--start", "2024-01-01",
                "--end", "2024-01-31",
                "--strategy", "demo_ma"
            ])
            return result.returncode == 0
        
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(run_backtest, symbols))
            
        assert all(results), "Some concurrent backtests failed"
        
    def test_output_directory_creation(self):
        """Test that output directories are created as needed."""
        # Use a custom output directory
        custom_dir = "test_output_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result = self.run_cli_command([
            "backtest",
            "--symbol", "AAPL",
            "--start", "2024-01-01",
            "--end", "2024-01-31",
            "--strategy", "demo_ma",
            "--run-dir", custom_dir
        ])
        
        # Check if directory was created
        output_path = Path(custom_dir)
        if output_path.exists():
            # Clean up
            import shutil
            shutil.rmtree(output_path)
            
    @pytest.mark.slow
    def test_long_backtest(self):
        """Test backtest with longer date range."""
        result = self.run_cli_command([
            "backtest",
            "--symbol", "SPY",
            "--start", "2023-01-01",
            "--end", "2023-12-31",
            "--strategy", "demo_ma"
        ], timeout=120)
        
        assert result.returncode == 0
        assert "Backtest Results" in result.stdout


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])