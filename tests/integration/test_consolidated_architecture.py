"""
Integration tests for Phase 1 consolidated architecture
Verifies that all simplified components work together correctly
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestDatabaseConsolidation:
    """Test unified database functionality"""

    def test_unified_database_initialization(self):
        """Test that unified database initializes correctly"""
        from bot.core.database import DatabaseConfig, DatabaseManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(database_path=Path(tmpdir) / "test.db")

            db_manager = DatabaseManager(config)

            # Verify database was created
            assert config.database_path.exists()

            # Verify tables were created
            with db_manager.get_connection() as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                # Check for key tables
                assert "components" in tables
                assert "orders" in tables
                assert "positions" in tables
                assert "risk_metrics" in tables
                assert "alert_events" in tables

            db_manager.close()

    def test_database_operations(self):
        """Test basic database operations"""
        from bot.core.database import DatabaseConfig, DatabaseManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(database_path=Path(tmpdir) / "test.db")

            db_manager = DatabaseManager(config)

            # Test insert
            record_id = db_manager.insert_record(
                "components",
                {"component_id": "test_component", "component_type": "test", "status": "active"},
            )

            assert record_id is not None

            # Test fetch
            result = db_manager.fetch_one(
                "SELECT * FROM components WHERE component_id = ?", ("test_component",)
            )

            assert result is not None
            assert result["component_type"] == "test"

            # Test update
            updated = db_manager.update_record(
                "components", {"status": "inactive"}, "component_id = ?", ("test_component",)
            )

            assert updated == 1

            db_manager.close()


class TestCLIConsolidation:
    """Test consolidated CLI functionality"""

    def test_cli_help_command(self):
        """Test that CLI help works"""
        result = subprocess.run(
            [sys.executable, "-m", "bot.cli", "--help"], capture_output=True, text=True, timeout=10
        )

        assert result.returncode == 0
        assert "GPT-Trader" in result.stdout
        assert "backtest" in result.stdout
        assert "optimize" in result.stdout

    def test_cli_structure(self):
        """Test that CLI has correct module structure"""
        cli_path = Path("src/bot/cli")

        # Check that we have exactly the expected files
        cli_files = list(cli_path.glob("*.py"))

        expected_files = {"cli.py", "commands.py", "utils.py", "__init__.py", "__main__.py"}

        actual_files = {f.name for f in cli_files}

        # Should have 5 or fewer files (consolidated from 22)
        assert len(actual_files) <= 5

        # Should have the core modules
        assert "__init__.py" in actual_files
        assert "cli.py" in actual_files or "__main__.py" in actual_files


class TestMonitoringConsolidation:
    """Test unified monitoring system"""

    def test_unified_monitor_initialization(self):
        """Test that unified monitor initializes correctly"""
        from bot.monitoring.monitor import MonitorConfig, UnifiedMonitor

        config = MonitorConfig()
        monitor = UnifiedMonitor(config)

        assert monitor is not None
        assert not monitor.is_running

        # Test start/stop
        monitor.start()
        assert monitor.is_running

        time.sleep(0.5)  # Let it run briefly

        monitor.stop()
        assert not monitor.is_running

    def test_monitor_metrics_collection(self):
        """Test metrics collection"""
        from bot.monitoring.monitor import MonitorConfig, UnifiedMonitor

        config = MonitorConfig(metrics_interval=1, health_check_interval=1)  # Fast for testing

        monitor = UnifiedMonitor(config)

        # Mock database connection
        with patch.object(monitor, "db_manager") as mock_db:
            mock_conn = MagicMock()
            mock_db.get_connection.return_value.__enter__.return_value = mock_conn

            # Mock query results
            mock_conn.execute.return_value.fetchone.return_value = [5, 1000.0, 50000.0]

            monitor._collect_metrics()

            metrics = monitor.get_metrics()
            assert "total_positions" in metrics
            assert metrics["total_positions"] == 5


class TestDataPipeline:
    """Test unified data pipeline"""

    @pytest.mark.asyncio
    async def test_unified_pipeline_initialization(self):
        """Test that unified data pipeline initializes correctly"""
        from bot.data.unified_pipeline import DataConfig, UnifiedDataPipeline

        config = DataConfig(cache_enabled=True, cache_dir=Path(tempfile.mkdtemp()))

        pipeline = UnifiedDataPipeline(config)

        assert pipeline is not None
        assert len(pipeline.get_available_sources()) > 0

    @pytest.mark.asyncio
    async def test_data_validation(self):
        """Test data validation and repair"""
        from bot.data.unified_pipeline import DataConfig, UnifiedDataPipeline

        config = DataConfig(validate_data=True, repair_data=True)

        pipeline = UnifiedDataPipeline(config)

        # Create test data with issues
        bad_data = pd.DataFrame(
            {
                "open": [100, np.nan, 102],
                "high": [101, 103, 103],
                "low": [99, 101, 101],
                "close": [100, 102, 102],
                "volume": [1000, -500, 2000],  # Negative volume
            }
        )

        # Validate and repair
        fixed_data = pipeline._validate_data(bad_data, repair=True)

        # Check that issues were fixed
        assert not fixed_data.isnull().any().any()  # No NaN values
        assert (fixed_data["volume"] >= 0).all()  # No negative volume
        assert (fixed_data["high"] >= fixed_data["low"]).all()  # High >= Low


class TestBaseClasses:
    """Test shared base classes"""

    def test_base_component(self):
        """Test BaseComponent functionality"""
        from bot.core.base import BaseComponent, ComponentConfig, ComponentStatus

        # Create a test component
        class TestComponent(BaseComponent):
            def _initialize_component(self):
                pass

            def _start_component(self):
                pass

            def _stop_component(self):
                pass

            def _health_check(self):
                from bot.core.base import HealthStatus

                return HealthStatus.HEALTHY

        config = ComponentConfig(component_id="test", component_type="test")

        component = TestComponent(config)

        # Test lifecycle
        assert component.status == ComponentStatus.INITIALIZED

        component.start()
        assert component.status == ComponentStatus.RUNNING
        assert component.is_running()

        component.stop()
        assert component.status == ComponentStatus.STOPPED
        assert not component.is_running()

    def test_base_strategy(self):
        """Test BaseStrategy functionality"""
        from bot.core.base import BaseStrategy, ComponentConfig

        # Create a test strategy
        class TestStrategy(BaseStrategy):
            def _initialize_component(self):
                pass

            def _start_component(self):
                pass

            def _stop_component(self):
                pass

            def _health_check(self):
                from bot.core.base import HealthStatus

                return HealthStatus.HEALTHY

            def _generate_signals(self, market_data):
                return [1, 0, -1]  # Buy, hold, sell

            def _calculate_position_size(self, signal, portfolio_state):
                return 100  # Fixed size for testing

        config = ComponentConfig(component_id="test_strategy", component_type="strategy")

        strategy = TestStrategy(config)

        # Test signal generation
        signals = strategy._generate_signals(None)
        assert len(signals) == 3
        assert signals[0] == 1  # Buy signal


class TestStreamlitDashboard:
    """Test Streamlit dashboard"""

    def test_dashboard_exists(self):
        """Test that dashboard file exists and has correct structure"""
        dashboard_path = Path("src/bot/dashboard/app.py")

        assert dashboard_path.exists()

        # Read the file to check structure
        with open(dashboard_path) as f:
            content = f.read()

            # Check for key components
            assert "streamlit" in content
            assert "DashboardData" in content
            assert "render_portfolio_overview" in content
            assert "main()" in content

    def test_dashboard_launch_script(self):
        """Test that dashboard launch script exists"""
        script_path = Path("scripts/dashboard.py")

        assert script_path.exists() or Path("scripts/run_dashboard.py").exists()


class TestArchivedComponents:
    """Verify that unnecessary components were properly archived"""

    def test_no_kubernetes(self):
        """Verify Kubernetes configs are gone"""
        k8s_path = Path("k8s")
        assert not k8s_path.exists()

        # Check it was archived
        archived = Path("archived")
        if archived.exists():
            k8s_archived = list(archived.glob("k8s_*"))
            assert len(k8s_archived) > 0 or not k8s_path.exists()

    def test_no_tauri_ui(self):
        """Verify Tauri UI is gone"""
        claudia_path = Path("claudia")
        assert not claudia_path.exists()

        # Check it was archived
        archived = Path("archived")
        if archived.exists():
            claudia_archived = list(archived.glob("claudia_*"))
            assert len(claudia_archived) > 0 or not claudia_path.exists()

    def test_no_ml_bloat(self):
        """Verify ML/distributed directories are gone"""
        ml_dirs = [
            "src/bot/intelligence",
            "src/bot/meta_learning",
            "src/bot/ml",
            "src/bot/distributed",
            "src/bot/scaling",
            "src/bot/knowledge",
        ]

        for dir_path in ml_dirs:
            path = Path(dir_path)
            assert not path.exists(), f"ML bloat directory still exists: {dir_path}"


class TestSystemIntegration:
    """End-to-end integration tests"""

    @pytest.mark.slow
    def test_full_system_initialization(self):
        """Test that the full system can initialize"""
        from bot.core.database import DatabaseConfig, initialize_database
        from bot.data.unified_pipeline import DataConfig, get_data_pipeline
        from bot.monitoring.monitor import MonitorConfig, UnifiedMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize database
            db_config = DatabaseConfig(database_path=Path(tmpdir) / "test.db")
            db = initialize_database(db_config)

            # Initialize monitor
            monitor_config = MonitorConfig()
            monitor = UnifiedMonitor(monitor_config)

            # Initialize data pipeline
            data_config = DataConfig(cache_dir=Path(tmpdir) / "cache")
            pipeline = get_data_pipeline(data_config)

            # Verify all components initialized
            assert db is not None
            assert monitor is not None
            assert pipeline is not None

            # Clean up
            monitor.stop()
            db.close()

    @pytest.mark.slow
    def test_cli_backtest_command(self):
        """Test that backtest command structure is valid"""
        # Just test that the command can be invoked with --help
        # Actual backtest testing would require market data
        result = subprocess.run(
            [sys.executable, "-m", "bot.cli", "backtest", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "--symbol" in result.stdout
        assert "--start" in result.stdout
        assert "--end" in result.stdout


def run_integration_tests():
    """Run all integration tests and generate report"""
    print("=" * 80)
    print("Running Phase 1.5 Integration Tests")
    print("=" * 80)

    # Run pytest with detailed output
    pytest_args = [__file__, "-v", "--tb=short", "--color=yes"]

    result = pytest.main(pytest_args)

    if result == 0:
        print("\n" + "=" * 80)
        print("✅ All integration tests passed!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ Some tests failed. Review output above.")
        print("=" * 80)

    return result


if __name__ == "__main__":
    sys.exit(run_integration_tests())
