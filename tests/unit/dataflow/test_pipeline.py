"""Tests for the unified data pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from bot.dataflow.pipeline import DataPipeline, DataQualityMetrics, PipelineConfig


class TestDataQualityMetrics:
    """Test data quality metrics calculations."""

    def test_success_rate_calculation(self):
        metrics = DataQualityMetrics()
        metrics.total_symbols_requested = 10
        metrics.symbols_loaded_successfully = 8

        assert metrics.success_rate == 80.0

    def test_success_rate_zero_requests(self):
        metrics = DataQualityMetrics()
        assert metrics.success_rate == 0.0

    def test_cache_hit_rate_calculation(self):
        metrics = DataQualityMetrics()
        metrics.cache_hits = 7
        metrics.cache_misses = 3

        assert metrics.cache_hit_rate == 70.0

    def test_cache_hit_rate_no_requests(self):
        metrics = DataQualityMetrics()
        assert metrics.cache_hit_rate == 0.0

    def test_to_dict(self):
        metrics = DataQualityMetrics(
            total_symbols_requested=5,
            symbols_loaded_successfully=4,
            symbols_failed=1,
            cache_hits=2,
            cache_misses=3,
            validation_errors=1,
            adjustment_applied=2,
            avg_load_time_ms=150.5,
            errors=["Error 1", "Error 2"],
        )

        result = metrics.to_dict()

        expected = {
            "total_requested": 5,
            "loaded_successfully": 4,
            "failed": 1,
            "success_rate_pct": 80.0,
            "cache_hit_rate_pct": 40.0,
            "validation_errors": 1,
            "adjustments_applied": 2,
            "avg_load_time_ms": 150.5,
            "error_count": 2,
        }

        assert result == expected


class TestPipelineConfig:
    """Test pipeline configuration."""

    def test_default_config(self):
        config = PipelineConfig()

        assert config.use_cache is True
        assert config.cache_ttl_hours == 24
        assert config.strict_validation is True
        assert config.min_data_points == 10
        assert config.timeout_seconds == 30.0
        assert config.retry_attempts == 3
        assert config.apply_adjustments is True
        assert config.fail_on_missing_symbols is False
        assert config.max_missing_data_pct == 10.0

    def test_custom_config(self):
        config = PipelineConfig(
            use_cache=False, cache_ttl_hours=12, strict_validation=False, min_data_points=5
        )

        assert config.use_cache is False
        assert config.cache_ttl_hours == 12
        assert config.strict_validation is False
        assert config.min_data_points == 5


class TestDataPipeline:
    """Test the main data pipeline functionality."""

    @pytest.fixture
    def mock_yfinance_source(self):
        """Mock YFinance source."""
        mock_source = Mock()
        return mock_source

    @pytest.fixture
    def sample_dataframe(self):
        """Sample OHLCV DataFrame for testing."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "Open": [100.0 + i for i in range(30)],
                "High": [105.0 + i for i in range(30)],
                "Low": [95.0 + i for i in range(30)],
                "Close": [102.0 + i for i in range(30)],
                "Volume": [1000 + i * 100 for i in range(30)],
            },
            index=dates,
        )

    @pytest.fixture
    def pipeline(self, mock_yfinance_source):
        """Create pipeline with mocked source."""
        pipeline = DataPipeline()
        pipeline.source = mock_yfinance_source
        return pipeline

    def test_pipeline_initialization(self):
        """Test pipeline initialization with default config."""
        pipeline = DataPipeline()

        assert isinstance(pipeline.config, PipelineConfig)
        assert pipeline.source is not None
        assert pipeline._cache == {}
        assert isinstance(pipeline.metrics, DataQualityMetrics)

    def test_pipeline_initialization_with_config(self):
        """Test pipeline initialization with custom config."""
        config = PipelineConfig(use_cache=False, strict_validation=False)
        pipeline = DataPipeline(config)

        assert pipeline.config == config
        assert pipeline.config.use_cache is False
        assert pipeline.config.strict_validation is False

    def test_validate_symbols_valid_input(self, pipeline):
        """Test symbol validation with valid input."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        result = pipeline._validate_symbols(symbols)

        assert result == ["AAPL", "GOOGL", "MSFT"]

    def test_validate_symbols_empty_input(self, pipeline):
        """Test symbol validation with empty input."""
        with pytest.raises(ValueError, match="No symbols provided"):
            pipeline._validate_symbols([])

    def test_validate_symbols_invalid_symbols(self, pipeline):
        """Test symbol validation with invalid symbols."""
        with pytest.raises(ValueError):
            pipeline._validate_symbols(["INVALID!@#", ""])

    def test_validate_date_range_valid(self, pipeline):
        """Test date range validation with valid dates."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)

        result_start, result_end = pipeline._validate_date_range(start, end)

        assert result_start == start
        assert result_end == end

    def test_validate_date_range_invalid_order(self, pipeline):
        """Test date range validation with invalid order."""
        start = datetime(2023, 12, 31)
        end = datetime(2023, 1, 1)

        with pytest.raises(ValueError, match="Start date .* must be before end date"):
            pipeline._validate_date_range(start, end)

    def test_validate_date_range_future_date(self, pipeline):
        """Test date range validation with future start date."""
        start = datetime.now() + timedelta(days=30)
        end = start + timedelta(days=30)

        with pytest.raises(ValueError, match="Start date .* cannot be in the future"):
            pipeline._validate_date_range(start, end)

    @patch("bot.dataflow.pipeline.validate_daily_bars")
    @patch("bot.dataflow.pipeline.adjust_to_adjclose")
    def test_fetch_symbol_data_success(
        self, mock_adjust, mock_validate, pipeline, sample_dataframe
    ):
        """Test successful symbol data fetching."""
        # Setup mocks
        pipeline.source.get_daily_bars.return_value = sample_dataframe
        mock_adjust.return_value = (sample_dataframe, False)
        mock_validate.return_value = None

        # Test
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)

        result = pipeline._fetch_symbol_data("AAPL", start_date, end_date, use_cache=False)

        # Assertions
        assert result is not None
        assert len(result) == 30
        assert pipeline.source.get_daily_bars.called
        mock_adjust.assert_called_once()
        mock_validate.assert_called_once()

    def test_fetch_symbol_data_empty_response(self, pipeline):
        """Test handling of empty data response."""
        pipeline.source.get_daily_bars.return_value = pd.DataFrame()

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)

        result = pipeline._fetch_symbol_data("AAPL", start_date, end_date, use_cache=False)

        assert result is None

    @patch("bot.dataflow.pipeline.validate_daily_bars")
    @patch("bot.dataflow.pipeline.adjust_to_adjclose")
    def test_fetch_symbol_data_with_cache(
        self, mock_adjust, mock_validate, pipeline, sample_dataframe
    ):
        """Test data fetching with caching enabled."""
        # Setup mocks
        pipeline.source.get_daily_bars.return_value = sample_dataframe
        mock_adjust.return_value = (sample_dataframe, False)
        mock_validate.return_value = None

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)

        # First call should fetch and cache
        result1 = pipeline._fetch_symbol_data("AAPL", start_date, end_date, use_cache=True)
        assert result1 is not None
        assert pipeline.source.get_daily_bars.call_count == 1

        # Second call should use cache
        result2 = pipeline._fetch_symbol_data("AAPL", start_date, end_date, use_cache=True)
        assert result2 is not None
        assert pipeline.source.get_daily_bars.call_count == 1  # Still 1, not 2

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    @patch("bot.dataflow.pipeline.validate_daily_bars")
    @patch("bot.dataflow.pipeline.adjust_to_adjclose")
    def test_fetch_and_validate_single_symbol(
        self, mock_adjust, mock_validate, pipeline, sample_dataframe
    ):
        """Test fetch_and_validate with single symbol."""
        # Setup mocks
        pipeline.source.get_daily_bars.return_value = sample_dataframe
        mock_adjust.return_value = (sample_dataframe, False)
        mock_validate.return_value = None

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)

        result = pipeline.fetch_and_validate(["AAPL"], start_date, end_date)

        assert len(result) == 1
        assert "AAPL" in result
        assert len(result["AAPL"]) == 30

        # Check metrics
        assert pipeline.metrics.total_symbols_requested == 1
        assert pipeline.metrics.symbols_loaded_successfully == 1
        assert pipeline.metrics.symbols_failed == 0

    @patch("bot.dataflow.pipeline.validate_daily_bars")
    @patch("bot.dataflow.pipeline.adjust_to_adjclose")
    def test_fetch_and_validate_multiple_symbols(
        self, mock_adjust, mock_validate, pipeline, sample_dataframe
    ):
        """Test fetch_and_validate with multiple symbols."""
        # Setup mocks
        pipeline.source.get_daily_bars.return_value = sample_dataframe
        mock_adjust.return_value = (sample_dataframe, False)
        mock_validate.return_value = None

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        symbols = ["AAPL", "GOOGL", "MSFT"]

        result = pipeline.fetch_and_validate(symbols, start_date, end_date)

        assert len(result) == 3
        for symbol in symbols:
            assert symbol in result
            assert len(result[symbol]) == 30

        # Check metrics
        assert pipeline.metrics.total_symbols_requested == 3
        assert pipeline.metrics.symbols_loaded_successfully == 3
        assert pipeline.metrics.symbols_failed == 0

    def test_fetch_and_validate_partial_failure(self, pipeline, sample_dataframe):
        """Test fetch_and_validate with some symbols failing."""

        def mock_get_daily_bars(symbol, start, end):
            if symbol == "AAPL":
                return sample_dataframe
            elif symbol == "GOOGL":
                return pd.DataFrame()  # Empty response
            else:
                raise Exception("Network error")

        pipeline.source.get_daily_bars.side_effect = mock_get_daily_bars

        with patch("bot.dataflow.pipeline.validate_daily_bars"):
            with patch("bot.dataflow.pipeline.adjust_to_adjclose") as mock_adjust:
                mock_adjust.return_value = (sample_dataframe, False)

                start_date = datetime(2023, 1, 1)
                end_date = datetime(2023, 1, 31)
                symbols = ["AAPL", "GOOGL", "MSFT"]

                result = pipeline.fetch_and_validate(symbols, start_date, end_date)

                # Only AAPL should succeed
                assert len(result) == 1
                assert "AAPL" in result
                assert "GOOGL" not in result
                assert "MSFT" not in result

                # Check metrics
                assert pipeline.metrics.total_symbols_requested == 3
                assert pipeline.metrics.symbols_loaded_successfully == 1
                assert pipeline.metrics.symbols_failed == 2

    def test_fetch_and_validate_all_fail(self, pipeline):
        """Test fetch_and_validate when all symbols fail."""
        pipeline.source.get_daily_bars.side_effect = Exception("Network error")

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        symbols = ["AAPL", "GOOGL"]

        with pytest.raises(ValueError, match="No data could be loaded for any symbol"):
            pipeline.fetch_and_validate(symbols, start_date, end_date)

    def test_fetch_and_validate_fail_on_missing_enabled(self, pipeline, sample_dataframe):
        """Test fetch_and_validate with fail_on_missing_symbols enabled."""
        # Configure to fail on missing symbols
        pipeline.config.fail_on_missing_symbols = True

        def mock_get_daily_bars(symbol, start, end):
            if symbol == "AAPL":
                return sample_dataframe
            else:
                return pd.DataFrame()  # Empty response for others

        pipeline.source.get_daily_bars.side_effect = mock_get_daily_bars

        with patch("bot.dataflow.pipeline.validate_daily_bars"):
            with patch("bot.dataflow.pipeline.adjust_to_adjclose") as mock_adjust:
                mock_adjust.return_value = (sample_dataframe, False)

                start_date = datetime(2023, 1, 1)
                end_date = datetime(2023, 1, 31)
                symbols = ["AAPL", "GOOGL"]

                with pytest.raises(ValueError, match="Failed to load 1 symbols"):
                    pipeline.fetch_and_validate(symbols, start_date, end_date)

    def test_perform_quality_checks_insufficient_data(self, pipeline):
        """Test quality checks with insufficient data points."""
        # Create DataFrame with fewer data points than minimum
        small_df = pd.DataFrame(
            {"Open": [100.0], "High": [105.0], "Low": [95.0], "Close": [102.0]},
            index=pd.DatetimeIndex(["2023-01-01"]),
        )

        with pytest.raises(ValueError, match="insufficient data points"):
            pipeline._perform_quality_checks(small_df, "TEST")

    def test_perform_quality_checks_non_positive_prices(self, pipeline):
        """Test quality checks with non-positive prices."""
        # Create DataFrame with invalid prices
        invalid_df = pd.DataFrame(
            {
                "Open": [-1.0, 100.0],
                "High": [105.0, 105.0],
                "Low": [95.0, 95.0],
                "Close": [102.0, 102.0],
            },
            index=pd.date_range("2023-01-01", periods=2),
        )

        # Adjust config to allow this many data points
        pipeline.config.min_data_points = 2

        with pytest.raises(ValueError, match="non-positive prices detected"):
            pipeline._perform_quality_checks(invalid_df, "TEST")

    def test_perform_quality_checks_unrealistic_prices(self, pipeline):
        """Test quality checks with unrealistically high prices."""
        # Create DataFrame with unrealistic prices
        unrealistic_df = pd.DataFrame(
            {
                "Open": [100000.0, 100.0],
                "High": [200000.0, 105.0],
                "Low": [95.0, 95.0],
                "Close": [102.0, 102.0],
            },
            index=pd.date_range("2023-01-01", periods=2),
        )

        # Adjust config to allow this many data points
        pipeline.config.min_data_points = 2

        with pytest.raises(ValueError, match="unrealistically high prices"):
            pipeline._perform_quality_checks(unrealistic_df, "TEST")

    @patch("bot.dataflow.pipeline.validate_daily_bars")
    @patch("bot.dataflow.pipeline.adjust_to_adjclose")
    def test_warm_cache(self, mock_adjust, mock_validate, pipeline, sample_dataframe):
        """Test cache warming functionality."""
        # Setup mocks
        pipeline.source.get_daily_bars.return_value = sample_dataframe
        mock_adjust.return_value = (sample_dataframe, False)
        mock_validate.return_value = None

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        symbols = ["AAPL", "GOOGL"]

        results = pipeline.warm_cache(symbols, start_date, end_date, quiet=True)

        assert len(results) == 2
        assert results["AAPL"] is True
        assert results["GOOGL"] is True

        # Check that data is actually cached
        assert len(pipeline._cache) == 2

    def test_clear_cache_all(self, pipeline):
        """Test clearing entire cache."""
        # Add some mock cache entries
        pipeline._cache = {
            "AAPL_2023-01-01_2023-01-31": (pd.DataFrame(), datetime.now()),
            "GOOGL_2023-01-01_2023-01-31": (pd.DataFrame(), datetime.now()),
        }

        pipeline.clear_cache()

        assert len(pipeline._cache) == 0

    def test_clear_cache_specific_symbol(self, pipeline):
        """Test clearing cache for specific symbol."""
        # Add some mock cache entries
        pipeline._cache = {
            "AAPL_2023-01-01_2023-01-31": (pd.DataFrame(), datetime.now()),
            "AAPL_2023-02-01_2023-02-28": (pd.DataFrame(), datetime.now()),
            "GOOGL_2023-01-01_2023-01-31": (pd.DataFrame(), datetime.now()),
        }

        pipeline.clear_cache("AAPL")

        assert len(pipeline._cache) == 1
        assert "GOOGL_2023-01-01_2023-01-31" in pipeline._cache

    def test_get_cache_info(self, pipeline):
        """Test cache information retrieval."""
        # Add some mock cache entries with different ages
        now = datetime.now()
        pipeline._cache = {
            "AAPL_recent": (pd.DataFrame({"A": [1, 2, 3]}), now),
            "GOOGL_old": (pd.DataFrame({"B": [4, 5, 6]}), now - timedelta(hours=25)),
        }

        info = pipeline.get_cache_info()

        assert info["total_entries"] == 2
        assert "estimated_memory_mb" in info
        assert info["ttl_hours"] == pipeline.config.cache_ttl_hours
        assert "age_distribution" in info
        assert info["age_distribution"]["<1h"] == 1
        assert info["age_distribution"][">24h"] == 1

    def test_get_metrics(self, pipeline):
        """Test metrics retrieval."""
        metrics = pipeline.get_metrics()
        assert isinstance(metrics, DataQualityMetrics)

    @patch("bot.dataflow.pipeline.validate_daily_bars")
    @patch("bot.dataflow.pipeline.adjust_to_adjclose")
    def test_health_check_success(self, mock_adjust, mock_validate, pipeline, sample_dataframe):
        """Test successful health check."""
        # Setup mocks
        pipeline.source.get_daily_bars.return_value = sample_dataframe
        mock_adjust.return_value = (sample_dataframe, False)
        mock_validate.return_value = None

        health = pipeline.health_check("AAPL")

        assert health["status"] == "healthy"
        assert len(health["errors"]) == 0
        assert health["tests"]["data_fetch"]["success"] is True
        assert health["tests"]["validation"]["success"] is True
        assert "response_time_ms" in health["tests"]["data_fetch"]

    def test_health_check_data_fetch_failure(self, pipeline):
        """Test health check with data fetch failure."""
        pipeline.source.get_daily_bars.side_effect = Exception("Network error")

        health = pipeline.health_check("AAPL")

        assert health["status"] == "unhealthy"
        assert len(health["errors"]) > 0
        assert health["tests"]["data_fetch"]["success"] is False
        assert "error" in health["tests"]["data_fetch"]

    @patch("bot.dataflow.pipeline.validate_daily_bars")
    def test_health_check_validation_failure(self, mock_validate, pipeline, sample_dataframe):
        """Test health check with validation failure."""
        # Setup mock to fail validation on test data
        mock_validate.side_effect = ValueError("Validation error")

        health = pipeline.health_check("AAPL")

        assert health["status"] == "unhealthy"
        assert len(health["errors"]) > 0
        assert health["tests"]["validation"]["success"] is False
