"""Unit tests for optimize CLI formatters."""

from __future__ import annotations

import json

import pytest

from gpt_trader.cli.commands.optimize.formatters import (
    format_comparison_json,
    format_comparison_text,
    format_run_list_json,
    format_run_list_text,
    format_run_summary_json,
    format_run_summary_text,
    format_trials_csv,
    format_trials_text,
)


@pytest.fixture
def sample_run():
    """Create a sample run for testing."""
    return {
        "run_id": "opt_20241128_abc123",
        "study_name": "test_sharpe",
        "started_at": "2024-11-28T10:00:00",
        "completed_at": "2024-11-28T10:30:00",
        "total_trials": 100,
        "feasible_trials": 85,
        "best_objective_value": 2.34,
        "best_parameters": {
            "short_ma_period": 8,
            "long_ma_period": 35,
            "rsi_period": 14,
        },
    }


@pytest.fixture
def sample_trials():
    """Create sample trials for testing."""
    return [
        {
            "trial_number": 1,
            "objective_value": 2.34,
            "is_feasible": True,
            "duration_seconds": 5.2,
            "parameters": {"short_ma_period": 8, "long_ma_period": 35},
        },
        {
            "trial_number": 2,
            "objective_value": 1.89,
            "is_feasible": True,
            "duration_seconds": 4.8,
            "parameters": {"short_ma_period": 10, "long_ma_period": 40},
        },
        {
            "trial_number": 3,
            "objective_value": -0.5,
            "is_feasible": False,
            "duration_seconds": 3.1,
            "parameters": {"short_ma_period": 3, "long_ma_period": 20},
        },
    ]


class TestFormatRunSummaryText:
    def test_includes_study_name(self, sample_run):
        """Test output includes study name."""
        result = format_run_summary_text(sample_run)
        assert "test_sharpe" in result

    def test_includes_run_id(self, sample_run):
        """Test output includes run ID."""
        result = format_run_summary_text(sample_run)
        assert "opt_20241128_abc123" in result

    def test_includes_best_value(self, sample_run):
        """Test output includes best objective value."""
        result = format_run_summary_text(sample_run)
        assert "2.34" in result

    def test_includes_parameters(self, sample_run):
        """Test output includes best parameters."""
        result = format_run_summary_text(sample_run)
        assert "short_ma_period" in result
        assert "8" in result

    def test_includes_trial_counts(self, sample_run):
        """Test output includes trial counts."""
        result = format_run_summary_text(sample_run)
        assert "100" in result
        assert "85" in result


class TestFormatRunSummaryJson:
    def test_returns_valid_json(self, sample_run):
        """Test output is valid JSON."""
        result = format_run_summary_json(sample_run)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_includes_status(self, sample_run):
        """Test output includes status."""
        result = format_run_summary_json(sample_run)
        parsed = json.loads(result)
        assert parsed["status"] == "completed"

    def test_includes_best_trial(self, sample_run):
        """Test output includes best trial info."""
        result = format_run_summary_json(sample_run)
        parsed = json.loads(result)
        assert parsed["best_trial"]["objective_value"] == 2.34

    def test_compact_mode(self, sample_run):
        """Test compact mode produces shorter output."""
        normal = format_run_summary_json(sample_run, compact=False)
        compact = format_run_summary_json(sample_run, compact=True)
        assert len(compact) < len(normal)


class TestFormatTrialsText:
    def test_shows_limited_trials(self, sample_trials):
        """Test output respects limit."""
        result = format_trials_text(sample_trials, limit=2)
        lines = result.split("\n")
        # Data lines contain trial numbers (right-aligned) - look for lines with objective values
        data_lines = [
            line for line in lines if line.strip() and "." in line and "Objective" not in line
        ]
        # Should have 2 data lines (top 2 trials)
        assert len(data_lines) == 2

    def test_empty_trials(self):
        """Test handling of empty trials list."""
        result = format_trials_text([])
        assert "No trials found" in result

    def test_shows_parameters_when_requested(self, sample_trials):
        """Test parameters are shown when show_all_params=True."""
        result = format_trials_text(sample_trials, limit=1, show_all_params=True)
        assert "short_ma_period" in result


class TestFormatTrialsCsv:
    def test_returns_valid_csv(self, sample_trials):
        """Test output is valid CSV format."""
        result = format_trials_csv(sample_trials)
        lines = result.strip().split("\n")
        # Header + 3 data rows
        assert len(lines) == 4

    def test_includes_header(self, sample_trials):
        """Test output includes header row."""
        result = format_trials_csv(sample_trials)
        header = result.split("\n")[0]
        assert "trial_number" in header
        assert "objective_value" in header

    def test_empty_trials(self):
        """Test handling of empty trials list."""
        result = format_trials_csv([])
        assert result == ""


class TestFormatRunListText:
    def test_shows_runs(self, sample_run):
        """Test output shows run info."""
        runs = [sample_run]
        result = format_run_list_text(runs)
        assert "opt_20241128_abc123" in result
        assert "test_sharpe" in result

    def test_empty_runs(self):
        """Test handling of empty runs list."""
        result = format_run_list_text([])
        assert "No optimization runs found" in result


class TestFormatRunListJson:
    def test_returns_valid_json(self, sample_run):
        """Test output is valid JSON."""
        runs = [sample_run]
        result = format_run_list_json(runs)
        parsed = json.loads(result)
        assert "runs" in parsed
        assert len(parsed["runs"]) == 1


class TestFormatComparisonText:
    def test_shows_multiple_runs(self, sample_run):
        """Test output shows multiple runs."""
        run2 = dict(sample_run)
        run2["run_id"] = "opt_20241128_def456"
        run2["best_objective_value"] = 1.89

        result = format_comparison_text([sample_run, run2])
        assert "opt_20241128_abc123" in result
        assert "opt_20241128_def456" in result

    def test_single_run_shows_summary(self, sample_run):
        """Test single run shows summary."""
        result = format_comparison_text([sample_run])
        assert "test_sharpe" in result


class TestFormatComparisonJson:
    def test_returns_valid_json(self, sample_run):
        """Test output is valid JSON."""
        run2 = dict(sample_run)
        run2["run_id"] = "opt_20241128_def456"

        result = format_comparison_json([sample_run, run2])
        parsed = json.loads(result)
        assert "comparison" in parsed
        assert len(parsed["comparison"]) == 2
