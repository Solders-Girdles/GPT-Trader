"""Tests for account CLI commands."""

import json
from unittest.mock import Mock, patch

import pytest

from bot_v2.cli.commands.account import handle_account_snapshot


@pytest.fixture
def mock_bot_with_telemetry():
    """Create a mock PerpsBot with account telemetry."""
    bot = Mock()
    telemetry = Mock()
    telemetry.supports_snapshots.return_value = True
    telemetry.collect_snapshot.return_value = {
        "account_id": "acc-123",
        "balance": "10000.00",
        "available_balance": "9500.00",
        "positions": [
            {
                "symbol": "BTC-USD",
                "size": "0.5",
                "unrealized_pnl": "250.00"
            }
        ],
        "timestamp": "2023-01-01T12:00:00Z"
    }
    bot.account_telemetry = telemetry
    return bot


@pytest.fixture
def mock_bot_without_telemetry():
    """Create a mock PerpsBot without account telemetry."""
    bot = Mock()
    bot.account_telemetry = None
    return bot


@pytest.fixture
def mock_bot_telemetry_no_snapshots():
    """Create a mock PerpsBot with telemetry that doesn't support snapshots."""
    bot = Mock()
    telemetry = Mock()
    telemetry.supports_snapshots.return_value = False
    bot.account_telemetry = telemetry
    return bot


class TestHandleAccountSnapshot:
    """Tests for handle_account_snapshot function."""

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_successful_snapshot(self, mock_shutdown, mock_bot_with_telemetry, capsys):
        """Test successful account snapshot collection."""
        result = handle_account_snapshot(mock_bot_with_telemetry)

        assert result == 0
        mock_bot_with_telemetry.account_telemetry.supports_snapshots.assert_called_once()
        mock_bot_with_telemetry.account_telemetry.collect_snapshot.assert_called_once()
        mock_shutdown.assert_called_once_with(mock_bot_with_telemetry)

        # Check output
        captured = capsys.readouterr()
        assert "acc-123" in captured.out
        assert "BTC-USD" in captured.out

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_snapshot_output_is_json(self, mock_shutdown, mock_bot_with_telemetry, capsys):
        """Test that snapshot output is valid JSON."""
        handle_account_snapshot(mock_bot_with_telemetry)

        captured = capsys.readouterr()
        # Should be able to parse as JSON
        parsed = json.loads(captured.out)
        assert "account_id" in parsed
        assert "balance" in parsed
        assert "positions" in parsed

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_snapshot_json_formatting(self, mock_shutdown, mock_bot_with_telemetry, capsys):
        """Test that JSON output is formatted with indentation."""
        handle_account_snapshot(mock_bot_with_telemetry)

        captured = capsys.readouterr()
        # Should have indentation (newlines indicate formatting)
        assert "\n" in captured.out
        assert "  " in captured.out  # Indentation

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_no_telemetry_available(self, mock_shutdown, mock_bot_without_telemetry):
        """Test error when account telemetry is not available."""
        with pytest.raises(RuntimeError, match="Account snapshot telemetry is not available"):
            handle_account_snapshot(mock_bot_without_telemetry)

        mock_shutdown.assert_called_once_with(mock_bot_without_telemetry)

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_telemetry_doesnt_support_snapshots(self, mock_shutdown, mock_bot_telemetry_no_snapshots):
        """Test error when telemetry doesn't support snapshots."""
        with pytest.raises(RuntimeError, match="Account snapshot telemetry is not available"):
            handle_account_snapshot(mock_bot_telemetry_no_snapshots)

        mock_bot_telemetry_no_snapshots.account_telemetry.supports_snapshots.assert_called_once()
        mock_shutdown.assert_called_once_with(mock_bot_telemetry_no_snapshots)

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_collection_exception(self, mock_shutdown, mock_bot_with_telemetry):
        """Test exception during snapshot collection."""
        mock_bot_with_telemetry.account_telemetry.collect_snapshot.side_effect = Exception(
            "API error"
        )

        with pytest.raises(Exception, match="API error"):
            handle_account_snapshot(mock_bot_with_telemetry)

        # Shutdown should still be called in finally block
        mock_shutdown.assert_called_once_with(mock_bot_with_telemetry)

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_snapshot_with_empty_positions(self, mock_shutdown, mock_bot_with_telemetry, capsys):
        """Test snapshot with no positions."""
        mock_bot_with_telemetry.account_telemetry.collect_snapshot.return_value = {
            "account_id": "acc-456",
            "balance": "5000.00",
            "available_balance": "5000.00",
            "positions": [],
            "timestamp": "2023-01-01T12:00:00Z"
        }

        result = handle_account_snapshot(mock_bot_with_telemetry)

        assert result == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["positions"] == []

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_snapshot_with_multiple_positions(self, mock_shutdown, mock_bot_with_telemetry, capsys):
        """Test snapshot with multiple positions."""
        mock_bot_with_telemetry.account_telemetry.collect_snapshot.return_value = {
            "account_id": "acc-789",
            "balance": "20000.00",
            "positions": [
                {"symbol": "BTC-USD", "size": "1.0"},
                {"symbol": "ETH-USD", "size": "10.0"},
                {"symbol": "SOL-USD", "size": "100.0"},
            ]
        }

        result = handle_account_snapshot(mock_bot_with_telemetry)

        assert result == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert len(parsed["positions"]) == 3

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_snapshot_with_nested_data(self, mock_shutdown, mock_bot_with_telemetry, capsys):
        """Test snapshot with deeply nested data structures."""
        mock_bot_with_telemetry.account_telemetry.collect_snapshot.return_value = {
            "account_id": "acc-complex",
            "balance": "15000.00",
            "metadata": {
                "created_at": "2022-01-01",
                "tier": "premium",
                "features": {
                    "margin": True,
                    "futures": True,
                    "options": False
                }
            }
        }

        result = handle_account_snapshot(mock_bot_with_telemetry)

        assert result == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["metadata"]["tier"] == "premium"
        assert parsed["metadata"]["features"]["margin"] is True

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_snapshot_with_special_types(self, mock_shutdown, mock_bot_with_telemetry, capsys):
        """Test snapshot with non-JSON-serializable types."""
        from datetime import datetime
        from decimal import Decimal

        mock_bot_with_telemetry.account_telemetry.collect_snapshot.return_value = {
            "account_id": "acc-special",
            "balance": Decimal("10000.50"),
            "timestamp": datetime(2023, 1, 1, 12, 0, 0),
        }

        result = handle_account_snapshot(mock_bot_with_telemetry)

        assert result == 0
        # Should not crash thanks to default=str in json.dumps
        captured = capsys.readouterr()
        assert "acc-special" in captured.out

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_telemetry_attribute_check(self, mock_shutdown):
        """Test that telemetry is checked via getattr."""
        bot = Mock(spec=[])  # Bot with no attributes

        with pytest.raises(RuntimeError):
            handle_account_snapshot(bot)

        mock_shutdown.assert_called_once_with(bot)

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_supports_snapshots_false(self, mock_shutdown):
        """Test when supports_snapshots() returns False."""
        bot = Mock()
        telemetry = Mock()
        telemetry.supports_snapshots.return_value = False
        bot.account_telemetry = telemetry

        with pytest.raises(RuntimeError):
            handle_account_snapshot(bot)

        telemetry.supports_snapshots.assert_called_once()

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_shutdown_called_on_success(self, mock_shutdown, mock_bot_with_telemetry):
        """Test that shutdown is called even on success."""
        handle_account_snapshot(mock_bot_with_telemetry)

        mock_shutdown.assert_called_once()

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_shutdown_called_on_error_before_collection(self, mock_shutdown, mock_bot_without_telemetry):
        """Test that shutdown is called when error occurs before collection."""
        with pytest.raises(RuntimeError):
            handle_account_snapshot(mock_bot_without_telemetry)

        mock_shutdown.assert_called_once()

    @patch("bot_v2.cli.commands.account.ensure_shutdown")
    def test_shutdown_called_on_error_during_collection(self, mock_shutdown, mock_bot_with_telemetry):
        """Test that shutdown is called when error occurs during collection."""
        mock_bot_with_telemetry.account_telemetry.collect_snapshot.side_effect = Exception("Error")

        with pytest.raises(Exception):
            handle_account_snapshot(mock_bot_with_telemetry)

        mock_shutdown.assert_called_once()
