"""Tests for PreferencesService watchlist functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from gpt_trader.tui.services.preferences_service import PreferencesService


@pytest.fixture
def temp_prefs_file():
    """Create a temporary preferences file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({}, f)
        return Path(f.name)


@pytest.fixture
def prefs_service(temp_prefs_file):
    """Create PreferencesService with temp file."""
    return PreferencesService(preferences_path=temp_prefs_file)


class TestWatchlistHelpers:
    """Tests for watchlist helper methods."""

    def test_get_last_symbols_empty(self, prefs_service):
        """Get symbols returns empty list initially."""
        symbols = prefs_service.get_last_symbols()
        assert symbols == []

    def test_set_last_symbols(self, prefs_service):
        """Set symbols and retrieve them."""
        prefs_service.set_last_symbols(["BTC-USD", "ETH-USD"])
        symbols = prefs_service.get_last_symbols()
        assert symbols == ["BTC-USD", "ETH-USD"]

    def test_add_watchlist_symbol(self, prefs_service):
        """Add a symbol to the watchlist."""
        result = prefs_service.add_watchlist_symbol("BTC-USD")
        assert result is True

        symbols = prefs_service.get_last_symbols()
        assert "BTC-USD" in symbols

    def test_add_watchlist_symbol_duplicate(self, prefs_service):
        """Adding duplicate symbol returns True but doesn't duplicate."""
        prefs_service.add_watchlist_symbol("BTC-USD")
        result = prefs_service.add_watchlist_symbol("BTC-USD")
        assert result is True

        symbols = prefs_service.get_last_symbols()
        assert symbols.count("BTC-USD") == 1

    def test_add_watchlist_symbol_normalized(self, prefs_service):
        """Symbol is normalized (uppercase, trimmed)."""
        prefs_service.add_watchlist_symbol("  btc-usd  ")

        symbols = prefs_service.get_last_symbols()
        assert "BTC-USD" in symbols

    def test_add_watchlist_symbol_empty(self, prefs_service):
        """Adding empty symbol returns False."""
        result = prefs_service.add_watchlist_symbol("")
        assert result is False

        result = prefs_service.add_watchlist_symbol("   ")
        assert result is False

    def test_remove_watchlist_symbol(self, prefs_service):
        """Remove a symbol from the watchlist."""
        prefs_service.set_last_symbols(["BTC-USD", "ETH-USD"])

        result = prefs_service.remove_watchlist_symbol("BTC-USD")
        assert result is True

        symbols = prefs_service.get_last_symbols()
        assert "BTC-USD" not in symbols
        assert "ETH-USD" in symbols

    def test_remove_watchlist_symbol_not_present(self, prefs_service):
        """Removing non-existent symbol returns True."""
        result = prefs_service.remove_watchlist_symbol("NONEXISTENT")
        assert result is True

    def test_clear_watchlist(self, prefs_service):
        """Clear all symbols from the watchlist."""
        prefs_service.set_last_symbols(["BTC-USD", "ETH-USD", "SOL-USD"])

        result = prefs_service.clear_watchlist()
        assert result is True

        symbols = prefs_service.get_last_symbols()
        assert symbols == []

    def test_reorder_watchlist(self, prefs_service):
        """Reorder watchlist preserves order."""
        original = ["BTC-USD", "ETH-USD", "SOL-USD"]
        prefs_service.set_last_symbols(original)

        reordered = ["SOL-USD", "BTC-USD", "ETH-USD"]
        result = prefs_service.reorder_watchlist(reordered)
        assert result is True

        symbols = prefs_service.get_last_symbols()
        assert symbols == reordered

    def test_reorder_watchlist_deduplicates(self, prefs_service):
        """Reorder removes duplicates while preserving order."""
        result = prefs_service.reorder_watchlist(["BTC-USD", "ETH-USD", "BTC-USD"])
        assert result is True

        symbols = prefs_service.get_last_symbols()
        assert symbols == ["BTC-USD", "ETH-USD"]

    def test_reorder_watchlist_normalizes(self, prefs_service):
        """Reorder normalizes symbols."""
        result = prefs_service.reorder_watchlist(["btc-usd", "  eth-usd  "])
        assert result is True

        symbols = prefs_service.get_last_symbols()
        assert symbols == ["BTC-USD", "ETH-USD"]

    def test_get_last_symbols_returns_copy(self, prefs_service):
        """Get symbols returns a copy, not the original list."""
        prefs_service.set_last_symbols(["BTC-USD"])

        symbols = prefs_service.get_last_symbols()
        symbols.append("ETH-USD")  # Modify the returned list

        # Original should be unchanged
        symbols2 = prefs_service.get_last_symbols()
        assert symbols2 == ["BTC-USD"]

    def test_persistence(self, temp_prefs_file):
        """Symbols persist across service instances."""
        # First instance
        service1 = PreferencesService(preferences_path=temp_prefs_file)
        service1.set_last_symbols(["BTC-USD", "ETH-USD"])

        # New instance with same file
        service2 = PreferencesService(preferences_path=temp_prefs_file)
        symbols = service2.get_last_symbols()
        assert symbols == ["BTC-USD", "ETH-USD"]
