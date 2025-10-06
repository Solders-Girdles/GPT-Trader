"""Unit tests for shared symbol utilities."""

import os
from unittest.mock import Mock, patch

import pytest

from bot_v2.orchestration.shared.symbol_utils import (
    PERPS_ALLOWLIST,
    TOP_VOLUME_BASES,
    derivatives_enabled,
    normalize_symbols,
)


class TestConstants:
    """Test module constants."""

    def test_perps_allowlist_contains_expected_symbols(self):
        """PERPS_ALLOWLIST contains major perpetual symbols."""
        assert "BTC-PERP" in PERPS_ALLOWLIST
        assert "ETH-PERP" in PERPS_ALLOWLIST
        assert "SOL-PERP" in PERPS_ALLOWLIST
        assert "XRP-PERP" in PERPS_ALLOWLIST

    def test_top_volume_bases_contains_major_coins(self):
        """TOP_VOLUME_BASES contains top trading base currencies."""
        assert "BTC" in TOP_VOLUME_BASES
        assert "ETH" in TOP_VOLUME_BASES
        assert "SOL" in TOP_VOLUME_BASES
        assert len(TOP_VOLUME_BASES) >= 5  # At least 5 major coins


class TestDerivativesEnabled:
    """Test derivatives_enabled function."""

    def test_spot_profile_disables_derivatives(self):
        """SPOT profile always disables derivatives."""
        profile = Mock()
        profile.value = "spot"

        with patch.dict(os.environ, {"COINBASE_ENABLE_DERIVATIVES": "1"}):
            assert derivatives_enabled(profile) is False

    def test_non_spot_profile_with_env_enabled(self):
        """Non-SPOT profile with COINBASE_ENABLE_DERIVATIVES=1 enables derivatives."""
        profile = Mock()
        profile.value = "prod"

        with patch.dict(os.environ, {"COINBASE_ENABLE_DERIVATIVES": "1"}):
            assert derivatives_enabled(profile) is True

    def test_non_spot_profile_without_env(self):
        """Non-SPOT profile without environment variable disables derivatives."""
        profile = Mock()
        profile.value = "prod"

        with patch.dict(os.environ, {}, clear=True):
            assert derivatives_enabled(profile) is False

    def test_derivatives_disabled_by_default(self):
        """Derivatives are disabled by default."""
        profile = Mock()
        profile.value = "canary"

        with patch.dict(os.environ, {"COINBASE_ENABLE_DERIVATIVES": "0"}):
            assert derivatives_enabled(profile) is False

    def test_handles_profile_as_string(self):
        """Function works when profile is a plain string."""
        with patch.dict(os.environ, {"COINBASE_ENABLE_DERIVATIVES": "1"}):
            assert derivatives_enabled("prod") is True
            assert derivatives_enabled("spot") is False


class TestNormalizeSymbols:
    """Test normalize_symbols function."""

    def test_spot_profile_keeps_spot_symbols(self):
        """SPOT profile keeps spot symbols unchanged."""
        profile = Mock()
        profile.value = "spot"

        symbols = ["BTC-USD", "ETH-USD"]
        normalized, derivs = normalize_symbols(profile, symbols)

        assert normalized == ["BTC-USD", "ETH-USD"]
        assert derivs is False

    def test_spot_profile_converts_perp_to_spot(self):
        """SPOT profile converts perpetual symbols to spot."""
        profile = Mock()
        profile.value = "spot"

        symbols = ["BTC-PERP", "ETH-PERP"]

        with patch.dict(os.environ, {"COINBASE_ENABLE_DERIVATIVES": "0"}):
            normalized, derivs = normalize_symbols(profile, symbols)

        assert "BTC-USD" in normalized
        assert "ETH-USD" in normalized
        assert derivs is False

    def test_perp_profile_keeps_allowed_perps(self):
        """Non-SPOT profile with derivatives enabled keeps allowed perpetuals."""
        profile = Mock()
        profile.value = "prod"

        symbols = ["BTC-PERP", "ETH-PERP"]

        with patch.dict(os.environ, {"COINBASE_ENABLE_DERIVATIVES": "1"}):
            normalized, derivs = normalize_symbols(profile, symbols)

        assert "BTC-PERP" in normalized
        assert "ETH-PERP" in normalized
        assert derivs is True

    def test_filters_unsupported_perps(self):
        """Filters out perpetuals not in PERPS_ALLOWLIST."""
        profile = Mock()
        profile.value = "prod"

        symbols = ["BTC-PERP", "UNKNOWN-PERP"]

        with patch.dict(os.environ, {"COINBASE_ENABLE_DERIVATIVES": "1"}):
            normalized, derivs = normalize_symbols(profile, symbols)

        assert "BTC-PERP" in normalized
        assert "UNKNOWN-PERP" not in normalized
        assert derivs is True

    def test_normalizes_case(self):
        """Symbols are normalized to uppercase."""
        profile = Mock()
        profile.value = "spot"

        symbols = ["btc-usd", "eth-usd"]
        normalized, _ = normalize_symbols(profile, symbols)

        assert normalized == ["BTC-USD", "ETH-USD"]

    def test_removes_duplicates(self):
        """Duplicate symbols are removed while preserving order."""
        profile = Mock()
        profile.value = "spot"

        symbols = ["BTC-USD", "ETH-USD", "BTC-USD", "SOL-USD"]
        normalized, _ = normalize_symbols(profile, symbols)

        assert normalized == ["BTC-USD", "ETH-USD", "SOL-USD"]

    def test_empty_symbols_uses_defaults(self):
        """Empty symbol list returns defaults based on profile."""
        profile = Mock()
        profile.value = "spot"

        with patch.dict(os.environ, {}, clear=True):
            normalized, derivs = normalize_symbols(profile, [])

        # Should return spot defaults from TOP_VOLUME_BASES
        assert len(normalized) > 0
        assert all("-USD" in sym for sym in normalized)
        assert derivs is False

    def test_perp_profile_default_returns_perps(self):
        """Perp profile with empty symbols returns perp defaults."""
        profile = Mock()
        profile.value = "prod"

        with patch.dict(os.environ, {"COINBASE_ENABLE_DERIVATIVES": "1"}):
            normalized, derivs = normalize_symbols(profile, [])

        assert "BTC-PERP" in normalized
        assert "ETH-PERP" in normalized
        assert derivs is True

    def test_custom_quote_currency(self):
        """Custom quote currency is applied."""
        profile = Mock()
        profile.value = "spot"

        symbols = []
        normalized, _ = normalize_symbols(profile, symbols, quote="USDT")

        # Default symbols should use USDT quote
        assert any("-USDT" in sym for sym in normalized)

    def test_strips_whitespace(self):
        """Symbol whitespace is stripped."""
        profile = Mock()
        profile.value = "spot"

        symbols = ["  BTC-USD  ", " ETH-USD"]
        normalized, _ = normalize_symbols(profile, symbols)

        assert normalized == ["BTC-USD", "ETH-USD"]

    def test_filters_empty_strings(self):
        """Empty strings are filtered out."""
        profile = Mock()
        profile.value = "spot"

        symbols = ["BTC-USD", "", "  ", "ETH-USD"]
        normalized, _ = normalize_symbols(profile, symbols)

        assert normalized == ["BTC-USD", "ETH-USD"]

    def test_none_symbols_uses_defaults(self):
        """None symbols list returns defaults."""
        profile = Mock()
        profile.value = "spot"

        with patch.dict(os.environ, {}, clear=True):
            normalized, _ = normalize_symbols(profile, None)

        assert len(normalized) > 0
