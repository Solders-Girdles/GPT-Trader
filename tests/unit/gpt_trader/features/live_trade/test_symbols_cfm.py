"""Tests for CFM (Coinbase Financial Markets) symbols in gpt_trader.features.live_trade.symbols."""

from __future__ import annotations

from gpt_trader.app.config import BotConfig
from gpt_trader.features.live_trade import symbols


class TestCfmEnabled:
    """Tests for cfm_enabled function."""

    def test_returns_false_by_default(self) -> None:
        config = BotConfig()
        result = symbols.cfm_enabled(config)
        assert result is False

    def test_returns_true_when_cfm_flag_enabled(self) -> None:
        config = BotConfig(cfm_enabled=True)
        result = symbols.cfm_enabled(config)
        assert result is True

    def test_returns_true_when_cfm_in_trading_modes(self) -> None:
        config = BotConfig(trading_modes=["spot", "cfm"])
        result = symbols.cfm_enabled(config)
        assert result is True

    def test_returns_true_when_cfm_only_mode(self) -> None:
        config = BotConfig(trading_modes=["cfm"])
        result = symbols.cfm_enabled(config)
        assert result is True


class TestGetCfmSymbol:
    """Tests for get_cfm_symbol function."""

    def test_maps_btc_base(self) -> None:
        result = symbols.get_cfm_symbol("BTC")
        assert result == "BTC-20DEC30-CDE"

    def test_maps_btc_spot_symbol(self) -> None:
        result = symbols.get_cfm_symbol("BTC-USD")
        assert result == "BTC-20DEC30-CDE"

    def test_maps_eth_base(self) -> None:
        result = symbols.get_cfm_symbol("ETH")
        assert result == "ETH-20DEC30-CDE"

    def test_maps_sol_to_slp(self) -> None:
        result = symbols.get_cfm_symbol("SOL")
        assert result == "SLP-20DEC30-CDE"

    def test_returns_none_for_unmapped(self) -> None:
        result = symbols.get_cfm_symbol("DOGE")
        assert result is None

    def test_case_insensitive(self) -> None:
        result = symbols.get_cfm_symbol("btc-usd")
        assert result == "BTC-20DEC30-CDE"


class TestGetSpotSymbol:
    """Tests for get_spot_symbol function."""

    def test_converts_cfm_to_spot(self) -> None:
        result = symbols.get_spot_symbol("BTC-20DEC30-CDE")
        assert result == "BTC-USD"

    def test_converts_base_to_spot(self) -> None:
        result = symbols.get_spot_symbol("BTC")
        assert result == "BTC-USD"

    def test_handles_slp_to_sol(self) -> None:
        result = symbols.get_spot_symbol("SLP-20DEC30-CDE")
        assert result == "SOL-USD"

    def test_custom_quote_currency(self) -> None:
        result = symbols.get_spot_symbol("BTC", quote="EUR")
        assert result == "BTC-EUR"

    def test_case_insensitive(self) -> None:
        result = symbols.get_spot_symbol("btc-20dec30-cde")
        assert result == "BTC-USD"


class TestNormalizeSymbolForMode:
    """Tests for normalize_symbol_for_mode function."""

    def test_spot_to_cfm(self) -> None:
        result = symbols.normalize_symbol_for_mode("BTC-USD", "cfm")
        assert result == "BTC-20DEC30-CDE"

    def test_cfm_to_spot(self) -> None:
        result = symbols.normalize_symbol_for_mode("BTC-20DEC30-CDE", "spot")
        assert result == "BTC-USD"

    def test_spot_to_spot_is_normalized(self) -> None:
        result = symbols.normalize_symbol_for_mode("btc-usd", "spot")
        assert result == "BTC-USD"

    def test_unmapped_cfm_returns_original(self) -> None:
        result = symbols.normalize_symbol_for_mode("DOGE-USD", "cfm")
        # Should return original when no CFM mapping exists
        assert result == "DOGE-USD"

    def test_custom_quote(self) -> None:
        result = symbols.normalize_symbol_for_mode("BTC-20DEC30-CDE", "spot", quote="EUR")
        assert result == "BTC-EUR"


class TestGetSymbolPairsForHybrid:
    """Tests for get_symbol_pairs_for_hybrid function."""

    def test_single_symbol(self) -> None:
        result = symbols.get_symbol_pairs_for_hybrid(["BTC-USD"])
        assert result == {
            "BTC": {"spot": "BTC-USD", "cfm": "BTC-20DEC30-CDE"},
        }

    def test_multiple_symbols(self) -> None:
        result = symbols.get_symbol_pairs_for_hybrid(["BTC-USD", "ETH-USD"])
        assert "BTC" in result
        assert "ETH" in result
        assert result["BTC"] == {"spot": "BTC-USD", "cfm": "BTC-20DEC30-CDE"}
        assert result["ETH"] == {"spot": "ETH-USD", "cfm": "ETH-20DEC30-CDE"}

    def test_handles_slp_to_sol_mapping(self) -> None:
        result = symbols.get_symbol_pairs_for_hybrid(["SOL-USD"])
        assert result == {
            "SOL": {"spot": "SOL-USD", "cfm": "SLP-20DEC30-CDE"},
        }

    def test_custom_quote(self) -> None:
        result = symbols.get_symbol_pairs_for_hybrid(["BTC-USD"], quote="EUR")
        assert result["BTC"]["spot"] == "BTC-EUR"

    def test_deduplicate_base_assets(self) -> None:
        result = symbols.get_symbol_pairs_for_hybrid(["BTC-USD", "BTC-EUR"])
        # Should only have one BTC entry (first wins)
        assert len(result) == 1
        assert "BTC" in result

    def test_unmapped_base_fallback_to_spot(self) -> None:
        result = symbols.get_symbol_pairs_for_hybrid(["DOGE-USD"])
        # When no CFM mapping, cfm should fallback to spot
        assert result["DOGE"]["cfm"] == "DOGE-USD"


class TestCfmSymbolMapping:
    """Tests for CFM_SYMBOL_MAPPING constant."""

    def test_contains_btc(self) -> None:
        assert "BTC" in symbols.CFM_SYMBOL_MAPPING

    def test_contains_eth(self) -> None:
        assert "ETH" in symbols.CFM_SYMBOL_MAPPING

    def test_contains_sol(self) -> None:
        assert "SOL" in symbols.CFM_SYMBOL_MAPPING

    def test_btc_mapping_format(self) -> None:
        # CFM symbols should have format: BASE-EXPIRY-CDE
        btc_cfm = symbols.CFM_SYMBOL_MAPPING["BTC"]
        assert btc_cfm.startswith("BTC-")
        assert btc_cfm.endswith("-CDE")
