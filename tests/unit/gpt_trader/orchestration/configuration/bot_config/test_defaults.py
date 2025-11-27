"""Tests for bot_config defaults module."""

from __future__ import annotations

from pathlib import Path

from gpt_trader.orchestration.configuration.bot_config.defaults import (
    DEFAULT_SPOT_RISK_PATH,
    DEFAULT_SPOT_SYMBOLS,
    TOP_VOLUME_BASES,
)


class TestTopVolumeBases:
    """Tests for TOP_VOLUME_BASES constant."""

    def test_is_list(self) -> None:
        assert isinstance(TOP_VOLUME_BASES, list)

    def test_contains_btc(self) -> None:
        assert "BTC" in TOP_VOLUME_BASES

    def test_contains_eth(self) -> None:
        assert "ETH" in TOP_VOLUME_BASES

    def test_has_ten_bases(self) -> None:
        assert len(TOP_VOLUME_BASES) == 10

    def test_btc_is_first(self) -> None:
        # BTC should be first since it has highest volume
        assert TOP_VOLUME_BASES[0] == "BTC"

    def test_eth_is_second(self) -> None:
        # ETH should be second since it has second highest volume
        assert TOP_VOLUME_BASES[1] == "ETH"


class TestDefaultSpotSymbols:
    """Tests for DEFAULT_SPOT_SYMBOLS constant."""

    def test_is_list(self) -> None:
        assert isinstance(DEFAULT_SPOT_SYMBOLS, list)

    def test_has_ten_symbols(self) -> None:
        assert len(DEFAULT_SPOT_SYMBOLS) == 10

    def test_all_symbols_end_with_usd(self) -> None:
        for symbol in DEFAULT_SPOT_SYMBOLS:
            assert symbol.endswith("-USD"), f"{symbol} does not end with -USD"

    def test_contains_btc_usd(self) -> None:
        assert "BTC-USD" in DEFAULT_SPOT_SYMBOLS

    def test_contains_eth_usd(self) -> None:
        assert "ETH-USD" in DEFAULT_SPOT_SYMBOLS

    def test_symbols_match_bases(self) -> None:
        expected = [f"{base}-USD" for base in TOP_VOLUME_BASES]
        assert DEFAULT_SPOT_SYMBOLS == expected


class TestDefaultSpotRiskPath:
    """Tests for DEFAULT_SPOT_RISK_PATH constant."""

    def test_is_path(self) -> None:
        assert isinstance(DEFAULT_SPOT_RISK_PATH, Path)

    def test_ends_with_spot_top10_json(self) -> None:
        assert DEFAULT_SPOT_RISK_PATH.name == "spot_top10.json"

    def test_contains_config_in_path(self) -> None:
        assert "config" in str(DEFAULT_SPOT_RISK_PATH)

    def test_contains_risk_in_path(self) -> None:
        assert "risk" in str(DEFAULT_SPOT_RISK_PATH)
