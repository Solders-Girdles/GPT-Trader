"""
Enhanced YFinance data source with additional market data features.
Provides dividends, splits, earnings, and other data for more sophisticated strategies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.logging import get_logger

logger = get_logger("enhanced_data")


@dataclass
class EnhancedMarketData:
    """Enhanced market data with additional features."""

    # Basic OHLCV data
    ohlcv: pd.DataFrame

    # Additional market data
    dividends: pd.DataFrame
    splits: pd.DataFrame
    earnings: pd.DataFrame
    info: dict[str, Any]

    # Calculated features
    volume_indicators: pd.DataFrame
    price_indicators: pd.DataFrame
    volatility_indicators: pd.DataFrame


class EnhancedYFinanceSource(YFinanceSource):
    """Enhanced YFinance source with additional data features."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cache_dir = Path(os.getenv("YF_CACHE_DIR", "data/cache/yf_enhanced"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_enhanced_data(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        include_dividends: bool = True,
        include_splits: bool = True,
        include_earnings: bool = True,
        include_info: bool = True,
    ) -> EnhancedMarketData:
        """Get enhanced market data with additional features."""

        # Get basic OHLCV data
        ohlcv = self.get_daily_bars(symbol, start, end)

        # Initialize additional data
        dividends = pd.DataFrame()
        splits = pd.DataFrame()
        earnings = pd.DataFrame()
        info = {}

        try:
            ticker = yf.Ticker(symbol)

            # Get dividends
            if include_dividends:
                dividends = self._get_dividends(ticker, start, end)

            # Get splits
            if include_splits:
                splits = self._get_splits(ticker, start, end)

            # Get earnings
            if include_earnings:
                earnings = self._get_earnings(ticker)

            # Get company info
            if include_info:
                info = self._get_company_info(ticker)

        except Exception as e:
            logger.warning(f"Failed to get enhanced data for {symbol}: {e}")

        # Calculate additional indicators
        volume_indicators = self._calculate_volume_indicators(ohlcv)
        price_indicators = self._calculate_price_indicators(ohlcv)
        volatility_indicators = self._calculate_volatility_indicators(ohlcv)

        return EnhancedMarketData(
            ohlcv=ohlcv,
            dividends=dividends,
            splits=splits,
            earnings=earnings,
            info=info,
            volume_indicators=volume_indicators,
            price_indicators=price_indicators,
            volatility_indicators=volatility_indicators,
        )

    def _get_dividends(self, ticker: yf.Ticker, start: str | None, end: str | None) -> pd.DataFrame:
        """Get dividend data."""
        try:
            dividends = ticker.dividends
            if not dividends.empty:
                dividends = dividends.sort_index()
                if start:
                    dividends = dividends[dividends.index >= start]
                if end:
                    dividends = dividends[dividends.index <= end]
            return dividends
        except Exception as e:
            logger.debug(f"Failed to get dividends: {e}")
            return pd.DataFrame()

    def _get_splits(self, ticker: yf.Ticker, start: str | None, end: str | None) -> pd.DataFrame:
        """Get stock split data."""
        try:
            splits = ticker.splits
            if not splits.empty:
                splits = splits.sort_index()
                if start:
                    splits = splits[splits.index >= start]
                if end:
                    splits = splits[splits.index <= end]
            return splits
        except Exception as e:
            logger.debug(f"Failed to get splits: {e}")
            return pd.DataFrame()

    def _get_earnings(self, ticker: yf.Ticker) -> pd.DataFrame:
        """Get earnings data."""
        try:
            earnings = ticker.earnings
            if not earnings.empty:
                earnings = earnings.sort_index()
            return earnings
        except Exception as e:
            logger.debug(f"Failed to get earnings: {e}")
            return pd.DataFrame()

    def _get_company_info(self, ticker: yf.Ticker) -> dict[str, Any]:
        """Get company information."""
        try:
            info = ticker.info
            # Filter out large objects that might cause issues
            filtered_info = {}
            for key, value in info.items():
                if isinstance(value, str | int | float | bool) and value is not None:
                    filtered_info[key] = value
            return filtered_info
        except Exception as e:
            logger.debug(f"Failed to get company info: {e}")
            return {}

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        indicators = pd.DataFrame(index=df.index)

        if "Volume" not in df.columns:
            return indicators

        # Volume moving averages
        indicators["volume_sma_5"] = df["Volume"].rolling(window=5).mean()
        indicators["volume_sma_20"] = df["Volume"].rolling(window=20).mean()
        indicators["volume_sma_50"] = df["Volume"].rolling(window=50).mean()

        # Volume ratios
        indicators["volume_ratio_5"] = df["Volume"] / indicators["volume_sma_5"]
        indicators["volume_ratio_20"] = df["Volume"] / indicators["volume_sma_20"]
        indicators["volume_ratio_50"] = df["Volume"] / indicators["volume_sma_50"]

        # Volume momentum
        indicators["volume_momentum_5"] = df["Volume"].pct_change(5)
        indicators["volume_momentum_20"] = df["Volume"].pct_change(20)

        # Volume volatility
        indicators["volume_volatility_20"] = (
            df["Volume"].rolling(window=20).std() / indicators["volume_sma_20"]
        )

        # Volume trend
        indicators["volume_trend_20"] = (
            indicators["volume_sma_5"] - indicators["volume_sma_20"]
        ) / indicators["volume_sma_20"]

        return indicators

    def _calculate_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based indicators."""
        indicators = pd.DataFrame(index=df.index)

        # Price momentum
        indicators["price_momentum_1"] = df["Close"].pct_change(1)
        indicators["price_momentum_5"] = df["Close"].pct_change(5)
        indicators["price_momentum_20"] = df["Close"].pct_change(20)
        indicators["price_momentum_50"] = df["Close"].pct_change(50)

        # Price relative to moving averages
        indicators["price_vs_sma_5"] = df["Close"] / df["Close"].rolling(window=5).mean() - 1
        indicators["price_vs_sma_20"] = df["Close"] / df["Close"].rolling(window=20).mean() - 1
        indicators["price_vs_sma_50"] = df["Close"] / df["Close"].rolling(window=50).mean() - 1
        indicators["price_vs_sma_200"] = df["Close"] / df["Close"].rolling(window=200).mean() - 1

        # Price ranges
        indicators["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
        indicators["gap_up"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        indicators["gap_down"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

        # Price efficiency
        indicators["price_efficiency_20"] = self._calculate_price_efficiency(df["Close"], 20)
        indicators["price_efficiency_50"] = self._calculate_price_efficiency(df["Close"], 50)

        return indicators

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based indicators."""
        indicators = pd.DataFrame(index=df.index)

        # Returns
        returns = df["Close"].pct_change()

        # Rolling volatility
        indicators["volatility_5"] = returns.rolling(window=5).std() * np.sqrt(252)
        indicators["volatility_20"] = returns.rolling(window=20).std() * np.sqrt(252)
        indicators["volatility_50"] = returns.rolling(window=50).std() * np.sqrt(252)

        # Volatility ratios
        indicators["vol_ratio_5_20"] = indicators["volatility_5"] / indicators["volatility_20"]
        indicators["vol_ratio_20_50"] = indicators["volatility_20"] / indicators["volatility_50"]

        # Volatility momentum
        indicators["vol_momentum_5"] = indicators["volatility_20"].pct_change(5)
        indicators["vol_momentum_20"] = indicators["volatility_20"].pct_change(20)

        # Realized volatility
        indicators["realized_vol_20"] = returns.rolling(window=20).apply(
            lambda x: np.sqrt(np.sum(x**2) * 252 / len(x))
        )

        # Parkinson volatility (using high-low range)
        indicators["parkinson_vol_20"] = self._calculate_parkinson_volatility(df, 20)

        return indicators

    def _calculate_price_efficiency(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate price efficiency (random walk measure)."""

        def efficiency(x):
            if len(x) < 2:
                return 0.0
            # Calculate the ratio of actual distance to path length
            actual_distance = abs(x.iloc[-1] - x.iloc[0])
            path_length = np.sum(np.abs(x.diff().dropna()))
            if path_length == 0:
                return 0.0
            return actual_distance / path_length

        return prices.rolling(window=window).apply(efficiency)

    def _calculate_parkinson_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Parkinson volatility using high-low range."""

        def parkinson_vol(x):
            if len(x) < 2:
                return 0.0
            # Parkinson volatility formula
            log_hl = np.log(x["High"] / x["Low"])
            return np.sqrt(np.mean(log_hl**2) * 252 / (4 * np.log(2)))

        return df.rolling(window=window).apply(parkinson_vol)

    def get_market_regime_data(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Get market regime data for a symbol."""
        try:
            # Get enhanced data
            enhanced_data = self.get_enhanced_data(
                symbol,
                start,
                end,
                include_dividends=False,
                include_splits=False,
                include_earnings=False,
                include_info=False,
            )

            df = enhanced_data.ohlcv.copy()

            # Add regime indicators
            df["trend_20"] = df["Close"] > df["Close"].rolling(window=20).mean()
            df["trend_50"] = df["Close"] > df["Close"].rolling(window=50).mean()
            df["trend_200"] = df["Close"] > df["Close"].rolling(window=200).mean()

            # Volatility regime
            returns = df["Close"].pct_change()
            vol_20 = returns.rolling(window=20).std()
            vol_50 = returns.rolling(window=50).std()
            df["vol_regime"] = vol_20 / vol_50

            # Volume regime
            if "Volume" in df.columns:
                vol_ratio = df["Volume"] / df["Volume"].rolling(window=20).mean()
                df["volume_regime"] = vol_ratio

            # Combined regime
            df["regime_score"] = (
                df["trend_20"].astype(int)
                + df["trend_50"].astype(int)
                + df["trend_200"].astype(int)
                + (df["vol_regime"] < 1.2).astype(int)
                + (df.get("volume_regime", 1) > 1.0).astype(int)
            ) / 5.0

            return df

        except Exception as e:
            logger.error(f"Failed to get market regime data for {symbol}: {e}")
            return pd.DataFrame()

    def get_correlation_data(
        self, symbols: list[str], start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Get correlation data for multiple symbols."""
        try:
            # Get price data for all symbols
            price_data = {}
            for symbol in symbols:
                try:
                    ohlcv = self.get_daily_bars(symbol, start, end)
                    price_data[symbol] = ohlcv["Close"]
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")

            if not price_data:
                return pd.DataFrame()

            # Create price matrix
            price_df = pd.DataFrame(price_data)

            # Calculate rolling correlations
            correlation_data = pd.DataFrame(index=price_df.index)

            for i, symbol1 in enumerate(symbols):
                for _j, symbol2 in enumerate(symbols[i + 1 :], i + 1):
                    if symbol1 in price_df.columns and symbol2 in price_df.columns:
                        corr_20 = price_df[symbol1].rolling(window=20).corr(price_df[symbol2])
                        corr_50 = price_df[symbol1].rolling(window=50).corr(price_df[symbol2])

                        correlation_data[f"{symbol1}_{symbol2}_corr_20"] = corr_20
                        correlation_data[f"{symbol1}_{symbol2}_corr_50"] = corr_50

            return correlation_data

        except Exception as e:
            logger.error(f"Failed to get correlation data: {e}")
            return pd.DataFrame()

    def get_sector_data(
        self, sector_symbols: dict[str, list[str]], start: str | None = None, end: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """Get sector-specific data."""
        sector_data = {}

        for sector, symbols in sector_symbols.items():
            try:
                # Get data for all symbols in sector
                sector_prices = {}
                for symbol in symbols:
                    try:
                        ohlcv = self.get_daily_bars(symbol, start, end)
                        sector_prices[symbol] = ohlcv["Close"]
                    except Exception as e:
                        logger.debug(f"Failed to get data for {symbol} in {sector}: {e}")

                if sector_prices:
                    # Create sector price index
                    sector_df = pd.DataFrame(sector_prices)
                    sector_df["sector_index"] = sector_df.mean(axis=1)

                    # Calculate sector indicators
                    sector_df["sector_momentum_20"] = sector_df["sector_index"].pct_change(20)
                    sector_df["sector_volatility_20"] = (
                        sector_df["sector_index"].pct_change().rolling(window=20).std()
                    )

                    sector_data[sector] = sector_df

            except Exception as e:
                logger.error(f"Failed to get sector data for {sector}: {e}")

        return sector_data
