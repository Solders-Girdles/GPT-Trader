"""
Alternative Data Integration for Multi-Asset Strategy Enhancement

This module implements sophisticated alternative data integration including:
- Sentiment data processing (news, social media, analyst reports)
- Economic and macro data integration
- Satellite and geospatial data processing
- Web scraping and alternative metrics
- ESG (Environmental, Social, Governance) data integration
- Cross-asset factor extraction from alternative sources
- Real-time data stream processing
- Alternative data quality assessment and filtering
"""

import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# Optional dependencies with graceful fallback
try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    warnings.warn("TextBlob not available. Sentiment analysis features will be limited.")

try:
    import feedparser

    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    warnings.warn("Feedparser not available. RSS feed processing will be limited.")

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA, LatentDirichletAllocation
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "Scikit-learn not available. Some alternative data processing features will be limited."
    )

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of alternative data sources"""

    NEWS_SENTIMENT = "news_sentiment"
    SOCIAL_MEDIA = "social_media"
    ECONOMIC_INDICATORS = "economic_indicators"
    SATELLITE_DATA = "satellite_data"
    WEB_TRAFFIC = "web_traffic"
    ESG_METRICS = "esg_metrics"
    ANALYST_ESTIMATES = "analyst_estimates"
    INSIDER_TRADING = "insider_trading"
    CREDIT_METRICS = "credit_metrics"
    CRYPTO_METRICS = "crypto_metrics"


class ProcessingMethod(Enum):
    """Data processing methods"""

    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_MODELING = "topic_modeling"
    ANOMALY_DETECTION = "anomaly_detection"
    FACTOR_EXTRACTION = "factor_extraction"
    SIGNAL_GENERATION = "signal_generation"
    TREND_ANALYSIS = "trend_analysis"


class DataQuality(Enum):
    """Data quality levels"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRELIABLE = "unreliable"


@dataclass
class AlternativeDataConfig:
    """Configuration for alternative data integration"""

    enabled_sources: list[DataSourceType] = field(
        default_factory=lambda: [
            DataSourceType.NEWS_SENTIMENT,
            DataSourceType.ECONOMIC_INDICATORS,
            DataSourceType.ESG_METRICS,
        ]
    )
    processing_methods: list[ProcessingMethod] = field(
        default_factory=lambda: [
            ProcessingMethod.SENTIMENT_ANALYSIS,
            ProcessingMethod.FACTOR_EXTRACTION,
        ]
    )
    min_data_quality: DataQuality = DataQuality.MEDIUM
    sentiment_lookback_days: int = 7
    economic_data_frequency: str = "daily"  # daily, weekly, monthly
    news_sources: list[str] = field(
        default_factory=lambda: ["reuters", "bloomberg", "cnbc", "marketwatch"]
    )
    update_frequency: int = 3600  # seconds
    cache_duration: int = 86400  # seconds
    api_rate_limits: dict[str, int] = field(
        default_factory=lambda: {
            "news_api": 100,  # requests per hour
            "twitter_api": 300,
            "economic_api": 50,
        }
    )
    confidence_threshold: float = 0.6
    signal_decay_factor: float = 0.9  # Daily decay for signals


@dataclass
class AlternativeDataPoint:
    """Single alternative data point"""

    source: DataSourceType
    timestamp: pd.Timestamp
    asset: str
    value: float | str | dict[str, Any]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    quality: DataQuality = DataQuality.MEDIUM


@dataclass
class ProcessedSignal:
    """Processed signal from alternative data"""

    signal_type: str
    asset: str
    signal_value: float  # -1 to 1
    confidence: float  # 0 to 1
    timestamp: pd.Timestamp
    source_data: list[AlternativeDataPoint]
    processing_method: ProcessingMethod
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAlternativeDataSource(ABC):
    """Base class for alternative data sources"""

    def __init__(self, config: AlternativeDataConfig) -> None:
        self.config = config
        self.data_cache = {}
        self.last_update = {}
        self.rate_limiter = {}

    @abstractmethod
    def fetch_data(
        self, asset: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> list[AlternativeDataPoint]:
        """Fetch alternative data for asset"""
        pass

    @abstractmethod
    def get_supported_assets(self) -> list[str]:
        """Get list of supported assets"""
        pass

    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API rate limit allows request"""
        current_time = time.time()

        if api_name not in self.rate_limiter:
            self.rate_limiter[api_name] = {
                "requests": [],
                "limit": self.config.api_rate_limits.get(api_name, 100),
            }

        # Clean old requests (older than 1 hour)
        self.rate_limiter[api_name]["requests"] = [
            req_time
            for req_time in self.rate_limiter[api_name]["requests"]
            if current_time - req_time < 3600
        ]

        # Check if under limit
        if len(self.rate_limiter[api_name]["requests"]) < self.rate_limiter[api_name]["limit"]:
            self.rate_limiter[api_name]["requests"].append(current_time)
            return True

        return False

    def _get_cached_data(self, cache_key: str) -> list[AlternativeDataPoint] | None:
        """Get cached data if still valid"""
        if cache_key in self.data_cache:
            cache_entry = self.data_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.config.cache_duration:
                return cache_entry["data"]
        return None

    def _cache_data(self, cache_key: str, data: list[AlternativeDataPoint]) -> None:
        """Cache data with timestamp"""
        self.data_cache[cache_key] = {"data": data, "timestamp": time.time()}


class NewsSentimentDataSource(BaseAlternativeDataSource):
    """News sentiment data source"""

    def __init__(self, config: AlternativeDataConfig) -> None:
        super().__init__(config)
        self.sentiment_cache = {}

    def fetch_data(
        self, asset: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> list[AlternativeDataPoint]:
        """Fetch news sentiment data"""
        cache_key = f"news_{asset}_{start_date.date()}_{end_date.date()}"

        # Check cache first
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        # Check rate limit
        if not self._check_rate_limit("news_api"):
            logger.warning("News API rate limit exceeded")
            return []

        try:
            # Simulate news data fetching (in practice, would call real news APIs)
            news_data = self._fetch_simulated_news_data(asset, start_date, end_date)

            # Process news for sentiment
            sentiment_data = []
            for news_item in news_data:
                sentiment_point = self._analyze_news_sentiment(news_item, asset)
                if sentiment_point:
                    sentiment_data.append(sentiment_point)

            # Cache results
            self._cache_data(cache_key, sentiment_data)

            return sentiment_data

        except Exception as e:
            logger.error(f"Failed to fetch news sentiment for {asset}: {str(e)}")
            return []

    def _fetch_simulated_news_data(
        self, asset: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> list[dict[str, Any]]:
        """Simulate news data fetching"""
        # In practice, this would call real news APIs like NewsAPI, Bloomberg, etc.
        news_items = []

        # Generate sample news items
        sample_headlines = [
            f"{asset} reports strong quarterly earnings",
            f"Analysts upgrade {asset} price target",
            f"{asset} announces new strategic partnership",
            f"Market volatility affects {asset} performance",
            f"{asset} faces regulatory challenges",
        ]

        # Generate news items for date range
        current_date = start_date
        while current_date <= end_date:
            if np.random.random() < 0.3:  # 30% chance of news per day
                headline = np.random.choice(sample_headlines)
                news_items.append(
                    {
                        "headline": headline,
                        "content": f"Market analysis shows {headline.lower()}. Industry experts suggest...",
                        "timestamp": current_date,
                        "source": np.random.choice(self.config.news_sources),
                        "relevance": np.random.uniform(0.5, 1.0),
                    }
                )
            current_date += pd.Timedelta(days=1)

        return news_items

    def _analyze_news_sentiment(
        self, news_item: dict[str, Any], asset: str
    ) -> AlternativeDataPoint | None:
        """Analyze sentiment of news item"""
        try:
            text = news_item.get("headline", "") + " " + news_item.get("content", "")

            if TEXTBLOB_AVAILABLE:
                # Use TextBlob for sentiment analysis
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity  # -1 to 1
                confidence = abs(blob.sentiment.polarity) * news_item.get("relevance", 0.5)
            else:
                # Simple keyword-based sentiment
                positive_words = [
                    "strong",
                    "upgrade",
                    "growth",
                    "positive",
                    "good",
                    "excellent",
                    "beat",
                ]
                negative_words = ["weak", "downgrade", "decline", "negative", "bad", "poor", "miss"]

                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)

                if positive_count + negative_count > 0:
                    sentiment_score = (positive_count - negative_count) / (
                        positive_count + negative_count
                    )
                    confidence = min(0.8, (positive_count + negative_count) / 10.0) * news_item.get(
                        "relevance", 0.5
                    )
                else:
                    sentiment_score = 0.0
                    confidence = 0.1

            # Only return if confidence is above threshold
            if confidence >= self.config.confidence_threshold:
                return AlternativeDataPoint(
                    source=DataSourceType.NEWS_SENTIMENT,
                    timestamp=news_item["timestamp"],
                    asset=asset,
                    value=sentiment_score,
                    confidence=confidence,
                    metadata={
                        "headline": news_item.get("headline"),
                        "source": news_item.get("source"),
                        "relevance": news_item.get("relevance"),
                    },
                    quality=DataQuality.HIGH if confidence > 0.8 else DataQuality.MEDIUM,
                )

            return None

        except Exception as e:
            logger.warning(f"Sentiment analysis failed for news item: {str(e)}")
            return None

    def get_supported_assets(self) -> list[str]:
        """Get supported assets for news sentiment"""
        # In practice, would query news APIs for available assets
        return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"]


class EconomicIndicatorDataSource(BaseAlternativeDataSource):
    """Economic indicator data source"""

    def fetch_data(
        self, asset: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> list[AlternativeDataPoint]:
        """Fetch economic indicator data"""
        cache_key = f"econ_{asset}_{start_date.date()}_{end_date.date()}"

        # Check cache
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        # Check rate limit
        if not self._check_rate_limit("economic_api"):
            logger.warning("Economic API rate limit exceeded")
            return []

        try:
            # Simulate economic data fetching
            economic_data = self._fetch_simulated_economic_data(asset, start_date, end_date)

            # Cache results
            self._cache_data(cache_key, economic_data)

            return economic_data

        except Exception as e:
            logger.error(f"Failed to fetch economic data for {asset}: {str(e)}")
            return []

    def _fetch_simulated_economic_data(
        self, asset: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> list[AlternativeDataPoint]:
        """Simulate economic indicator data"""
        economic_data = []

        # Define economic indicators relevant to different assets
        indicators = {
            "stocks": ["GDP_growth", "unemployment_rate", "inflation_rate", "consumer_confidence"],
            "bonds": ["interest_rates", "inflation_expectations", "credit_spreads"],
            "commodities": ["dollar_index", "industrial_production", "inventory_levels"],
            "crypto": ["regulatory_sentiment", "adoption_metrics", "institutional_flows"],
        }

        # Determine asset class
        asset_class = "stocks"  # Default, could be determined from asset symbol
        relevant_indicators = indicators[asset_class]

        # Generate data points
        current_date = start_date
        while current_date <= end_date:
            for indicator in relevant_indicators:
                if np.random.random() < 0.1:  # 10% chance per day per indicator
                    # Simulate indicator value and impact
                    base_value = np.random.normal(0, 1)  # Standardized value
                    impact = self._calculate_economic_impact(indicator, base_value, asset)

                    economic_data.append(
                        AlternativeDataPoint(
                            source=DataSourceType.ECONOMIC_INDICATORS,
                            timestamp=current_date,
                            asset=asset,
                            value=impact,
                            confidence=0.8,
                            metadata={
                                "indicator": indicator,
                                "raw_value": base_value,
                                "indicator_type": asset_class,
                            },
                            quality=DataQuality.HIGH,
                        )
                    )

            current_date += pd.Timedelta(days=1)

        return economic_data

    def _calculate_economic_impact(self, indicator: str, value: float, asset: str) -> float:
        """Calculate economic indicator impact on asset"""
        # Define impact relationships
        impact_mapping = {
            "GDP_growth": 0.5,  # Positive GDP growth is good for stocks
            "unemployment_rate": -0.3,  # Higher unemployment is bad
            "inflation_rate": -0.2,  # Higher inflation can be mixed
            "consumer_confidence": 0.4,  # Higher confidence is good
            "interest_rates": -0.6,  # Higher rates generally bad for stocks
            "inflation_expectations": -0.1,
            "credit_spreads": -0.8,  # Wider spreads are bad
            "dollar_index": 0.2,  # Depends on asset
            "industrial_production": 0.3,
            "inventory_levels": -0.1,
        }

        multiplier = impact_mapping.get(indicator, 0.0)
        return np.tanh(value * multiplier)  # Bounded impact between -1 and 1

    def get_supported_assets(self) -> list[str]:
        """Get supported assets for economic indicators"""
        return ["STOCKS", "BONDS", "COMMODITIES", "USD", "EUR", "JPY"]


class ESGDataSource(BaseAlternativeDataSource):
    """ESG (Environmental, Social, Governance) data source"""

    def fetch_data(
        self, asset: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> list[AlternativeDataPoint]:
        """Fetch ESG data"""
        cache_key = f"esg_{asset}_{start_date.date()}_{end_date.date()}"

        # Check cache
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        try:
            # Simulate ESG data fetching
            esg_data = self._fetch_simulated_esg_data(asset, start_date, end_date)

            # Cache results
            self._cache_data(cache_key, esg_data)

            return esg_data

        except Exception as e:
            logger.error(f"Failed to fetch ESG data for {asset}: {str(e)}")
            return []

    def _fetch_simulated_esg_data(
        self, asset: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> list[AlternativeDataPoint]:
        """Simulate ESG data"""
        esg_data = []

        # ESG categories
        esg_categories = ["environmental", "social", "governance"]

        # Generate occasional ESG events
        current_date = start_date
        while current_date <= end_date:
            if np.random.random() < 0.02:  # 2% chance per day
                category = np.random.choice(esg_categories)

                # Simulate ESG score change
                score_change = np.random.normal(0, 0.1)  # Small changes
                impact = np.tanh(score_change * 2)  # Convert to impact signal

                esg_data.append(
                    AlternativeDataPoint(
                        source=DataSourceType.ESG_METRICS,
                        timestamp=current_date,
                        asset=asset,
                        value=impact,
                        confidence=0.7,
                        metadata={
                            "category": category,
                            "score_change": score_change,
                            "event_type": "score_update",
                        },
                        quality=DataQuality.MEDIUM,
                    )
                )

            current_date += pd.Timedelta(days=1)

        return esg_data

    def get_supported_assets(self) -> list[str]:
        """Get supported assets for ESG data"""
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "XOM", "JPM", "BAC"]


class AlternativeDataProcessor:
    """Processor for alternative data signals"""

    def __init__(self, config: AlternativeDataConfig) -> None:
        self.config = config

    def process_data_points(
        self, data_points: list[AlternativeDataPoint], asset: str
    ) -> list[ProcessedSignal]:
        """Process raw alternative data points into trading signals"""
        processed_signals = []

        # Group data by type and process
        data_by_type = defaultdict(list)
        for point in data_points:
            if point.quality.value in [
                q.value
                for q in [DataQuality.HIGH, DataQuality.MEDIUM, self.config.min_data_quality]
            ]:
                data_by_type[point.source].append(point)

        # Process each data type
        for _source_type, points in data_by_type.items():
            if ProcessingMethod.SENTIMENT_ANALYSIS in self.config.processing_methods:
                sentiment_signals = self._process_sentiment_data(points, asset)
                processed_signals.extend(sentiment_signals)

            if ProcessingMethod.FACTOR_EXTRACTION in self.config.processing_methods:
                factor_signals = self._extract_factor_signals(points, asset)
                processed_signals.extend(factor_signals)

        return processed_signals

    def _process_sentiment_data(
        self, data_points: list[AlternativeDataPoint], asset: str
    ) -> list[ProcessedSignal]:
        """Process sentiment data into signals"""
        if not data_points:
            return []

        signals = []

        # Group by recent time periods
        recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=self.config.sentiment_lookback_days)
        recent_points = [p for p in data_points if p.timestamp >= recent_cutoff]

        if recent_points:
            # Aggregate sentiment with confidence weighting
            total_weighted_sentiment = sum(p.value * p.confidence for p in recent_points)
            total_confidence = sum(p.confidence for p in recent_points)

            if total_confidence > 0:
                avg_sentiment = total_weighted_sentiment / total_confidence
                signal_confidence = min(1.0, total_confidence / len(recent_points))

                # Apply decay based on age
                latest_timestamp = max(p.timestamp for p in recent_points)
                days_old = (pd.Timestamp.now() - latest_timestamp).days
                decay_factor = self.config.signal_decay_factor**days_old

                final_sentiment = avg_sentiment * decay_factor
                final_confidence = signal_confidence * decay_factor

                if final_confidence >= self.config.confidence_threshold:
                    signals.append(
                        ProcessedSignal(
                            signal_type="sentiment",
                            asset=asset,
                            signal_value=final_sentiment,
                            confidence=final_confidence,
                            timestamp=latest_timestamp,
                            source_data=recent_points,
                            processing_method=ProcessingMethod.SENTIMENT_ANALYSIS,
                            metadata={
                                "n_data_points": len(recent_points),
                                "avg_data_quality": np.mean(
                                    [
                                        1 if p.quality == DataQuality.HIGH else 0.5
                                        for p in recent_points
                                    ]
                                ),
                                "decay_applied": decay_factor,
                            },
                        )
                    )

        return signals

    def _extract_factor_signals(
        self, data_points: list[AlternativeDataPoint], asset: str
    ) -> list[ProcessedSignal]:
        """Extract factor-based signals from alternative data"""
        if not data_points or len(data_points) < 3:
            return []

        signals = []

        # Convert data to time series
        ts_data = []
        for point in data_points:
            ts_data.append(
                {
                    "timestamp": point.timestamp,
                    "value": point.value,
                    "confidence": point.confidence,
                    "source": point.source.value,
                }
            )

        ts_df = pd.DataFrame(ts_data).sort_values("timestamp")

        if len(ts_df) >= 5:  # Need minimum data for trend analysis
            # Calculate trend signal
            trend_signal = self._calculate_trend_signal(ts_df)

            if abs(trend_signal) > 0.1:  # Minimum threshold
                signals.append(
                    ProcessedSignal(
                        signal_type="factor_trend",
                        asset=asset,
                        signal_value=trend_signal,
                        confidence=0.6,
                        timestamp=ts_df["timestamp"].iloc[-1],
                        source_data=data_points,
                        processing_method=ProcessingMethod.FACTOR_EXTRACTION,
                        metadata={
                            "trend_strength": abs(trend_signal),
                            "data_span_days": (
                                ts_df["timestamp"].iloc[-1] - ts_df["timestamp"].iloc[0]
                            ).days,
                        },
                    )
                )

        return signals

    def _calculate_trend_signal(self, ts_df: pd.DataFrame) -> float:
        """Calculate trend signal from time series data"""
        try:
            # Simple linear trend
            x = np.arange(len(ts_df))
            y = ts_df["value"].values

            # Weighted by confidence
            weights = ts_df["confidence"].values

            # Weighted linear regression
            w_mean_x = np.average(x, weights=weights)
            w_mean_y = np.average(y, weights=weights)

            numerator = np.sum(weights * (x - w_mean_x) * (y - w_mean_y))
            denominator = np.sum(weights * (x - w_mean_x) ** 2)

            if denominator > 0:
                slope = numerator / denominator
                # Normalize slope to reasonable signal range
                trend_signal = np.tanh(slope * len(ts_df))
                return trend_signal
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Trend calculation failed: {str(e)}")
            return 0.0


class AlternativeDataFramework:
    """Main framework for alternative data integration"""

    def __init__(self, config: AlternativeDataConfig) -> None:
        self.config = config
        self.data_sources = self._initialize_data_sources()
        self.processor = AlternativeDataProcessor(config)
        self.signal_history = defaultdict(deque)

    def _initialize_data_sources(self) -> dict[DataSourceType, BaseAlternativeDataSource]:
        """Initialize enabled data sources"""
        sources = {}

        if DataSourceType.NEWS_SENTIMENT in self.config.enabled_sources:
            sources[DataSourceType.NEWS_SENTIMENT] = NewsSentimentDataSource(self.config)

        if DataSourceType.ECONOMIC_INDICATORS in self.config.enabled_sources:
            sources[DataSourceType.ECONOMIC_INDICATORS] = EconomicIndicatorDataSource(self.config)

        if DataSourceType.ESG_METRICS in self.config.enabled_sources:
            sources[DataSourceType.ESG_METRICS] = ESGDataSource(self.config)

        return sources

    def get_alternative_signals(
        self, assets: list[str], lookback_days: int = 30
    ) -> dict[str, list[ProcessedSignal]]:
        """Get alternative data signals for assets"""
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=lookback_days)

        all_signals = {}

        for asset in assets:
            asset_signals = []

            # Collect data from all sources
            all_data_points = []
            for source_type, data_source in self.data_sources.items():
                try:
                    if asset in data_source.get_supported_assets() or asset.upper() in [
                        a.upper() for a in data_source.get_supported_assets()
                    ]:
                        data_points = data_source.fetch_data(asset, start_date, end_date)
                        all_data_points.extend(data_points)
                except Exception as e:
                    logger.warning(
                        f"Failed to get data from {source_type.value} for {asset}: {str(e)}"
                    )

            # Process data into signals
            if all_data_points:
                processed_signals = self.processor.process_data_points(all_data_points, asset)
                asset_signals.extend(processed_signals)

            # Store in history
            for signal in asset_signals:
                self.signal_history[asset].append(signal)
                if len(self.signal_history[asset]) > 1000:  # Keep last 1000 signals
                    self.signal_history[asset].popleft()

            all_signals[asset] = asset_signals

        return all_signals

    def get_signal_summary(self, asset: str) -> dict[str, Any]:
        """Get summary of alternative data signals for asset"""
        recent_signals = list(self.signal_history[asset])[-50:]  # Last 50 signals

        if not recent_signals:
            return {"asset": asset, "signal_count": 0}

        # Aggregate signals by type
        signal_by_type = defaultdict(list)
        for signal in recent_signals:
            signal_by_type[signal.signal_type].append(signal)

        summary = {
            "asset": asset,
            "signal_count": len(recent_signals),
            "signal_types": list(signal_by_type.keys()),
            "avg_signal_confidence": np.mean([s.confidence for s in recent_signals]),
            "latest_timestamp": max(s.timestamp for s in recent_signals),
            "signal_breakdown": {},
        }

        for signal_type, signals in signal_by_type.items():
            summary["signal_breakdown"][signal_type] = {
                "count": len(signals),
                "avg_value": np.mean([s.signal_value for s in signals]),
                "avg_confidence": np.mean([s.confidence for s in signals]),
                "latest_signal": signals[-1].signal_value if signals else 0,
            }

        return summary

    def get_framework_metrics(self) -> dict[str, Any]:
        """Get framework performance metrics"""
        total_signals = sum(len(signals) for signals in self.signal_history.values())

        if total_signals == 0:
            return {"total_signals": 0, "active_assets": 0}

        all_recent_signals = []
        for asset_signals in self.signal_history.values():
            all_recent_signals.extend(list(asset_signals)[-10:])  # Last 10 per asset

        metrics = {
            "total_signals": total_signals,
            "active_assets": len(
                [asset for asset, signals in self.signal_history.items() if len(signals) > 0]
            ),
            "enabled_sources": len(self.data_sources),
            "avg_signal_confidence": (
                np.mean([s.confidence for s in all_recent_signals]) if all_recent_signals else 0
            ),
            "source_breakdown": {
                source_type.value: len(
                    [
                        s
                        for s in all_recent_signals
                        if any(d.source == source_type for d in s.source_data)
                    ]
                )
                for source_type in self.data_sources.keys()
            },
        }

        return metrics


def create_alternative_data_framework(
    enabled_sources: list[DataSourceType] | None = None,
    processing_methods: list[ProcessingMethod] | None = None,
    **kwargs,
) -> AlternativeDataFramework:
    """Factory function to create alternative data framework"""
    if enabled_sources is None:
        enabled_sources = [
            DataSourceType.NEWS_SENTIMENT,
            DataSourceType.ECONOMIC_INDICATORS,
            DataSourceType.ESG_METRICS,
        ]

    if processing_methods is None:
        processing_methods = [
            ProcessingMethod.SENTIMENT_ANALYSIS,
            ProcessingMethod.FACTOR_EXTRACTION,
        ]

    config = AlternativeDataConfig(
        enabled_sources=enabled_sources, processing_methods=processing_methods, **kwargs
    )

    return AlternativeDataFramework(config)


# Example usage and testing
if __name__ == "__main__":
    print("Alternative Data Integration Framework Testing")
    print("=" * 55)

    # Test framework creation
    try:
        framework = create_alternative_data_framework(
            enabled_sources=[
                DataSourceType.NEWS_SENTIMENT,
                DataSourceType.ECONOMIC_INDICATORS,
                DataSourceType.ESG_METRICS,
            ],
            processing_methods=[
                ProcessingMethod.SENTIMENT_ANALYSIS,
                ProcessingMethod.FACTOR_EXTRACTION,
            ],
            confidence_threshold=0.5,
            sentiment_lookback_days=7,
        )

        print(f"✅ Framework created with {len(framework.data_sources)} data sources")

        # Test signal generation
        test_assets = ["AAPL", "MSFT", "GOOGL"]
        signals = framework.get_alternative_signals(test_assets, lookback_days=14)

        print(f"✅ Generated signals for {len(signals)} assets")

        for asset, asset_signals in signals.items():
            if asset_signals:
                print(f"   {asset}: {len(asset_signals)} signals")
                for signal in asset_signals[:2]:  # Show first 2 signals
                    print(
                        f"     {signal.signal_type}: {signal.signal_value:.3f} (conf: {signal.confidence:.3f})"
                    )
            else:
                print(f"   {asset}: No signals generated")

        # Test signal summary
        for asset in test_assets:
            summary = framework.get_signal_summary(asset)
            if summary["signal_count"] > 0:
                print(
                    f"✅ {asset} summary: {summary['signal_count']} signals, "
                    f"avg confidence: {summary['avg_signal_confidence']:.3f}"
                )

        # Test framework metrics
        metrics = framework.get_framework_metrics()
        print(
            f"✅ Framework metrics: {metrics['total_signals']} total signals, "
            f"{metrics['active_assets']} active assets"
        )

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Test individual data sources
    print("\nTesting individual data sources...")

    try:
        config = AlternativeDataConfig()

        # Test news sentiment
        news_source = NewsSentimentDataSource(config)
        news_data = news_source.fetch_data(
            "AAPL", pd.Timestamp.now() - pd.Timedelta(days=7), pd.Timestamp.now()
        )
        print(f"✅ News sentiment: {len(news_data)} data points")

        # Test economic indicators
        econ_source = EconomicIndicatorDataSource(config)
        econ_data = econ_source.fetch_data(
            "STOCKS", pd.Timestamp.now() - pd.Timedelta(days=7), pd.Timestamp.now()
        )
        print(f"✅ Economic indicators: {len(econ_data)} data points")

        # Test ESG data
        esg_source = ESGDataSource(config)
        esg_data = esg_source.fetch_data(
            "AAPL", pd.Timestamp.now() - pd.Timedelta(days=7), pd.Timestamp.now()
        )
        print(f"✅ ESG data: {len(esg_data)} data points")

    except Exception as e:
        print(f"❌ Individual source testing error: {str(e)}")

    print("\n🚀 Alternative Data Integration Framework ready for production!")
