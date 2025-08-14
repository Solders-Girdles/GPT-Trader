"""
GPT-Trader Unified Data Pipeline
Consolidated data management for all data sources and streams
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data pipeline"""
    
    # Data sources
    primary_source: str = "yfinance"  # yfinance, alpaca, ibkr
    fallback_sources: List[str] = field(default_factory=lambda: ["yfinance"])
    
    # Caching
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    cache_ttl_seconds: int = 300  # 5 minutes
    
    # Streaming
    stream_enabled: bool = False
    stream_buffer_size: int = 1000
    
    # Validation
    validate_data: bool = True
    repair_data: bool = True
    
    # Performance
    batch_size: int = 100
    max_concurrent_requests: int = 5


class UnifiedDataPipeline:
    """
    Unified data pipeline that consolidates:
    - Historical data fetching
    - Real-time streaming
    - Data caching
    - Data validation
    - Multiple data source management
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        
        # Data sources
        self._sources = {}
        self._init_sources()
        
        # Cache
        self._cache = {}
        self._cache_timestamps = {}
        
        # Streaming
        self._stream_handlers = []
        self._stream_buffer = []
        self._streaming = False
        
        # Ensure cache directory exists
        if self.config.cache_enabled:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Unified data pipeline initialized")
    
    def _init_sources(self) -> None:
        """Initialize data sources"""
        # Import source modules as needed
        if "yfinance" in [self.config.primary_source] + self.config.fallback_sources:
            from ..dataflow.sources.yfinance_source import YFinanceSource
            self._sources["yfinance"] = YFinanceSource()
        
        # Add more sources as needed (alpaca, ibkr, etc.)
    
    async def fetch_historical(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
        source: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data with caching and fallback
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1m, 5m, 1h, 1d, etc.)
            source: Specific source to use (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
        if self.config.cache_enabled:
            cached_data = self._get_cached(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data
        
        # Determine source to use
        source = source or self.config.primary_source
        
        # Try primary source
        try:
            data = await self._fetch_from_source(
                source, symbol, start_date, end_date, interval
            )
            
            if data is not None and not data.empty:
                # Validate and repair if configured
                if self.config.validate_data:
                    data = self._validate_data(data, repair=self.config.repair_data)
                
                # Cache the data
                if self.config.cache_enabled:
                    self._cache_data(cache_key, data)
                
                return data
                
        except Exception as e:
            logger.warning(f"Primary source {source} failed: {e}")
        
        # Try fallback sources
        for fallback in self.config.fallback_sources:
            if fallback == source:
                continue  # Skip already tried source
                
            try:
                logger.info(f"Trying fallback source: {fallback}")
                data = await self._fetch_from_source(
                    fallback, symbol, start_date, end_date, interval
                )
                
                if data is not None and not data.empty:
                    # Validate and repair
                    if self.config.validate_data:
                        data = self._validate_data(data, repair=self.config.repair_data)
                    
                    # Cache the data
                    if self.config.cache_enabled:
                        self._cache_data(cache_key, data)
                    
                    return data
                    
            except Exception as e:
                logger.warning(f"Fallback source {fallback} failed: {e}")
        
        # All sources failed
        raise Exception(f"Failed to fetch data for {symbol} from all sources")
    
    async def _fetch_from_source(
        self,
        source: str,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str
    ) -> pd.DataFrame:
        """Fetch data from specific source"""
        if source not in self._sources:
            raise ValueError(f"Unknown data source: {source}")
        
        source_obj = self._sources[source]
        
        # Convert dates to appropriate format
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Fetch data (make async if source supports it)
        return source_obj.fetch(symbol, start_date, end_date, interval)
    
    def fetch_realtime(self, symbols: List[str], handler: Callable) -> None:
        """
        Start real-time data streaming
        
        Args:
            symbols: List of symbols to stream
            handler: Callback function for data updates
        """
        if not self.config.stream_enabled:
            raise RuntimeError("Streaming not enabled in configuration")
        
        self._stream_handlers.append(handler)
        
        # Start streaming (implementation depends on source)
        if not self._streaming:
            self._streaming = True
            asyncio.create_task(self._stream_loop(symbols))
    
    async def _stream_loop(self, symbols: List[str]) -> None:
        """Main streaming loop"""
        logger.info(f"Starting stream for symbols: {symbols}")
        
        while self._streaming:
            try:
                # This would connect to actual streaming source
                # For now, simulate with periodic updates
                await asyncio.sleep(1)
                
                # Generate mock tick data
                for symbol in symbols:
                    tick = {
                        'symbol': symbol,
                        'price': 100.0,  # Mock price
                        'volume': 1000,
                        'timestamp': datetime.now()
                    }
                    
                    # Add to buffer
                    self._stream_buffer.append(tick)
                    if len(self._stream_buffer) > self.config.stream_buffer_size:
                        self._stream_buffer.pop(0)
                    
                    # Call handlers
                    for handler in self._stream_handlers:
                        try:
                            handler(tick)
                        except Exception as e:
                            logger.error(f"Stream handler error: {e}")
                            
            except Exception as e:
                logger.error(f"Stream loop error: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    def stop_streaming(self) -> None:
        """Stop real-time streaming"""
        self._streaming = False
        self._stream_handlers.clear()
        logger.info("Streaming stopped")
    
    def _validate_data(self, data: pd.DataFrame, repair: bool = True) -> pd.DataFrame:
        """
        Validate and optionally repair data
        
        Args:
            data: DataFrame to validate
            repair: Whether to attempt repairs
            
        Returns:
            Validated/repaired DataFrame
        """
        if data.empty:
            return data
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            if not repair:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Add missing columns with default values
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 0
                else:
                    data[col] = data.get('close', 0)
        
        # Check for invalid values
        if repair:
            # Remove rows with all NaN
            data = data.dropna(how='all')
            
            # Forward fill NaN values
            data = data.ffill()
            
            # Ensure positive prices
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns:
                    data[col] = data[col].clip(lower=0.01)
            
            # Ensure high >= low
            if 'high' in data.columns and 'low' in data.columns:
                data['high'] = data[['high', 'low']].max(axis=1)
                data['low'] = data[['high', 'low']].min(axis=1)
            
            # Ensure volume is non-negative
            if 'volume' in data.columns:
                data['volume'] = data['volume'].clip(lower=0)
        
        return data
    
    def _get_cached(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if valid"""
        if cache_key not in self._cache:
            return None
        
        # Check if cache is still valid
        timestamp = self._cache_timestamps.get(cache_key)
        if timestamp:
            age = (datetime.now() - timestamp).total_seconds()
            if age > self.config.cache_ttl_seconds:
                # Cache expired
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]
                return None
        
        return self._cache[cache_key].copy()
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Store data in cache"""
        self._cache[cache_key] = data.copy()
        self._cache_timestamps[cache_key] = datetime.now()
        
        # Also save to disk if configured
        if self.config.cache_dir:
            cache_file = self.config.cache_dir / f"{cache_key}.parquet"
            try:
                data.to_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Failed to save cache to disk: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._cache.clear()
        self._cache_timestamps.clear()
        
        # Clear disk cache
        if self.config.cache_dir and self.config.cache_dir.exists():
            for cache_file in self.config.cache_dir.glob("*.parquet"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info("Cache cleared")
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources"""
        return list(self._sources.keys())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = 0
        if self.config.cache_dir and self.config.cache_dir.exists():
            for cache_file in self.config.cache_dir.glob("*.parquet"):
                total_size += cache_file.stat().st_size
        
        return {
            'entries': len(self._cache),
            'memory_items': len(self._cache),
            'disk_size_mb': total_size / (1024 * 1024),
            'oldest_entry': min(self._cache_timestamps.values()) if self._cache_timestamps else None,
            'newest_entry': max(self._cache_timestamps.values()) if self._cache_timestamps else None
        }


# Global instance
_pipeline = None


def get_data_pipeline(config: Optional[DataConfig] = None) -> UnifiedDataPipeline:
    """Get or create global data pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = UnifiedDataPipeline(config)
    return _pipeline