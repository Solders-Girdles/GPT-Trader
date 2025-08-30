"""
Simple data provider using yfinance for historical data.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any
import pandas as pd
import yfinance as yf
from core.interfaces import IDataProvider, ComponentConfig
from core.events import DataEvent, Event, EventType, get_event_bus
from core.types import MarketData


class SimpleDataProvider(IDataProvider):
    """
    Simple data provider that fetches historical data using yfinance.
    
    Features:
    - Fetches historical OHLCV data
    - Caches data to avoid repeated downloads
    - Publishes DATA_RECEIVED events
    - Validates data quality
    """
    
    def __init__(self, config: ComponentConfig):
        """
        Initialize the data provider.
        
        Args:
            config: Component configuration
        """
        super().__init__(config)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=15)  # Cache for 15 minutes
        self._event_bus = get_event_bus()
        
    def initialize(self) -> None:
        """Initialize the data provider."""
        # Nothing special needed for yfinance
        pass
    
    def shutdown(self) -> None:
        """Cleanup and shutdown."""
        self._cache.clear()
        self._cache_expiry.clear()
    
    def get_historical_data(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch historical market data.
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            interval: Data interval (1d, 1h, 5m, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_key = f"{symbol}_{start}_{end}_{interval}"
        if self._is_cached(cache_key):
            data = self._cache[cache_key]
            # Publish event
            self._publish_data_event(symbol, data, "cache")
            return data
        
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start,
                end=end,
                interval=interval
            )
            
            if data.empty:
                # Publish error event
                self._publish_error_event(symbol, "No data available")
                raise ValueError(f"No data available for {symbol}")
            
            # Clean and standardize data
            data = self._standardize_data(data)
            
            # Validate data
            if not self._validate_data(data):
                self._publish_error_event(symbol, "Invalid data received")
                raise ValueError(f"Invalid data for {symbol}")
            
            # Cache data
            self._cache_data(cache_key, data)
            
            # Publish success event
            self._publish_data_event(symbol, data, "download")
            
            return data
            
        except Exception as e:
            self._publish_error_event(symbol, str(e))
            raise
    
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, pd.Series]:
        """
        Fetch real-time market data.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict of symbol -> latest data
        """
        realtime_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Extract current data
                current = pd.Series({
                    'Open': info.get('open', 0),
                    'High': info.get('dayHigh', 0),
                    'Low': info.get('dayLow', 0),
                    'Close': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'Volume': info.get('volume', 0),
                    'Bid': info.get('bid', 0),
                    'Ask': info.get('ask', 0)
                })
                
                realtime_data[symbol] = current
                
                # Publish event for each symbol
                self._publish_data_event(symbol, current, "realtime")
                
            except Exception as e:
                self._publish_error_event(symbol, str(e))
                # Continue with other symbols
                
        return realtime_data
    
    def subscribe_to_feed(self, symbols: List[str], callback: Callable) -> None:
        """
        Subscribe to real-time data feed.
        
        Note: This is a simplified version that polls periodically.
        In production, you'd use websockets or a proper streaming API.
        """
        # For now, just register the callback
        # In a real implementation, this would start a background thread
        # that polls data and calls the callback
        for symbol in symbols:
            self.subscribe(callback)
    
    def _standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and format."""
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # yfinance returns with these columns, just ensure they exist
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Remove any extra columns we don't need
        data = data[required_cols].copy()
        
        # Ensure no NaN values
        data = data.dropna()
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality."""
        if data.empty:
            return False
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            return False
        
        # Check for valid price relationships
        if not (data['High'] >= data['Low']).all():
            return False
        
        if not (data['High'] >= data['Close']).all():
            return False
            
        if not (data['Close'] >= data['Low']).all():
            return False
        
        # Check for positive prices
        if not (data[['Open', 'High', 'Low', 'Close']] > 0).all().all():
            return False
        
        # Check for positive volume
        if not (data['Volume'] >= 0).all():
            return False
        
        return True
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid."""
        if cache_key not in self._cache:
            return False
        
        expiry = self._cache_expiry.get(cache_key)
        if expiry and datetime.now() > expiry:
            # Cache expired
            del self._cache[cache_key]
            del self._cache_expiry[cache_key]
            return False
        
        return True
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Cache data with expiry."""
        self._cache[cache_key] = data
        self._cache_expiry[cache_key] = datetime.now() + self._cache_ttl
    
    def _publish_data_event(self, symbol: str, data: Any, source: str) -> None:
        """Publish a data received event."""
        event = DataEvent(
            symbol=symbol,
            data=data,
            source=f"{self.name}_{source}"
        )
        self._event_bus.publish(event)
        self.notify(event)  # Also notify direct subscribers
    
    def _publish_error_event(self, symbol: str, error: str) -> None:
        """Publish a data error event."""
        event = Event(
            event_type=EventType.DATA_ERROR,
            source=self.name,
            data={'symbol': symbol, 'error': error}
        )
        self._event_bus.publish(event)