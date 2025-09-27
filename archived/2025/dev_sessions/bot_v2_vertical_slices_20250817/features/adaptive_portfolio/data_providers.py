"""
Clean data provider abstraction for adaptive portfolio.

Provides abstract interface and implementations for market data access,
eliminating ugly try/except import blocks throughout the codebase.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Safe imports that are always available
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
    DataFrame = pd.DataFrame
except ImportError:
    HAS_PANDAS = False
    from typing import Any
    DataFrame = Any


class DataProvider(ABC):
    """Abstract interface for market data providers."""
    
    @abstractmethod
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "60d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> DataFrame:
        """
        Get historical OHLCV data for symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Period string (e.g., '60d', '1y') or None if using start/end
            start: Start date (YYYY-MM-DD) - optional
            end: End date (YYYY-MM-DD) - optional
            
        Returns:
            DataFrame with OHLCV data indexed by date
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current/latest price for symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price as float
        """
        pass
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            True if market is open, False otherwise
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols for this provider.
        
        Returns:
            List of available symbol strings
        """
        pass


class MockDataProvider(DataProvider):
    """
    Mock data provider that generates realistic synthetic data.
    
    Always available and useful for testing and development.
    """
    
    def __init__(self):
        """Initialize mock provider."""
        self.logger = logging.getLogger(__name__)
        self._base_prices = {
            'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 120.0, 'AMZN': 140.0,
            'TSLA': 200.0, 'META': 250.0, 'NVDA': 400.0, 'NFLX': 400.0,
            'SPY': 450.0, 'QQQ': 350.0, 'IWM': 180.0, 'VTI': 220.0,
            'BRK-B': 350.0, 'JNJ': 160.0, 'V': 230.0, 'JPM': 150.0,
            'UNH': 500.0, 'HD': 300.0, 'PG': 140.0, 'DIS': 100.0,
            'MA': 350.0, 'BAC': 35.0, 'ADBE': 480.0, 'CRM': 200.0, 'PYPL': 60.0
        }
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "60d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> DataFrame:
        """Generate realistic synthetic historical data."""
        if not HAS_PANDAS:
            # Return simple dict-based data structure when pandas not available
            return self._generate_simple_data(symbol, period, start, end)
        
        # Parse period to get number of days
        if period.endswith('d'):
            days = int(period[:-1])
        elif period.endswith('y'):
            days = int(period[:-1]) * 365
        elif period.endswith('mo'):
            days = int(period[:-2]) * 30
        else:
            days = 60  # Default
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Filter to weekdays only (no weekends)
        dates = dates[dates.weekday < 5]
        
        # Get base price for symbol
        base_price = self._base_prices.get(symbol, 100.0)
        
        # Generate realistic price series
        import random
        random.seed(hash(symbol) % 2**32)  # Deterministic but unique per symbol
        
        # Generate returns with some trending and volatility
        daily_returns = [random.normalvariate(0.0008, 0.02) for _ in range(len(dates))]
        
        # Add some trending behavior
        for i, ret in enumerate(daily_returns):
            trend_factor = (i / len(dates) - 0.5) * 0.0002  # Small trend component
            daily_returns[i] += trend_factor
        
        # Calculate prices
        prices = [base_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            # Generate intraday range
            volatility = abs(random.normalvariate(0, 0.01))  # Daily volatility
            high = close_price * (1 + volatility)
            low = close_price * (1 - volatility)
            
            # Open is close of previous day (with small gap)
            if i == 0:
                open_price = close_price
            else:
                gap = random.normalvariate(0, 0.005)  # Small gap
                open_price = prices[i-1] * (1 + gap)
            
            # Ensure OHLC relationships make sense
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume (higher volume on bigger moves)
            price_change = abs(close_price - open_price) / open_price
            base_volume = 1000000
            volume = int(base_volume * (1 + price_change * 10) * random.uniform(0.5, 2.0))
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Date'
        
        self.logger.debug(f"Generated {len(df)} days of mock data for {symbol}")
        return df
    
    def _generate_simple_data(self, symbol: str, period: str, start: Optional[str], end: Optional[str]):
        """Generate simple data structure when pandas is not available."""
        # Parse period to get number of days
        if period.endswith('d'):
            days = int(period[:-1])
        elif period.endswith('y'):
            days = int(period[:-1]) * 365
        elif period.endswith('mo'):
            days = int(period[:-2]) * 30
        else:
            days = 60  # Default
        
        # Generate date range (simplified without pandas)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            # Only include weekdays
            if current_date.weekday() < 5:
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Get base price for symbol
        base_price = self._base_prices.get(symbol, 100.0)
        
        # Generate realistic price series
        import random
        random.seed(hash(symbol) % 2**32)  # Deterministic but unique per symbol
        
        # Generate returns with some trending and volatility
        daily_returns = [random.normalvariate(0.0008, 0.02) for _ in range(len(dates))]
        
        # Calculate prices
        prices = [base_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data as simple dict structure
        simple_data = {
            'Open': [],
            'High': [],
            'Low': [],
            'Close': [],
            'Volume': [],
            'Date': dates
        }
        
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            # Generate intraday range
            volatility = abs(random.normalvariate(0, 0.01))  # Daily volatility
            high = close_price * (1 + volatility)
            low = close_price * (1 - volatility)
            
            # Open is close of previous day (with small gap)
            if i == 0:
                open_price = close_price
            else:
                gap = random.normalvariate(0, 0.005)  # Small gap
                open_price = prices[i-1] * (1 + gap)
            
            # Ensure OHLC relationships make sense
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume (higher volume on bigger moves)
            price_change = abs(close_price - open_price) / open_price
            base_volume = 1000000
            volume = int(base_volume * (1 + price_change * 10) * random.uniform(0.5, 2.0))
            
            simple_data['Open'].append(round(open_price, 2))
            simple_data['High'].append(round(high, 2))
            simple_data['Low'].append(round(low, 2))
            simple_data['Close'].append(round(close_price, 2))
            simple_data['Volume'].append(volume)
        
        # Create a simple class that mimics basic DataFrame functionality
        class SimpleDataFrame:
            def __init__(self, data):
                self.data = data
                self.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                self.index = data['Date']
            
            def __len__(self):
                return len(self.data['Close'])
            
            def __getitem__(self, key):
                if key in self.data:
                    return self.data[key]
                raise KeyError(f"Column '{key}' not found")
            
            @property
            def iloc(self):
                # Simple iloc implementation
                class IlocAccessor:
                    def __init__(self, df):
                        self.df = df
                    
                    def __getitem__(self, index):
                        if isinstance(index, int):
                            if index < 0:
                                index = len(self.df) + index  # Handle negative indexing
                            if index >= len(self.df) or index < 0:
                                raise IndexError("Index out of range")
                            
                            result = {}
                            for col in self.df.columns:
                                result[col] = self.df.data[col][index]
                            return result
                        return None
                
                return IlocAccessor(self)
            
            def equals(self, other):
                if not isinstance(other, SimpleDataFrame):
                    return False
                for col in self.columns:
                    if self.data[col] != other.data[col]:
                        return False
                return True
        
        result = SimpleDataFrame(simple_data)
        self.logger.debug(f"Generated {len(result)} days of simple mock data for {symbol}")
        return result
    
    def get_current_price(self, symbol: str) -> float:
        """Return current price (last price from recent data)."""
        try:
            recent_data = self.get_historical_data(symbol, period="1d")
            if len(recent_data) > 0:
                return float(recent_data['Close'].iloc[-1])
            else:
                return self._base_prices.get(symbol, 100.0)
        except Exception:
            return self._base_prices.get(symbol, 100.0)
    
    def is_market_open(self) -> bool:
        """Mock market hours (9:30 AM - 4:00 PM ET on weekdays)."""
        now = datetime.now()
        
        # Simple check: weekday and reasonable hours (ignoring timezone for mock)
        if now.weekday() >= 5:  # Weekend
            return False
        
        hour = now.hour
        if 9 <= hour < 16:  # Rough market hours
            return True
        
        return False
    
    def get_available_symbols(self) -> List[str]:
        """Return list of symbols we can provide mock data for."""
        return list(self._base_prices.keys())


class YfinanceDataProvider(DataProvider):
    """
    Real data provider using yfinance.
    
    Only available if yfinance is installed.
    """
    
    def __init__(self):
        """Initialize yfinance provider."""
        try:
            import yfinance as yf
            self.yf = yf
            self.logger = logging.getLogger(__name__)
            self.logger.info("YfinanceDataProvider initialized successfully")
        except ImportError as e:
            raise ImportError(
                "YfinanceDataProvider requires yfinance. "
                "Install with: pip install yfinance"
            ) from e
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "60d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> DataFrame:
        """Get historical data from yfinance."""
        try:
            ticker = self.yf.Ticker(symbol)
            
            if start and end:
                data = ticker.history(start=start, end=end)
            else:
                data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data available for symbol {symbol}")
            
            self.logger.debug(f"Retrieved {len(data)} days of data for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from yfinance."""
        try:
            ticker = self.yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                raise ValueError(f"No current price data for {symbol}")
            
            current_price = float(hist['Close'].iloc[-1])
            self.logger.debug(f"Current price for {symbol}: ${current_price:.2f}")
            return current_price
            
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {e}")
            raise
    
    def is_market_open(self) -> bool:
        """
        Check if market is open by trying to get real-time data.
        
        This is a simplified implementation. A production version would
        use proper market calendar APIs.
        """
        try:
            # Simple heuristic: try to get very recent data for SPY
            ticker = self.yf.Ticker("SPY")
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                return False
            
            # Check if we have data from the last few minutes
            latest_time = hist.index[-1]
            now = datetime.now()
            time_diff = now - latest_time.to_pydatetime().replace(tzinfo=None)
            
            # If latest data is within 10 minutes, assume market is open
            return time_diff.total_seconds() < 600
            
        except Exception:
            # Fallback to weekday business hours check
            now = datetime.now()
            if now.weekday() >= 5:  # Weekend
                return False
            
            hour = now.hour
            if 9 <= hour < 16:  # Rough market hours (ignoring timezone)
                return True
            
            return False
    
    def get_available_symbols(self) -> List[str]:
        """
        Return common symbols (yfinance supports many symbols).
        
        In practice, this would query a symbol database or API.
        """
        return [
            # Major tech stocks
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            # ETFs
            "SPY", "QQQ", "IWM", "VTI",
            # Blue chips
            "BRK-B", "JNJ", "V", "JPM", "UNH", "HD", "PG", "DIS",
            # Others
            "MA", "BAC", "ADBE", "CRM", "PYPL"
        ]


def create_data_provider(prefer_real_data: bool = True) -> Tuple[DataProvider, str]:
    """
    Factory function to create appropriate data provider.
    
    Args:
        prefer_real_data: If True, try to create YfinanceDataProvider first
        
    Returns:
        Tuple of (provider_instance, provider_type)
        provider_type is 'yfinance' or 'mock'
    """
    logger = logging.getLogger(__name__)
    
    if prefer_real_data:
        try:
            provider = YfinanceDataProvider()
            logger.info("Created YfinanceDataProvider for real market data")
            return provider, 'yfinance'
        except ImportError:
            logger.warning("yfinance not available, falling back to MockDataProvider")
    
    # Fallback to mock provider
    provider = MockDataProvider()
    logger.info("Created MockDataProvider for synthetic data")
    return provider, 'mock'


def get_data_provider_info() -> Dict[str, bool]:
    """
    Get information about available data providers.
    
    Returns:
        Dict with provider availability information
    """
    info = {
        'mock_available': True,  # Always available
        'yfinance_available': False,
        'pandas_available': HAS_PANDAS
    }
    
    try:
        import yfinance
        info['yfinance_available'] = True
    except ImportError:
        pass
    
    return info