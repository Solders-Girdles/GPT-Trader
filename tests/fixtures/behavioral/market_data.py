"""
Deterministic market data generators for behavioral testing.

Provides predictable, repeatable market scenarios without mocks.
"""

from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class MarketDataGenerator:
    """Generate deterministic market data for testing."""
    
    @staticmethod
    def create_price_scenario(
        start_price: Decimal,
        movements: List[Tuple[str, Decimal]]
    ) -> List[Decimal]:
        """
        Create a price series from movements.
        
        Args:
            start_price: Initial price
            movements: List of (direction, magnitude) tuples
                      e.g., [('up', 100), ('down', 50)]
        
        Returns:
            List of prices following the movements
        """
        prices = [start_price]
        current = start_price
        
        for direction, magnitude in movements:
            if direction == 'up':
                current += magnitude
            elif direction == 'down':
                current -= magnitude
            elif direction == 'flat':
                pass  # No change
            else:
                raise ValueError(f"Unknown direction: {direction}")
            prices.append(current)
        
        return prices
    
    @staticmethod
    def create_funding_scenario(
        base_rate: Decimal = Decimal('0.0001'),
        variations: Optional[List[Decimal]] = None
    ) -> List[Decimal]:
        """
        Create funding rate series.
        
        Args:
            base_rate: Base funding rate (e.g., 0.01%)
            variations: Optional list of rate variations
        
        Returns:
            List of funding rates
        """
        if variations is None:
            # Default pattern: positive, negative, neutral
            variations = [
                Decimal('0'), 
                Decimal('-0.0002'),  # Shorts pay
                Decimal('0.0001'),   # Longs pay
                Decimal('0')
            ]
        
        rates = [base_rate]
        for var in variations:
            rates.append(base_rate + var)
        
        return rates
    
    @staticmethod
    def create_order_book(
        mid_price: Decimal,
        spread_bps: Decimal = Decimal('5'),
        depth_levels: int = 5,
        size_per_level: Decimal = Decimal('10')
    ) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        """
        Create deterministic order book.
        
        Args:
            mid_price: Mid market price
            spread_bps: Spread in basis points
            depth_levels: Number of price levels
            size_per_level: Size at each level
        
        Returns:
            Dict with 'bids' and 'asks' lists of (price, size) tuples
        """
        spread = mid_price * spread_bps / Decimal('10000')
        half_spread = spread / Decimal('2')
        
        # Create bid levels
        bids = []
        bid_price = mid_price - half_spread
        for i in range(depth_levels):
            price = bid_price - (i * spread)
            size = size_per_level * (Decimal('1') + Decimal(i) / Decimal('10'))
            bids.append((price, size))
        
        # Create ask levels
        asks = []
        ask_price = mid_price + half_spread
        for i in range(depth_levels):
            price = ask_price + (i * spread)
            size = size_per_level * (Decimal('1') + Decimal(i) / Decimal('10'))
            asks.append((price, size))
        
        return {'bids': bids, 'asks': asks}
    
    @staticmethod
    def calculate_impact(
        size: Decimal,
        order_book: Dict[str, List[Tuple[Decimal, Decimal]]],
        side: str
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate market impact for a given size.
        
        Args:
            size: Order size
            order_book: Order book from create_order_book
            side: 'buy' or 'sell'
        
        Returns:
            Tuple of (avg_fill_price, impact_bps)
        """
        levels = order_book['asks'] if side == 'buy' else order_book['bids']
        
        remaining = size
        total_cost = Decimal('0')
        total_filled = Decimal('0')
        
        for price, available in levels:
            fill_size = min(remaining, available)
            total_cost += price * fill_size
            total_filled += fill_size
            remaining -= fill_size
            
            if remaining == 0:
                break
        
        if total_filled == 0:
            return Decimal('0'), Decimal('0')
        
        avg_price = total_cost / total_filled
        
        # Calculate impact vs mid
        mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / Decimal('2')
        if side == 'buy':
            impact = (avg_price - mid_price) / mid_price
        else:
            impact = (mid_price - avg_price) / mid_price
        
        impact_bps = impact * Decimal('10000')
        
        return avg_price, impact_bps


class TradeScenarioBuilder:
    """Build complex trading scenarios for testing."""
    
    @staticmethod
    def create_round_trip(
        entry_price: Decimal,
        entry_size: Decimal,
        exit_price: Decimal,
        funding_payments: Optional[List[Decimal]] = None
    ) -> Dict[str, any]:
        """
        Create a complete round-trip trade scenario.
        
        Args:
            entry_price: Entry price
            entry_size: Position size
            exit_price: Exit price
            funding_payments: Optional funding payments during hold
        
        Returns:
            Dict with trade details and expected PnL
        """
        # Calculate raw PnL
        raw_pnl = (exit_price - entry_price) * entry_size
        
        # Add funding if provided
        total_funding = Decimal('0')
        if funding_payments:
            total_funding = sum(funding_payments)
        
        net_pnl = raw_pnl + total_funding
        
        return {
            'trades': [
                {'side': 'buy', 'price': entry_price, 'size': entry_size},
                {'side': 'sell', 'price': exit_price, 'size': entry_size}
            ],
            'funding': funding_payments or [],
            'expected': {
                'raw_pnl': raw_pnl,
                'funding': total_funding,
                'net_pnl': net_pnl
            }
        }
    
    @staticmethod
    def create_pyramid_entry(
        base_price: Decimal,
        base_size: Decimal,
        increments: List[Tuple[Decimal, Decimal]]
    ) -> Dict[str, any]:
        """
        Create pyramiding position scenario.
        
        Args:
            base_price: Initial entry price
            base_size: Initial position size
            increments: List of (price_delta, size) for adds
        
        Returns:
            Dict with trades and weighted average entry
        """
        trades = [{'side': 'buy', 'price': base_price, 'size': base_size}]
        
        total_cost = base_price * base_size
        total_size = base_size
        
        current_price = base_price
        for price_delta, size in increments:
            current_price += price_delta
            trades.append({'side': 'buy', 'price': current_price, 'size': size})
            total_cost += current_price * size
            total_size += size
        
        avg_entry = total_cost / total_size
        
        return {
            'trades': trades,
            'expected': {
                'avg_entry': avg_entry,
                'total_size': total_size
            }
        }
    
    @staticmethod
    def create_position_flip(
        initial_side: str,
        initial_price: Decimal,
        initial_size: Decimal,
        flip_price: Decimal,
        flip_size: Decimal
    ) -> Dict[str, any]:
        """
        Create position flip scenario (long to short or vice versa).
        
        Args:
            initial_side: 'long' or 'short'
            initial_price: Initial position price
            initial_size: Initial position size
            flip_price: Price at which to flip
            flip_size: Size of flip trade
        
        Returns:
            Dict with trades and expected outcomes
        """
        if initial_side == 'long':
            entry_trade = {'side': 'buy', 'price': initial_price, 'size': initial_size}
            flip_trade = {'side': 'sell', 'price': flip_price, 'size': flip_size}
            close_pnl = (flip_price - initial_price) * min(initial_size, flip_size)
        else:
            entry_trade = {'side': 'sell', 'price': initial_price, 'size': initial_size}
            flip_trade = {'side': 'buy', 'price': flip_price, 'size': flip_size}
            close_pnl = (initial_price - flip_price) * min(initial_size, flip_size)
        
        remaining = flip_size - initial_size
        new_side = None
        if remaining > 0:
            new_side = 'short' if initial_side == 'long' else 'long'
        
        return {
            'trades': [entry_trade, flip_trade],
            'expected': {
                'realized_pnl': close_pnl,
                'new_position': {
                    'side': new_side,
                    'size': abs(remaining) if remaining != 0 else Decimal('0'),
                    'entry': flip_price if remaining != 0 else Decimal('0')
                }
            }
        }


class ValidationConstants:
    """Standard validation constants for testing."""
    
    # Coinbase perpetuals specifications
    BTC_PERP = {
        'symbol': 'BTC-PERP',
        'base_increment': Decimal('0.0001'),  # 0.0001 BTC min
        'quote_increment': Decimal('0.01'),   # $0.01 price increment
        'min_notional': Decimal('10'),        # $10 min order
        'max_leverage': Decimal('10')         # 10x max leverage
    }
    
    ETH_PERP = {
        'symbol': 'ETH-PERP',
        'base_increment': Decimal('0.001'),   # 0.001 ETH min
        'quote_increment': Decimal('0.01'),   # $0.01 price increment
        'min_notional': Decimal('10'),        # $10 min order
        'max_leverage': Decimal('10')         # 10x max leverage
    }
    
    # Risk limits for testing
    CANARY_LIMITS = {
        'max_position_btc': Decimal('0.01'),
        'max_position_eth': Decimal('0.1'),
        'daily_loss_limit': Decimal('10'),    # $10
        'max_impact_bps': Decimal('50')       # 50bps max impact
    }
    
    PROD_LIMITS = {
        'max_position_btc': Decimal('1.0'),
        'max_position_eth': Decimal('10.0'),
        'daily_loss_limit': Decimal('1000'),  # $1000
        'max_impact_bps': Decimal('15')       # 15bps max impact
    }


class RealisticMarketData:
    """Current realistic market prices for behavioral testing."""
    
    # Current market prices as of late 2024
    CURRENT_PRICES = {
        'BTC-PERP': Decimal('95000'),   # ~$95k BTC
        'ETH-PERP': Decimal('3300'),    # ~$3.3k ETH  
        'SOL-PERP': Decimal('200'),     # ~$200 SOL
    }
    
    # Realistic trading ranges
    DAILY_VOLATILITY = {
        'BTC-PERP': Decimal('0.03'),    # 3% daily volatility
        'ETH-PERP': Decimal('0.04'),    # 4% daily volatility
        'SOL-PERP': Decimal('0.06'),    # 6% daily volatility
    }
    
    # Typical funding rates
    FUNDING_RATES = {
        'BTC-PERP': Decimal('0.0001'),  # 0.01% (0.03% APR)
        'ETH-PERP': Decimal('0.0001'),  # 0.01% (0.03% APR)
        'SOL-PERP': Decimal('0.0002'),  # 0.02% (0.06% APR)
    }
    
    @classmethod
    def get_realistic_price_range(cls, symbol: str, volatility_multiple: Decimal = Decimal('1')) -> tuple[Decimal, Decimal]:
        """
        Get realistic intraday price range for a symbol.
        
        Args:
            symbol: Trading symbol
            volatility_multiple: Multiplier for volatility (1 = normal, 2 = high vol day)
            
        Returns:
            (low_price, high_price) tuple
        """
        base_price = cls.CURRENT_PRICES[symbol]
        daily_vol = cls.DAILY_VOLATILITY[symbol] * volatility_multiple
        
        low_price = base_price * (Decimal('1') - daily_vol)
        high_price = base_price * (Decimal('1') + daily_vol)
        
        return low_price, high_price
    
    @classmethod
    def create_realistic_price_movement(
        cls, 
        symbol: str, 
        direction: str, 
        magnitude: str = "normal"
    ) -> tuple[Decimal, Decimal]:
        """
        Create realistic price movement for testing.
        
        Args:
            symbol: Trading symbol
            direction: "up", "down", or "sideways"
            magnitude: "small", "normal", "large"
            
        Returns:
            (start_price, end_price) tuple
        """
        base_price = cls.CURRENT_PRICES[symbol]
        daily_vol = cls.DAILY_VOLATILITY[symbol]
        
        # Magnitude multipliers
        magnitude_mult = {
            "small": Decimal('0.5'),
            "normal": Decimal('1.0'),
            "large": Decimal('2.0')
        }.get(magnitude, Decimal('1.0'))
        
        movement = daily_vol * magnitude_mult
        
        if direction == "up":
            end_price = base_price * (Decimal('1') + movement)
        elif direction == "down":
            end_price = base_price * (Decimal('1') - movement)
        else:  # sideways
            # Small random movement
            end_price = base_price * (Decimal('1') + (movement * Decimal('0.1')))
        
        return base_price, end_price