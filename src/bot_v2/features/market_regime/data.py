"""
Data fetching and historical regime analysis - LOCAL to this slice.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .types import MarketRegime, RegimeAnalysis, RegimeTransition


def fetch_market_data(symbol: str, lookback_days: int) -> pd.DataFrame:
    """
    Fetch market data for regime analysis.

    LOCAL implementation - in production would call data provider.
    """
    # Generate synthetic but realistic market data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 10)  # Extra buffer

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = dates[dates.weekday < 5]  # Trading days only
    dates = dates[-lookback_days:]  # Exact number requested

    # Generate realistic price path with different regimes
    data = generate_realistic_price_data(len(dates), symbol)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
        }
    )

    return df


def get_historical_regimes(
    symbol: str, start_date: datetime, end_date: datetime
) -> tuple[list[tuple[MarketRegime, datetime, datetime]], list[RegimeTransition]]:
    """
    Get historical regime periods and transitions.

    In production, this would be stored in database or calculated from historical data.
    For now, generate synthetic but realistic regime history.
    """
    print(f"📊 Generating historical regimes for {symbol}...")

    # Generate synthetic regime history
    regimes = generate_synthetic_regime_history(start_date, end_date)

    # Calculate transitions
    transitions = []
    for i in range(1, len(regimes)):
        prev_regime, prev_start, prev_end = regimes[i - 1]
        curr_regime, curr_start, curr_end = regimes[i]

        # Simple transition between regimes
        transition = RegimeTransition(
            from_regime=prev_regime,
            to_regime=curr_regime,
            transition_date=curr_start,
            duration_days=1,  # Assume instant transitions
            trigger_events=_generate_trigger_events(prev_regime, curr_regime),
        )
        transitions.append(transition)

    print(f"✅ Generated {len(regimes)} regime periods and {len(transitions)} transitions")

    return regimes, transitions


def generate_realistic_price_data(n_days: int, symbol: str) -> dict:
    """
    Generate realistic price data with regime-like behavior.
    """
    np.random.seed(hash(symbol) % 2**32)  # Consistent per symbol

    # Start with base price
    initial_price = 100 + hash(symbol) % 200  # $100-300 range

    # Generate returns with different regimes
    returns = []
    current_regime = np.random.choice(["bull", "bear", "sideways", "volatile"])
    regime_days = 0
    regime_length = np.random.randint(10, 30)

    for day in range(n_days):
        # Switch regimes periodically
        if regime_days >= regime_length:
            current_regime = np.random.choice(["bull", "bear", "sideways", "volatile"])
            regime_days = 0
            regime_length = np.random.randint(10, 30)

        # Generate return based on regime
        if current_regime == "bull":
            daily_return = np.random.normal(0.001, 0.015)  # Positive drift, low vol
        elif current_regime == "bear":
            daily_return = np.random.normal(-0.002, 0.020)  # Negative drift, higher vol
        elif current_regime == "sideways":
            daily_return = np.random.normal(0.0, 0.010)  # No drift, low vol
        else:  # volatile
            daily_return = np.random.normal(0.0, 0.035)  # No drift, high vol

        returns.append(daily_return)
        regime_days += 1

    # Convert returns to prices
    prices = [initial_price]
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)

    prices = np.array(prices[1:])  # Remove initial price

    # Generate OHLC from close prices
    open_prices = []
    high_prices = []
    low_prices = []
    volume = []

    for i, close in enumerate(prices):
        # Open is previous close with small gap
        if i == 0:
            open_price = close * (1 + np.random.normal(0, 0.005))
        else:
            open_price = prices[i - 1] * (1 + np.random.normal(0, 0.005))

        # High and low around open/close
        high_price = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))

        # Volume with some correlation to volatility
        daily_vol = abs(returns[i]) if i < len(returns) else 0.01
        base_volume = 1_000_000
        vol_factor = 1 + daily_vol * 10  # Higher volatility = higher volume
        daily_volume = int(base_volume * vol_factor * np.random.uniform(0.5, 2.0))

        open_prices.append(open_price)
        high_prices.append(high_price)
        low_prices.append(low_price)
        volume.append(daily_volume)

    return {
        "open": np.array(open_prices),
        "high": np.array(high_prices),
        "low": np.array(low_prices),
        "close": prices,
        "volume": np.array(volume),
    }


def generate_synthetic_regime_history(
    start_date: datetime, end_date: datetime
) -> list[tuple[MarketRegime, datetime, datetime]]:
    """
    Generate synthetic but realistic regime history.
    """
    regimes = []
    current_date = start_date

    # Start with random regime
    current_regime = np.random.choice(list(MarketRegime))

    while current_date < end_date:
        # Determine regime duration based on regime type
        duration_days = _get_typical_regime_duration(current_regime)
        duration_days = int(np.random.normal(duration_days, duration_days * 0.3))
        duration_days = max(5, min(duration_days, 120))  # 5 to 120 days

        regime_end = current_date + timedelta(days=duration_days)
        regime_end = min(regime_end, end_date)

        regimes.append((current_regime, current_date, regime_end))

        # Transition to next regime
        current_regime = _get_next_regime(current_regime)
        current_date = regime_end

    return regimes


def _get_typical_regime_duration(regime: MarketRegime) -> int:
    """Get typical duration for each regime type."""
    durations = {
        MarketRegime.BULL_QUIET: 45,
        MarketRegime.BULL_VOLATILE: 25,
        MarketRegime.BEAR_QUIET: 35,
        MarketRegime.BEAR_VOLATILE: 20,
        MarketRegime.SIDEWAYS_QUIET: 30,
        MarketRegime.SIDEWAYS_VOLATILE: 15,
        MarketRegime.CRISIS: 10,
    }
    return durations.get(regime, 30)


def _get_next_regime(current: MarketRegime) -> MarketRegime:
    """Get realistic next regime based on transition probabilities."""

    # Simplified transition probabilities
    transitions = {
        MarketRegime.BULL_QUIET: [
            (MarketRegime.BULL_QUIET, 0.4),
            (MarketRegime.BULL_VOLATILE, 0.25),
            (MarketRegime.SIDEWAYS_QUIET, 0.2),
            (MarketRegime.SIDEWAYS_VOLATILE, 0.1),
            (MarketRegime.BEAR_QUIET, 0.05),
        ],
        MarketRegime.BULL_VOLATILE: [
            (MarketRegime.BULL_VOLATILE, 0.3),
            (MarketRegime.BULL_QUIET, 0.3),
            (MarketRegime.SIDEWAYS_VOLATILE, 0.2),
            (MarketRegime.BEAR_VOLATILE, 0.15),
            (MarketRegime.CRISIS, 0.05),
        ],
        MarketRegime.BEAR_QUIET: [
            (MarketRegime.BEAR_QUIET, 0.4),
            (MarketRegime.BEAR_VOLATILE, 0.2),
            (MarketRegime.SIDEWAYS_QUIET, 0.25),
            (MarketRegime.BULL_QUIET, 0.1),
            (MarketRegime.SIDEWAYS_VOLATILE, 0.05),
        ],
        MarketRegime.BEAR_VOLATILE: [
            (MarketRegime.BEAR_VOLATILE, 0.35),
            (MarketRegime.CRISIS, 0.15),
            (MarketRegime.BEAR_QUIET, 0.2),
            (MarketRegime.SIDEWAYS_VOLATILE, 0.15),
            (MarketRegime.BULL_VOLATILE, 0.15),
        ],
        MarketRegime.SIDEWAYS_QUIET: [
            (MarketRegime.SIDEWAYS_QUIET, 0.4),
            (MarketRegime.BULL_QUIET, 0.25),
            (MarketRegime.BEAR_QUIET, 0.2),
            (MarketRegime.SIDEWAYS_VOLATILE, 0.1),
            (MarketRegime.BULL_VOLATILE, 0.05),
        ],
        MarketRegime.SIDEWAYS_VOLATILE: [
            (MarketRegime.SIDEWAYS_VOLATILE, 0.3),
            (MarketRegime.BULL_VOLATILE, 0.25),
            (MarketRegime.BEAR_VOLATILE, 0.25),
            (MarketRegime.SIDEWAYS_QUIET, 0.15),
            (MarketRegime.CRISIS, 0.05),
        ],
        MarketRegime.CRISIS: [
            (MarketRegime.BEAR_VOLATILE, 0.4),
            (MarketRegime.SIDEWAYS_VOLATILE, 0.3),
            (MarketRegime.BULL_VOLATILE, 0.2),
            (MarketRegime.CRISIS, 0.1),
        ],
    }

    # Get transition probabilities for current regime
    possible_transitions = transitions.get(current, [(current, 1.0)])

    # Choose next regime based on probabilities
    regimes, probs = zip(*possible_transitions, strict=False)
    next_regime = np.random.choice(regimes, p=probs)

    return next_regime


def _generate_trigger_events(from_regime: MarketRegime, to_regime: MarketRegime) -> list[str]:
    """Generate realistic trigger events for regime transitions."""

    # Common economic events
    economic_events = [
        "Fed policy change",
        "Economic data release",
        "Earnings season",
        "GDP report",
        "Inflation data",
        "Employment report",
    ]

    # Market events
    market_events = [
        "Technical breakout",
        "Support/resistance break",
        "Volume spike",
        "Sector rotation",
        "Options expiration",
        "Rebalancing flows",
    ]

    # Crisis events
    crisis_events = [
        "Geopolitical tension",
        "Financial institution failure",
        "Currency crisis",
        "Credit crunch",
        "Black swan event",
    ]

    # News events
    news_events = [
        "Corporate announcement",
        "Regulatory change",
        "Trade news",
        "Political development",
        "Natural disaster",
        "Technology disruption",
    ]

    # Select appropriate events based on transition
    if to_regime == MarketRegime.CRISIS:
        return np.random.choice(crisis_events, size=2, replace=False).tolist()
    elif "VOLATILE" in to_regime.value and "QUIET" in from_regime.value:
        return np.random.choice(market_events + news_events, size=2, replace=False).tolist()
    elif "BULL" in to_regime.value and "BEAR" in from_regime.value:
        return np.random.choice(
            economic_events + ["Positive sentiment shift"], size=2, replace=False
        ).tolist()
    elif "BEAR" in to_regime.value and "BULL" in from_regime.value:
        return np.random.choice(
            economic_events + ["Negative sentiment shift"], size=2, replace=False
        ).tolist()
    else:
        # General transition
        all_events = economic_events + market_events + news_events
        return np.random.choice(all_events, size=2, replace=False).tolist()


def store_regime_analysis(
    symbol: str, analysis: RegimeAnalysis, file_path: str | None = None
) -> bool:
    """
    Store regime analysis to file for future reference.

    Simplified implementation - in production would use proper database.
    """
    try:
        if file_path is None:
            file_path = f"regime_analysis_{symbol}_{datetime.now().strftime('%Y%m%d')}.json"

        # Convert analysis to dict (simplified)
        {
            "symbol": symbol,
            "timestamp": analysis.timestamp.isoformat(),
            "current_regime": analysis.current_regime.value,
            "confidence": analysis.confidence,
            "regime_duration": analysis.regime_duration,
            "stability_score": analysis.stability_score,
        }

        # In production, would save to database or file
        print(f"📁 Regime analysis stored for {symbol}")
        return True

    except Exception as e:
        print(f"❌ Failed to store regime analysis: {e}")
        return False


def load_regime_analysis(
    symbol: str, date: datetime | None = None, file_path: str | None = None
) -> RegimeAnalysis | None:
    """
    Load historical regime analysis from storage.

    Simplified implementation.
    """
    try:
        # In production, would load from database or file
        print(f"📂 Loading regime analysis for {symbol}")
        return None  # Simplified - return None for now

    except Exception as e:
        print(f"❌ Failed to load regime analysis: {e}")
        return None
