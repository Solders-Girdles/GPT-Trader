from __future__ import annotations

import pandas as pd
from bot.validation import get_data_validator, get_math_validator


def atr(
    df: pd.DataFrame, period: int = 14, method: str = "wilder", validate_data: bool = True
) -> pd.Series:
    """
    Average True Range with validation.

    Parameters
    - df: DataFrame with columns High, Low, Close
    - period: lookback window
    - method: "wilder" (EWMA with alpha=1/period) or "sma" (simple rolling mean)
    - validate_data: Whether to validate input data
    """
    # Validate input data if requested
    if validate_data:
        data_validator = get_data_validator()
        df_validated, validation_result = data_validator.validate_ohlcv(
            df, symbol="ATR", repair=True
        )

        if not validation_result.is_valid and validation_result.issues:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"ATR input validation issues: {validation_result.issues}")

        # Use validated data
        high = df_validated["High"].astype(float)
        low = df_validated["Low"].astype(float)
        close = df_validated["Close"].astype(float)
    else:
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)

    prev_close = close.shift(1)

    # Calculate true range components
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Validate period
    if period <= 0:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Invalid ATR period {period}, using default 14")
        period = 14

    if method.lower() == "wilder":
        # Wilder smoothing ≈ EWMA with alpha = 1/period
        # Use safe division for alpha calculation
        math_validator = get_math_validator()
        alpha = math_validator.safe_divide(1.0, float(period), default=0.07, name="atr_alpha")
        return tr.ewm(alpha=alpha, adjust=False).mean()
    else:
        # Fallback to simple moving average
        return tr.rolling(int(period), min_periods=int(period)).mean()
