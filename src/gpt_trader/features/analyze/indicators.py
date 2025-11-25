from typing import Any
import pandas as pd

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    return pd.Series()

def calculate_macd(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()

def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    return pd.DataFrame({"adx": [0] * len(data)})

