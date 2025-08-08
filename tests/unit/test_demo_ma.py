import pandas as pd
from bot.strategy.demo_ma import DemoMAStrategy


def test_demo_ma_signals_basic() -> None:
    data = pd.DataFrame(
        {"Close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
    )
    strat = DemoMAStrategy(fast=3, slow=5)
    out = strat.generate_signals(data)
    assert "signal" in out.columns
    assert out["signal"].iloc[-1] in (-1, 0, 1)
