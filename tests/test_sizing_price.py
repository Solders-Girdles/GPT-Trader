# tests/test_sizing_price.py
import re
from pathlib import Path


def test_position_sizing_uses_open() -> None:
    s = Path("src/bot/backtest/engine_portfolio.py").read_text()
    # Any assignment like `= int(... Close ... )` should fail
    bad = re.findall(r"=\s*int\([^)]*Close[^)]*\)", s)
    assert not bad, f"Found sizing using Close: {bad}"
