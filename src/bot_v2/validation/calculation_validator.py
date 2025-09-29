"""Manual validation helpers for trading calculations."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


def manual_backtest_example() -> (
    dict[str, Any]
):  # noqa: D401 - utility function retained for compatibility
    """Run a manual MA crossover walkthrough for validation exercises."""

    logger.info("=" * 60)
    logger.info("MANUAL CALCULATION VALIDATION")
    logger.info("=" * 60)
    logger.info("\nScenario: Simple MA crossover with known data")
    logger.info("-" * 40)

    prices = [100, 98, 96, 97, 99, 102, 104, 103, 101, 100]

    logger.info("\nDay | Price | MA3  | MA5  | Signal | Action")
    logger.info("----|-------|------|------|--------|--------")

    ma3: list[float | None] = []
    ma5: list[float | None] = []
    signals: list[str] = []

    for i in range(len(prices)):
        if i >= 2:
            ma3_val = sum(prices[i - 2 : i + 1]) / 3
            ma3.append(ma3_val)
        else:
            ma3.append(None)

        if i >= 4:
            ma5_val = sum(prices[i - 4 : i + 1]) / 5
            ma5.append(ma5_val)
        else:
            ma5.append(None)

        if (
            i > 0
            and ma3[i] is not None
            and ma5[i] is not None
            and ma3[i - 1] is not None
            and ma5[i - 1] is not None
        ):
            if ma3[i] > ma5[i] and ma3[i - 1] <= ma5[i - 1]:
                signals.append("BUY")
            elif ma3[i] < ma5[i] and ma3[i - 1] >= ma5[i - 1]:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        else:
            signals.append("-")

        ma3_str = f"{ma3[i]:.1f}" if ma3[i] else "-"
        ma5_str = f"{ma5[i]:.1f}" if ma5[i] else "-"

        message = f"{i + 1:3} | {prices[i]:5} | {ma3_str:5} | {ma5_str:5} | {signals[i]:6} | "

        if signals[i] == "BUY":
            message += "Buy signal"
        elif signals[i] == "SELL":
            message += "Sell signal"
        else:
            message += "-"

        logger.info(message)

    logger.info("\n" + "=" * 60)
    logger.info("TRADING SIMULATION")
    logger.info("=" * 60)

    capital = 10000
    position = 0
    cash = capital
    trades: list[dict[str, object]] = []

    logger.info("\nStarting capital: $%s", f"{capital:,.2f}")
    logger.info("\nTrade Log:")
    logger.info("-" * 40)

    for i, signal in enumerate(signals):
        if signal == "BUY" and position == 0:
            shares = int(cash / prices[i])
            cost = shares * prices[i]
            cash -= cost
            position = shares
            trades.append(
                {
                    "day": i + 1,
                    "action": "BUY",
                    "price": prices[i],
                    "shares": shares,
                    "cost": cost,
                    "cash_after": cash,
                }
            )
            logger.info(
                "Day %s: BUY %s shares @ $%s = $%s",
                i + 1,
                shares,
                prices[i],
                f"{cost:.2f}",
            )
            logger.info("         Cash remaining: $%s", f"{cash:.2f}")

        elif signal == "SELL" and position > 0:
            proceeds = position * prices[i]
            cash += proceeds
            pnl = proceeds - trades[-1]["cost"]
            trades.append(
                {
                    "day": i + 1,
                    "action": "SELL",
                    "price": prices[i],
                    "shares": position,
                    "proceeds": proceeds,
                    "cash_after": cash,
                    "pnl": pnl,
                }
            )
            logger.info(
                "Day %s: SELL %s shares @ $%s = $%s",
                i + 1,
                position,
                prices[i],
                f"{proceeds:.2f}",
            )
            logger.info("         P&L: $%s", f"{pnl:.2f}")
            logger.info("         Cash after: $%s", f"{cash:.2f}")
            position = 0

    final_value = cash + (position * prices[-1])
    total_return = final_value - capital
    return_pct = (total_return / capital) * 100

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info("Final cash: $%s", f"{cash:.2f}")
    logger.info(
        "Position value: $%s",
        f"{position * prices[-1] if position > 0 else 0:.2f}",
    )
    logger.info("Total value: $%s", f"{final_value:.2f}")
    logger.info(
        "Total return: %s (%s%%)",
        f"{total_return:.2f}",
        f"{return_pct:.2f}",
    )

    buy_hold_shares = int(capital / prices[0])
    buy_hold_value = buy_hold_shares * prices[-1]
    buy_hold_return = ((buy_hold_value - capital) / capital) * 100

    logger.info(
        "\nBuy & Hold: $%s (%s%%)",
        f"{buy_hold_value:.2f}",
        f"{buy_hold_return:.2f}",
    )
    logger.info("Outperformance: %s%%", f"{return_pct - buy_hold_return:.2f}")

    return {
        "prices": prices,
        "ma3": ma3,
        "ma5": ma5,
        "signals": signals,
        "trades": trades,
        "final_value": final_value,
        "return_pct": return_pct,
    }


def verify_with_system() -> None:  # noqa: D401 - optional manual workflow
    """Replay the manual scenario through the trading system for comparison."""

    logger.info("\n" + "=" * 60)
    logger.info("SYSTEM VERIFICATION")
    logger.info("=" * 60)

    from executor import SimpleExecutor
    from ledger import TradeLedger

    prices = [100, 98, 96, 97, 99, 102, 104, 103, 101, 100]

    data = pd.DataFrame({"Close": prices}, index=pd.date_range("2024-01-01", periods=10))

    strategy = SimpleMAStrategy(fast_period=3, slow_period=5)
    signals = strategy.generate_signals(data)

    executor = SimpleExecutor(10000)
    ledger = TradeLedger()

    for i, (date, row) in enumerate(data.iterrows()):
        signal = signals.iloc[i]
        price = row["Close"]

        action = executor.process_signal("TEST", signal, price, date)

        if action["type"] in ["buy", "sell"]:
            ledger.record_transaction(
                date,
                "TEST",
                action["type"],
                action["quantity"],
                price,
            )

    final_value = executor.get_portfolio_value({"TEST": prices[-1]})

    logger.info("\nSystem final value: $%s", f"{final_value:.2f}")
    logger.info("System transactions: %s", len(ledger.transactions))


def validate() -> None:
    """Run both manual and system validations."""

    manual_results = manual_backtest_example()

    logger.info("\nManual validation complete.")
    logger.info("Final value: $%s", f"{manual_results['final_value']:.2f}")
    logger.info("Return: %s%%", f"{manual_results['return_pct']:.2f}")

    verify_with_system()


if __name__ == "__main__":
    validate()
