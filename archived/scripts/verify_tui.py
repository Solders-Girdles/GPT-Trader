import asyncio
import logging
import sys
from decimal import Decimal

# Add src to path
sys.path.append("src")

from gpt_trader.cli.services import instantiate_bot
from gpt_trader.app.config import BotConfig
from gpt_trader.tui.state import TuiState

# Configure logging to avoid noise
logging.basicConfig(level=logging.WARNING)


async def main():
    print("Starting TUI Integration Verification...")

    # 1. Configure Bot (Mock Mode)
    config = BotConfig(
        symbols=["BTC-USD"],
        mock_broker=True,
        dry_run=True,
        interval=1,
    )

    # 2. Instantiate Bot
    print("Instantiating TradingBot...")
    bot = instantiate_bot(config)

    # 3. Initialize TuiState
    print("Initializing TuiState...")
    tui_state = TuiState()

    # 4. Simulate Bot Activity
    print("Simulating Bot Cycle...")
    # We need to start the bot to initialize components, but we'll stop it quickly
    # Or we can just manually trigger what we need if we know the internals.
    # Let's try running it for a brief moment to let it populate initial state.

    # Start bot in background
    bot_task = asyncio.create_task(bot.run())

    # Wait for a cycle (mock broker should be fast)
    await asyncio.sleep(2)

    # 5. Extract Status (Mimic app.py logic)
    print("Fetching Status...")
    status = {}
    if hasattr(bot.engine, "status_reporter"):
        status = bot.engine.status_reporter.get_status()

    # Inject Risk Data (Mimic app.py logic)
    if hasattr(bot, "risk_manager") and bot.risk_manager:
        rm = bot.risk_manager
        risk_status = {
            "max_leverage": getattr(rm.config, "max_leverage", 0.0) if rm.config else 0.0,
            "daily_loss_limit_pct": (
                getattr(rm.config, "daily_loss_limit_pct", 0.0) if rm.config else 0.0
            ),
            "reduce_only_mode": getattr(rm, "_reduce_only_mode", False),
            "reduce_only_reason": getattr(rm, "_reduce_only_reason", ""),
            "current_daily_loss_pct": 0.0,
        }
        status["risk"] = risk_status

    # 6. Update State
    print("Updating TuiState...")
    tui_state.update_from_bot_status(status)

    # 7. Verify Data
    print("\n--- Verification Results ---")

    # Strategy Data
    print(f"Strategy Active: {tui_state.strategy_data.active_strategies}")
    print(f"Decisions: {len(tui_state.strategy_data.last_decisions)}")

    # Risk Data
    print(f"Risk Max Leverage: {tui_state.risk_data.max_leverage}")
    print(f"Risk Reduce Only: {tui_state.risk_data.reduce_only_mode}")

    # Order Data
    # Note: Mock broker might not generate orders immediately unless strategy fires.
    # But we can check if the structure is there.
    print(f"Orders: {len(tui_state.order_data.orders)}")
    if tui_state.order_data.orders:
        print(f"First Order: {tui_state.order_data.orders[0]}")

    # Market Data
    print(f"Market Prices: {tui_state.market_data.prices}")

    # System Data
    print(f"System Connection: {tui_state.system_data.connection_status}")
    print(f"System Memory: {tui_state.system_data.memory_usage}")
    print(f"System CPU: {tui_state.system_data.cpu_usage}")

    # 8. Cleanup
    print("\nShutting down...")
    await bot.stop()
    try:
        await bot_task
    except asyncio.CancelledError:
        pass

    print("Verification Complete.")


if __name__ == "__main__":
    asyncio.run(main())
