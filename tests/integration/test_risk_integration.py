import pytest
from decimal import Decimal

from bot_v2.orchestration.perps_bot import BotConfig, PerpsBot, Profile
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType
from bot_v2.features.live_trade.risk import ValidationError

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_kill_switch_blocks_trading(monkeypatch):
    # Ensure no WS threads
    monkeypatch.setenv('DISABLE_WS_STREAMING', '1')
    bot = PerpsBot(BotConfig.from_profile(Profile.DEV.value))

    # Seed recent marks and timestamps
    sym = bot.config.symbols[0]
    bot.mark_windows[sym] = [Decimal('100')] * 30
    bot.risk_manager.last_mark_update[sym] = bot.risk_manager.last_mark_update.get(sym) or __import__('datetime').datetime.utcnow()

    # Flip kill switch
    bot.risk_manager.config.kill_switch_enabled = True

    attempted_before = bot.order_stats['attempted']
    await bot.process_symbol(sym)

    assert bot.order_stats['attempted'] == attempted_before, "Kill switch should block order attempts"


@pytest.mark.asyncio
async def test_reduce_only_mode_enforcement(monkeypatch):
    monkeypatch.setenv('DISABLE_WS_STREAMING', '1')
    bot = PerpsBot(BotConfig.from_profile(Profile.DEV.value))

    # Enable reduce-only
    bot.risk_manager.config.reduce_only_mode = True

    sym = bot.config.symbols[0]
    product = bot.get_product(sym)

    with pytest.raises(ValidationError):
        # Attempt to place order that would increase position
        await bot._place_order(
            symbol=sym,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal('0.1'),
            product=product,
            reduce_only=False,
        )


@pytest.mark.asyncio
async def test_reduce_only_state_listener_syncs_bot(monkeypatch, tmp_path):
    monkeypatch.setenv('EVENT_STORE_ROOT', str(tmp_path))
    monkeypatch.setenv('DISABLE_WS_STREAMING', '1')
    bot = PerpsBot(BotConfig.from_profile(Profile.DEV.value))

    # Initial state mirrors config
    assert bot.is_reduce_only_mode() is bool(bot.config.reduce_only_mode)

    # Toggle via risk manager and ensure bot mirrors state
    bot.risk_manager.set_reduce_only_mode(True, reason="integration_test")
    assert bot.is_reduce_only_mode() is True
    assert bot.config.reduce_only_mode is True

    bot.risk_manager.set_reduce_only_mode(False, reason="integration_test_reset")
    assert bot.is_reduce_only_mode() is False
    assert bot.config.reduce_only_mode is False
