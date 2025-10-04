"""
Characterization Tests for PerpsBot Update Marks

Tests documenting update_marks behavior and side effects.
"""

import pytest
from decimal import Decimal
from datetime import datetime, UTC
from unittest.mock import Mock

from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.features.brokerages.core.interfaces import Quote


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotUpdateMarks:
    """Characterize update_marks behavior"""

    @pytest.mark.asyncio
    async def test_update_marks_updates_mark_windows(
        self, monkeypatch, tmp_path, minimal_config, mock_quote
    ):
        """Document: update_marks must append to mark_windows"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.broker.get_quote = Mock(return_value=mock_quote)

        await bot.update_marks()

        assert len(bot.mark_windows["BTC-USD"]) == 1
        assert bot.mark_windows["BTC-USD"][0] == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_update_marks_updates_risk_manager_timestamp(
        self, monkeypatch, tmp_path, minimal_config, mock_quote
    ):
        """Document: update_marks must update risk_manager.last_mark_update"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.broker.get_quote = Mock(return_value=mock_quote)

        await bot.update_marks()

        assert "BTC-USD" in bot.risk_manager.last_mark_update
        assert isinstance(bot.risk_manager.last_mark_update["BTC-USD"], datetime)

    @pytest.mark.asyncio
    async def test_update_marks_continues_after_symbol_error(
        self, monkeypatch, tmp_path, mock_quote
    ):
        """Document: Error on one symbol must not stop processing others"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
            mock_broker=True,
        )
        bot = PerpsBot(config)

        # ETH-USD will fail, others succeed
        def get_quote_side_effect(symbol):
            if symbol == "ETH-USD":
                raise Exception("ETH quote failed")
            return mock_quote

        bot.broker.get_quote = Mock(side_effect=get_quote_side_effect)

        await bot.update_marks()

        # BTC and SOL should still update despite ETH error
        assert len(bot.mark_windows["BTC-USD"]) == 1
        assert len(bot.mark_windows["ETH-USD"]) == 0  # Failed
        assert len(bot.mark_windows["SOL-USD"]) == 1

    @pytest.mark.asyncio
    async def test_update_marks_trims_window(self, monkeypatch, tmp_path):
        """Document: mark_windows must be trimmed to max(long_ma, short_ma) + 5"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-USD"],
            short_ma=10,
            long_ma=30,
            mock_broker=True,
        )
        bot = PerpsBot(config)

        quote = Mock(spec=Quote)
        quote.last = 50000.0
        quote.ts = datetime.now(UTC)
        bot.broker.get_quote = Mock(return_value=quote)

        # Add 50 marks
        for i in range(50):
            await bot.update_marks()

        max_expected = max(config.long_ma, config.short_ma) + 5  # 35
        assert len(bot.mark_windows["BTC-USD"]) == max_expected

    @pytest.mark.asyncio
    async def test_concurrent_update_marks_calls(self, monkeypatch, tmp_path, minimal_config):
        """Document: Concurrent update_marks calls must not corrupt mark_windows or risk_manager"""
        import asyncio
        from datetime import datetime, UTC

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        # Use large MA to prevent trimming interference
        minimal_config.short_ma = 100
        minimal_config.long_ma = 200
        bot = PerpsBot(minimal_config)

        # Mock broker to return valid quotes
        quote = Mock(spec=Quote)
        quote.last = 50000.0
        quote.ts = datetime.now(UTC)
        bot.broker.get_quote = Mock(return_value=quote)

        # Run 10 concurrent update_marks calls
        num_concurrent = 10
        tasks = [bot.update_marks() for _ in range(num_concurrent)]
        await asyncio.gather(*tasks)

        # Verify no corruption
        marks = bot.mark_windows["BTC-USD"]
        assert len(marks) == num_concurrent  # All updates succeeded
        assert all(isinstance(m, Decimal) for m in marks)  # No type corruption

        # Verify risk_manager updated (at least once)
        assert "BTC-USD" in bot.risk_manager.last_mark_update
        assert isinstance(bot.risk_manager.last_mark_update["BTC-USD"], datetime)

    @pytest.mark.asyncio
    async def test_update_marks_with_none_quote(self, monkeypatch, tmp_path, minimal_config):
        """Document: None quote must be handled gracefully (logged, not raised)"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.broker.get_quote = Mock(return_value=None)

        # Should not raise - error is logged and processing continues
        await bot.update_marks()

        # Mark window should remain empty (no valid quote)
        assert len(bot.mark_windows["BTC-USD"]) == 0

        # risk_manager should not be updated for this symbol
        assert "BTC-USD" not in bot.risk_manager.last_mark_update

    @pytest.mark.asyncio
    async def test_update_marks_with_invalid_price(self, monkeypatch, tmp_path, minimal_config):
        """Document: Invalid mark prices (<=0) must be handled gracefully (logged, not raised)"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Mock broker to return invalid price
        invalid_quote = Mock(spec=Quote)
        invalid_quote.last = 0.0  # Invalid: zero price
        invalid_quote.ts = datetime.now(UTC)
        bot.broker.get_quote = Mock(return_value=invalid_quote)

        # Should not raise - error is logged and processing continues
        await bot.update_marks()

        # Mark window should remain empty (invalid price rejected)
        assert len(bot.mark_windows["BTC-USD"]) == 0

        # risk_manager should not be updated for this symbol
        assert "BTC-USD" not in bot.risk_manager.last_mark_update

    @pytest.mark.asyncio
    async def test_exception_handling_preserves_state(self, monkeypatch, tmp_path):
        """Document: Exceptions during mark update must not corrupt risk_manager state"""
        from bot_v2.orchestration.configuration import BotConfig, Profile
        from datetime import datetime, UTC

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        # Multi-symbol config to test partial failure
        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
            mock_broker=True,
        )
        bot = PerpsBot(config)

        # Setup: BTC succeeds, ETH fails, SOL succeeds
        def get_quote_side_effect(symbol):
            if symbol == "ETH-USD":
                raise RuntimeError("Simulated broker error for ETH")
            quote = Mock(spec=Quote)
            quote.last = 50000.0
            quote.ts = datetime.now(UTC)
            return quote

        bot.broker.get_quote = Mock(side_effect=get_quote_side_effect)

        # Capture initial state
        initial_state = dict(bot.risk_manager.last_mark_update)

        # Run update_marks
        await bot.update_marks()

        # Verify successful symbols updated
        assert "BTC-USD" in bot.risk_manager.last_mark_update
        assert "SOL-USD" in bot.risk_manager.last_mark_update

        # Verify failed symbol didn't corrupt state
        # (either not in dict, or has old timestamp if it existed before)
        if "ETH-USD" in initial_state:
            # If existed before, timestamp should be unchanged
            assert bot.risk_manager.last_mark_update["ETH-USD"] == initial_state["ETH-USD"]
        else:
            # If didn't exist, should still not exist (or have no timestamp update)
            assert "ETH-USD" not in bot.risk_manager.last_mark_update

        # Verify mark windows
        assert len(bot.mark_windows["BTC-USD"]) == 1
        assert len(bot.mark_windows["ETH-USD"]) == 0  # Failed
        assert len(bot.mark_windows["SOL-USD"]) == 1
