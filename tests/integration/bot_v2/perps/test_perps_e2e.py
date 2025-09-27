#!/usr/bin/env python3
"""
Integration test: verify the dev-profile perps bot can run a minimal cycle.
"""

import pytest
from pathlib import Path
import sys
import asyncio

pytestmark = pytest.mark.integration


def test_perps_bot_minimal_cycle():
    # Import the actual PerpsBot from the correct module
    from bot_v2.orchestration.perps_bot import PerpsBot, BotConfig

    config = BotConfig.from_profile('dev', dry_run=True, symbols=["BTC-PERP"], update_interval=1)
    bot = PerpsBot(config)

    async def run_once():
        # PerpsBot doesn't have initialize(), just run the cycle
        await bot.run_cycle()

    asyncio.run(run_once())
