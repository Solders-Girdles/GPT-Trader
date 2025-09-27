#!/bin/bash
# Emergency Kill Switch for Week 3 Bot
# Immediately stops trading and optionally closes positions

echo "=================================================="
echo "🚨 EMERGENCY KILL SWITCH ACTIVATED 🚨"
echo "=================================================="
echo ""

# Find and kill the bot process
echo "1. Stopping bot process..."
pkill -f "run_perps_bot_v2_week3.py" && echo "   ✅ Bot stopped" || echo "   ⚠️ Bot not running"

# Check for open positions
echo ""
echo "2. Checking for open positions..."
python -c "
import os
import sys
sys.path.insert(0, '.')
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig

config = APIConfig(
    api_key=os.environ.get('COINBASE_API_KEY', ''),
    api_secret=os.environ.get('COINBASE_API_SECRET', ''),
    enable_derivatives=True
)

broker = CoinbaseBrokerage(config)
positions = broker.get_positions()

if positions:
    print(f'   ⚠️ Found {len(positions)} open positions:')
    for pos in positions:
        print(f'      - {pos.symbol}: {pos.qty} @ {pos.avg_price}')
    print('')
    print('   Run emergency_close_all.py to close positions')
else:
    print('   ✅ No open positions')
"

echo ""
echo "3. Reduce-only mode instructions:"
echo "   To restart in reduce-only mode (exits only):"
echo "   python scripts/run_perps_bot_v2_week3.py --reduce-only"
echo ""
echo "4. Emergency position close:"
echo "   To close all positions immediately:"
echo "   python scripts/emergency_close_all.py --confirm"
echo ""
echo "Kill switch complete. Review logs at /tmp/week3_*.log"