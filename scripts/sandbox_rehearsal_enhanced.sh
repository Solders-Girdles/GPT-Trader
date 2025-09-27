#!/bin/bash
# Enhanced Week 3 Phase 1: Sandbox Rehearsal with detailed logging
# Captures all metrics for review

set -e  # Exit on error

# Set up environment
source /tmp/week3_env.sh 2>/dev/null || true
export EVENT_STORE_ROOT=${EVENT_STORE_ROOT:-/tmp/week3_eventstore}
export COINBASE_SANDBOX=1
export LOG_DIR=/tmp/week3_sandbox_logs
mkdir -p $LOG_DIR $EVENT_STORE_ROOT

echo "=================================================="
echo "WEEK 3 PHASE 1: ENHANCED SANDBOX REHEARSAL"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  - EventStore: $EVENT_STORE_ROOT"
echo "  - Logs: $LOG_DIR"
echo "  - Mode: Coinbase Sandbox"
echo ""

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to run test and capture metrics
run_test() {
    local test_name=$1
    local test_cmd=$2
    local log_file="$LOG_DIR/${test_name}_${TIMESTAMP}.log"
    
    echo ""
    echo "🧪 Running: $test_name"
    echo "   Log: $log_file"
    echo "   Command: $test_cmd"
    echo ""
    
    # Run with detailed logging
    eval "$test_cmd" 2>&1 | tee "$log_file"
    
    # Extract key metrics from log
    echo ""
    echo "📊 Metrics for $test_name:"
    grep -E "SIZED_DOWN|post-only.*rejected|Stop trigger|TIF:|order_metrics|strategy.*accepted|funding.*accrued" "$log_file" || echo "   No specific metrics found"
    echo ""
}

# 1. Capability probe
run_test "capability_probe" "python scripts/probe_capabilities.py --live --symbol BTC-PERP"

# 2. Test post-only crossing detection
cat > /tmp/test_post_only.py << 'EOF'
import sys
sys.path.insert(0, '.')
import os
os.environ['COINBASE_SANDBOX'] = '1'
from decimal import Decimal
from bot_v2.orchestration.mock_broker import MockBroker
from bot_v2.features.live_trade.execution_v3 import AdvancedExecutionEngine, OrderConfig

broker = MockBroker()
config = OrderConfig(enable_post_only=True, reject_on_cross=True)
engine = AdvancedExecutionEngine(broker, config)

# Get quote
quote = broker.get_quote("BTC-PERP")
print(f"Quote: bid={quote.bid}, ask={quote.ask}")

# Test 1: Non-crossing (should succeed)
order = engine.place_order(
    symbol="BTC-PERP",
    side="buy",
    quantity=Decimal("0.0001"),
    order_type="limit",
    limit_price=quote.bid - Decimal("100"),
    post_only=True,
    client_id="test_noncross"
)
print(f"Non-crossing result: {'✅ Placed' if order else '❌ Rejected'}")

# Test 2: Crossing (should reject)
order = engine.place_order(
    symbol="BTC-PERP",
    side="buy",
    quantity=Decimal("0.0001"),
    order_type="limit",
    limit_price=quote.ask + Decimal("100"),
    post_only=True,
    client_id="test_cross"
)
print(f"Crossing result: {'✅ Correctly rejected' if not order else '❌ Should have rejected'}")
print(f"Post-only rejections: {engine.order_metrics['post_only_rejected']}")
EOF

run_test "post_only_crossing" "python /tmp/test_post_only.py"

# 3. Test SIZED_DOWN logging
cat > /tmp/test_sizing.py << 'EOF'
import sys
sys.path.insert(0, '.')
from decimal import Decimal
from bot_v2.orchestration.mock_broker import MockBroker
from bot_v2.features.live_trade.execution_v3 import AdvancedExecutionEngine, OrderConfig, SizingMode

broker = MockBroker()
config = OrderConfig(
    sizing_mode=SizingMode.CONSERVATIVE,
    max_impact_bps=Decimal("10")
)
engine = AdvancedExecutionEngine(broker, config)

# Test impact sizing with large notional
market_snapshot = {
    'depth_l1': Decimal('50000'),
    'depth_l10': Decimal('200000'),
    'mid': Decimal('50000')
}

target = Decimal('150000')  # Will exceed impact
adjusted, impact = engine.calculate_impact_aware_size(target, market_snapshot)
print(f"Target: ${target:,.0f}")
print(f"Result: Adjusted=${adjusted:,.0f}, Impact={impact:.1f}bps")
# Should see SIZED_DOWN log
EOF

run_test "impact_sizing" "python /tmp/test_sizing.py"

# 4. Test stop order triggers
cat > /tmp/test_stops.py << 'EOF'
import sys
sys.path.insert(0, '.')
from decimal import Decimal
from bot_v2.orchestration.mock_broker import MockBroker
from bot_v2.features.live_trade.execution_v3 import AdvancedExecutionEngine, OrderConfig

broker = MockBroker()
config = OrderConfig(enable_stop_orders=True)
engine = AdvancedExecutionEngine(broker, config)

# Place stop order
order = engine.place_order(
    symbol="BTC-PERP",
    side="sell",
    quantity=Decimal("0.001"),
    order_type="stop",
    stop_price=Decimal("48000"),
    client_id="test_stop"
)
print(f"Stop order placed: {order}")
print(f"Triggers tracked: {list(engine.stop_triggers.keys())}")

# Simulate price move to trigger
current_prices = {"BTC-PERP": Decimal("47500")}  # Below stop
triggered = engine.check_stop_triggers(current_prices)
print(f"Triggered stops: {triggered}")
print(f"Stop metrics: {engine.order_metrics['stop_triggered']}")
EOF

run_test "stop_triggers" "python /tmp/test_stops.py"

# 5. Test TIF mapping
cat > /tmp/test_tif.py << 'EOF'
import sys
sys.path.insert(0, '.')
from decimal import Decimal
from bot_v2.orchestration.mock_broker import MockBroker
from bot_v2.features.live_trade.execution_v3 import AdvancedExecutionEngine, OrderConfig

broker = MockBroker()
config = OrderConfig(enable_ioc=True)
engine = AdvancedExecutionEngine(broker, config)

# Test GTC
order = engine.place_order(
    symbol="BTC-PERP",
    side="buy",
    quantity=Decimal("0.0001"),
    order_type="market",
    time_in_force="GTC",
    client_id="test_gtc"
)
print(f"GTC order: {'✅ Placed' if order else '❌ Failed'}")

# Test IOC
order = engine.place_order(
    symbol="BTC-PERP",
    side="buy",
    quantity=Decimal("0.0001"),
    order_type="market",
    time_in_force="IOC",
    client_id="test_ioc"
)
print(f"IOC order: {'✅ Placed' if order else '❌ Failed'}")

# Test FOK (should be gated)
order = engine.place_order(
    symbol="BTC-PERP",
    side="buy",
    quantity=Decimal("0.0001"),
    order_type="market",
    time_in_force="FOK",
    client_id="test_fok"
)
print(f"FOK order: {'✅ Correctly gated' if not order else '❌ Should be gated'}")
EOF

run_test "tif_mapping" "python /tmp/test_tif.py"

# 6. Generate metrics summary
echo ""
echo "=================================================="
echo "METRICS SUMMARY"
echo "=================================================="

# Aggregate logs
cat > /tmp/metrics_summary.py << 'EOF'
import os
import json
from pathlib import Path

log_dir = "/tmp/week3_sandbox_logs"
event_store = "/tmp/week3_eventstore"

print("\n📊 Test Results Summary:\n")

# Count key events from logs
for log_file in Path(log_dir).glob("*.log"):
    if log_file.stat().st_size > 0:
        with open(log_file) as f:
            content = f.read()
            sized_down = content.count("SIZED_DOWN")
            post_only_reject = content.count("post-only") and content.count("rejected")
            stop_triggers = content.count("Stop trigger")
            
            print(f"{log_file.name}:")
            if sized_down: print(f"  - SIZED_DOWN events: {sized_down}")
            if post_only_reject: print(f"  - Post-only rejections detected")
            if stop_triggers: print(f"  - Stop triggers: {stop_triggers}")

# Check EventStore if exists
if os.path.exists(event_store):
    metrics_files = list(Path(event_store).glob("**/metrics_*.json"))
    if metrics_files:
        print(f"\n📁 EventStore has {len(metrics_files)} metric files")

print("\n✅ Sandbox rehearsal data collected")
print(f"   Logs: {log_dir}")
print(f"   EventStore: {event_store}")
EOF

python /tmp/metrics_summary.py

echo ""
echo "=================================================="
echo "PHASE 1 CHECKLIST"
echo "=================================================="
echo ""
echo "Review the following before proceeding to Phase 2:"
echo ""
echo "□ Post-only rejections working when crossing"
echo "□ SIZED_DOWN logs show original vs adjusted notional"
echo "□ Stop orders create triggers and activate correctly"
echo "□ TIF mapping: GTC/IOC work, FOK is gated"
echo "□ No unexpected errors in logs"
echo ""
echo "Logs stored at: $LOG_DIR"
echo "EventStore at: $EVENT_STORE_ROOT"
echo ""
echo "If all checks pass, proceed to Phase 2:"
echo "  bash scripts/demo_profile_test.sh"