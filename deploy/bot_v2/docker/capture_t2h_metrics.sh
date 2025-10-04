#!/bin/bash
# Phase 1 Short Soak Metrics Capture (T+2H)
# Run at: 2025-10-05 00:30 UTC

echo "=== Phase 1 Short Soak Metrics (T+2H) ==="
echo ""
echo "Capture Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

echo "1. Uptime:"
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -1 | jq '.uptime_seconds' | \
  awk '{printf "  %.2f hours (%.0f seconds)\n", $1/3600, $1}'

echo ""
echo "2. Memory Trend (last 20 samples):"
echo "  Latest:"
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -1 | jq '.system.memory_used_mb' | \
  awk '{printf "    Current: %.1f MB\n", $1}'
echo "  Average (last 20 cycles):"
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -20 | \
  jq -r '.system.memory_used_mb' | awk '{sum+=$1; count++} END {printf "    Avg: %.1f MB\n", sum/count}'
echo "  Range:"
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -20 | \
  jq -r '.system.memory_used_mb' | awk 'NR==1{min=max=$1} {if($1<min)min=$1; if($1>max)max=$1} END {printf "    Min: %.1f MB, Max: %.1f MB (Delta: %.1f MB)\n", min, max, max-min}'

echo ""
echo "3. CPU Trend (last 20 samples):"
echo "  Latest:"
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -1 | jq '.system.cpu_percent' | \
  awk '{printf "    Current: %.1f%%\n", $1}'
echo "  Average:"
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -20 | \
  jq -r '.system.cpu_percent' | awk '{sum+=$1; count++} END {printf "    Avg: %.1f%%\n", sum/count}'

echo ""
echo "4. Thread Count:"
echo "  Latest:"
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -1 | jq '.system.threads' | \
  awk '{printf "    Current: %d threads\n", $1}'
echo "  Trend (first vs last 5):"
docker logs bot_v2_main 2>&1 | grep "metrics_update" | head -5 | jq -r '.system.threads' | \
  awk '{sum+=$1; count++} END {printf "    Initial Avg: %.0f threads\n", sum/count}'
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -5 | jq -r '.system.threads' | \
  awk '{sum+=$1; count++} END {printf "    Recent Avg: %.0f threads\n", sum/count}'

echo ""
echo "5. Container Health:"
echo "  Restart Count:"
docker inspect bot_v2_main | jq '.[0].RestartCount' | \
  awk '{printf "    Restarts: %d\n", $1}'
echo "  Status:"
docker ps --filter name=bot_v2_main --format "    {{.Status}}"

echo ""
echo "6. Error Analysis:"
ERROR_COUNT=$(docker logs bot_v2_main 2>&1 | grep -E "ERROR" | wc -l | tr -d ' ')
CRITICAL_COUNT=$(docker logs bot_v2_main 2>&1 | grep -E "CRITICAL" | wc -l | tr -d ' ')
echo "  Total Errors: $ERROR_COUNT"
echo "  Critical Errors: $CRITICAL_COUNT"
if [ "$ERROR_COUNT" -gt 0 ]; then
  echo "  Recent Errors (last 5):"
  docker logs bot_v2_main 2>&1 | grep -E "ERROR" | tail -5 | sed 's/^/    /'
fi

echo ""
echo "7. Current Resource Usage:"
docker stats bot_v2_main --no-stream --format "  CPU: {{.CPUPerc}}\tMemory: {{.MemUsage}}\tNet I/O: {{.NetIO}}"

echo ""
echo "8. Cycle Performance:"
echo "  Strategy Execution (avg last 20):"
docker logs bot_v2_main 2>&1 | grep "strategy_duration" | tail -20 | \
  jq -r '.duration_ms' | awk '{sum+=$1; count++} END {printf "    Avg: %.3f ms\n", sum/count}'

echo ""
echo "=== Success Criteria Assessment ==="
echo ""

# Uptime check
UPTIME=$(docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -1 | jq -r '.uptime_seconds')
if [ "$UPTIME" -gt 7200 ]; then
  echo "✅ Uptime: PASS ($(echo "$UPTIME" | awk '{printf "%.1f", $1/3600}') hours)"
else
  echo "❌ Uptime: FAIL ($(echo "$UPTIME" | awk '{printf "%.1f", $1/3600}') hours < 2 hours)"
fi

# Memory growth check
MEM_FIRST=$(docker logs bot_v2_main 2>&1 | grep "metrics_update" | head -5 | jq -r '.system.memory_used_mb' | awk '{sum+=$1; count++} END {print sum/count}')
MEM_LAST=$(docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -5 | jq -r '.system.memory_used_mb' | awk '{sum+=$1; count++} END {print sum/count}')
MEM_GROWTH=$(echo "$MEM_FIRST $MEM_LAST" | awk '{printf "%.1f", (($2-$1)/$1)*100}')
if (( $(echo "$MEM_GROWTH < 20" | bc -l) )); then
  echo "✅ Memory Growth: PASS (${MEM_GROWTH}% < 20%)"
else
  echo "❌ Memory Growth: FAIL (${MEM_GROWTH}% >= 20%)"
fi

# CPU stability check
CPU_AVG=$(docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -20 | jq -r '.system.cpu_percent' | awk '{sum+=$1; count++} END {printf "%.1f", sum/count}')
if (( $(echo "$CPU_AVG < 5" | bc -l) )); then
  echo "✅ CPU Stability: PASS (${CPU_AVG}% avg < 5%)"
else
  echo "⚠️  CPU Stability: WARNING (${CPU_AVG}% avg >= 5%)"
fi

# Thread stability check
THREADS_FIRST=$(docker logs bot_v2_main 2>&1 | grep "metrics_update" | head -5 | jq -r '.system.threads' | awk '{sum+=$1; count++} END {printf "%.0f", sum/count}')
THREADS_LAST=$(docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -5 | jq -r '.system.threads' | awk '{sum+=$1; count++} END {printf "%.0f", sum/count}')
THREAD_GROWTH=$((THREADS_LAST - THREADS_FIRST))
if [ "$THREAD_GROWTH" -le 5 ]; then
  echo "✅ Thread Stability: PASS (growth: $THREAD_GROWTH threads)"
else
  echo "❌ Thread Stability: FAIL (growth: $THREAD_GROWTH threads)"
fi

# Restart check
RESTART_COUNT=$(docker inspect bot_v2_main | jq '.[0].RestartCount')
if [ "$RESTART_COUNT" -eq 0 ]; then
  echo "✅ Container Restarts: PASS (0 restarts)"
else
  echo "❌ Container Restarts: FAIL ($RESTART_COUNT restarts)"
fi

# Error check
if [ "$CRITICAL_COUNT" -eq 0 ]; then
  echo "✅ Critical Errors: PASS (0 critical errors)"
else
  echo "❌ Critical Errors: FAIL ($CRITICAL_COUNT critical errors)"
fi

echo ""
echo "=== Short Soak Complete ==="
