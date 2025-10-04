# T+2H Metrics Capture - Instructions

**Scheduled Time**: 2025-10-05 00:45 UTC (approximately 2 hours from soak restart at 22:45 UTC)
**Note**: Original soak (22:29 UTC) restarted due to health check configuration fix
**Purpose**: Complete Phase 1 provisional drift review with short soak validation

---

## Quick Capture (2 minutes)

**Option A - Automated Script:**
```bash
cd /Users/rj/PycharmProjects/GPT-Trader/deploy/bot_v2/docker
./capture_t2h_metrics.sh > T2H_METRICS.txt
cat T2H_METRICS.txt
```

**Option B - Manual Commands:**
```bash
# 1. Check uptime
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -1 | jq '.uptime_seconds'

# 2. Check memory
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -20 | jq '.system.memory_used_mb'

# 3. Check CPU
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -20 | jq '.system.cpu_percent'

# 4. Check threads
docker logs bot_v2_main 2>&1 | grep "metrics_update" | tail -1 | jq '.system.threads'

# 5. Check restarts
docker inspect bot_v2_main | jq '.[0].RestartCount'

# 6. Check errors
docker logs bot_v2_main 2>&1 | grep -c ERROR
```

---

## Expected Results (Success Criteria)

If soak is healthy, you should see:

✅ **Uptime**: ~7200 seconds (2 hours)
✅ **Memory**: < 20% growth from baseline (~2800 MB stable)
✅ **CPU**: < 5% average
✅ **Threads**: Stable (no continuous growth from initial ~23-29)
✅ **Restarts**: 0
✅ **Critical Errors**: 0

---

## Next Steps After Capture

**If All Criteria Pass:**
1. Mark Phase 1 drift review as "Provisionally Complete"
2. Document T+2H results in drift review
3. Close out Phase 1
4. Begin Phase 2 planning

**If Any Criteria Fail:**
1. Investigate root cause
2. Fix issue
3. Consider extending soak or re-running
4. Update drift review with findings

---

## Continuation in New Session

When you return at 00:30 UTC:

1. Run the capture script (or share metrics manually)
2. Share the output with me in a new conversation
3. I'll analyze results and update documentation
4. We'll confirm provisional completion
5. Move to Phase 2 planning

**What to share:**
- Output of `./capture_t2h_metrics.sh` (or paste T2H_METRICS.txt)
- Any anomalies or concerns you noticed

---

## Files Ready for Update

When metrics are captured, I'll update:
- `docs/monitoring/DRIFT_REVIEW_PROVISIONAL_COMPLETE.md` (add T+2H results)
- `docs/monitoring/48H_DRIFT_REVIEW_CHECKLIST.md` (mark provisional complete)
- Create Phase 1 completion summary
- Begin Phase 2 planning document

---

## Current Status

**Soak Restarted**: 2025-10-04 22:45 UTC (health check config fix)
**Reason for Restart**: Disabled incompatible HTTP health check for CLI mode
**Current Uptime**: ~1 minute
**Memory**: 2765 MB
**CPU**: 1.9%
**Threads**: 17 (initial)
**Status**: Running continuously ✅

**T+2H Target**: 2025-10-05 00:45 UTC (updated)
**Ready for Capture**: ✅ Script prepared and executable
