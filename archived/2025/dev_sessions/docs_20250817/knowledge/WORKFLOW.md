# V2 Slice-First Workflow

## Core Principle: Slice-Driven Truth

Stop creating reports. Start testing slices.

## The V2 Workflow

### 1. Start Every V2 Session
```bash
# Check which V2 slices actually work
poetry run python src/bot_v2/test_all_slices.py

# Check V2 slice isolation compliance
grep -r "from bot_v2.features" src/bot_v2/features/ && echo "❌ Isolation violations" || echo "✅ Complete isolation"

# Quick V2 slice health check
cat .knowledge/PROJECT_STATE.json | jq '.v2_slices'
```

### 2. Before Making V2 Changes
```bash
# Write a failing V2 slice test FIRST
echo "from src.bot_v2.features.my_slice import *; assert False" > src/bot_v2/test_my_slice.py
poetry run python src/bot_v2/test_my_slice.py  # Should fail
```

### 3. Make V2 Slice Work
```bash
# Implement slice with complete isolation
# No imports from other slices
# Local implementations only
poetry run python src/bot_v2/test_my_slice.py  # Should pass
```

### 4. Verify V2 Isolation Maintained
```bash
# Check slice isolation still intact
grep -r "from bot_v2.features" src/bot_v2/features/my_slice/ && echo "❌ Violation" || echo "✅ Isolated"

# Test slice loads independently
python -c "from src.bot_v2.features.my_slice import *; print('✅ Independent')"

# Run all V2 slices to ensure no breakage
poetry run python src/bot_v2/test_all_slices.py
```

### 5. Update V2 State (Not Reports)
```bash
# Update .knowledge/PROJECT_STATE.json v2_slices section
python -c "
import json
state = json.load(open('.knowledge/PROJECT_STATE.json'))
state['v2_slices']['my_slice']['status'] = 'working'
state['v2_slices']['my_slice']['isolation'] = 'verified'
json.dump(state, open('.knowledge/PROJECT_STATE.json', 'w'), indent=2)
"
```

## V2 Agent Best Practices

### DO ✅
- Use agents for specific V2 slice tasks
- Delegate single slice work with complete context
- Verify V2 slice isolation after agent work
- Test slice independence after changes
- Update .knowledge/PROJECT_STATE.json v2_slices section

### DON'T ❌
- Give agents multiple slice responsibilities
- Trust agent claims about isolation without verification
- Let agents create cross-slice dependencies
- Allow V1 imports in V2 context
- Create new report files (*_REPORT.md)
- Create complex orchestration systems
- Write report documents

## Example: Fixing a Broken V2 Slice

```bash
# 1. Identify which V2 slice is broken
poetry run python src/bot_v2/test_all_slices.py
# Output: "backtest: FAILED"

# 2. Write a test for the V2 slice fix
cat > src/bot_v2/test_backtest_fix.py << 'EOF'
def test_backtest_slice_works():
    from src.bot_v2.features.backtest import BacktestEngine
    engine = BacktestEngine()
    result = engine.run_test_backtest()
    assert result is not None
    assert result.success == True
EOF

# 3. Run V2 slice test (will fail)
poetry run python src/bot_v2/test_backtest_fix.py

# 4. Fix the V2 slice code
# Make minimal changes to src/bot_v2/features/backtest/
# Maintain complete slice isolation

# 5. Run V2 slice test again (should pass)
poetry run python src/bot_v2/test_backtest_fix.py

# 6. Verify V2 slice isolation maintained
grep -r "from bot_v2.features" src/bot_v2/features/backtest/ || echo "✅ Isolated"

# 7. Verify overall V2 system
poetry run python src/bot_v2/test_all_slices.py

# 8. Update .knowledge/PROJECT_STATE.json v2_slices section
python -c "
import json
state = json.load(open('.knowledge/PROJECT_STATE.json'))
state['v2_slices']['backtest']['status'] = 'working'
state['v2_slices']['backtest']['isolation'] = 'verified'
json.dump(state, open('.knowledge/PROJECT_STATE.json', 'w'), indent=2)
"
```

## The V2 Truth

- V2 system uses complete slice isolation (~500 tokens per slice)
- Each slice works independently 
- No cross-slice dependencies allowed
- Slice-driven development is the V2 way forward

## V2 Verification

If an agent says a V2 slice works, immediately run:
```bash
# Test specific slice
poetry run python src/bot_v2/test_[slice].py

# Check slice isolation
grep -r "from bot_v2.features" src/bot_v2/features/[slice]/

# Verify in state
cat .knowledge/PROJECT_STATE.json | jq '.v2_slices.[slice]'
```

If it's not in .knowledge/PROJECT_STATE.json v2_slices with status "working" and isolation "verified", it doesn't work.