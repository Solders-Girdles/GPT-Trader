# Simple Claude Code Workflow

## Core Principle: Test-Driven Truth

Stop creating reports. Start writing tests.

## The Only Workflow You Need

### 1. Start Every Session
```bash
# Check what actually works
poetry run python scripts/verify_capabilities.py
```

### 2. Before Making Changes
```bash
# Write a failing test FIRST
echo "def test_my_feature(): assert False" > test_feature.py
poetry run pytest test_feature.py  # Should fail
```

### 3. Make It Work
```bash
# Implement minimal code to pass the test
# Run test again - should pass
poetry run pytest test_feature.py
```

### 4. Verify Nothing Broke
```bash
# Run verification again
poetry run python scripts/verify_capabilities.py
```

### 5. Update State (Not Reports)
```python
# Update .knowledge/PROJECT_STATE.json if component status changed
# Do NOT create a markdown report
```

## Agent Best Practices

### DO ✅
- Use agents for specific, isolated tasks
- Verify their output with tests
- Keep tasks simple and verifiable

### DON'T ❌
- Chain multiple agents together
- Trust agent claims without verification
- Create complex orchestration systems
- Write report documents

## Example: Fixing a Broken Component

```bash
# 1. Identify what's broken
poetry run python scripts/verify_capabilities.py
# Output: "data_pipeline: FAILED"

# 2. Write a test for what you want to fix
cat > test_pipeline_fix.py << 'EOF'
def test_pipeline_loads_data():
    from bot.dataflow.pipeline import DataPipeline
    pipeline = DataPipeline()
    data = pipeline.get_data("AAPL", "2024-01-01", "2024-01-31")
    assert data is not None
    assert len(data) > 0
EOF

# 3. Run test (will fail)
poetry run pytest test_pipeline_fix.py -v

# 4. Fix the actual code
# (Make minimal changes to src/bot/dataflow/pipeline.py)

# 5. Run test again (should pass)
poetry run pytest test_pipeline_fix.py -v

# 6. Verify overall system
poetry run python scripts/verify_capabilities.py

# 7. Update .knowledge/PROJECT_STATE.json
# Change data_pipeline status to "working" if fixed
```

## The Truth

- This system is 12% functional
- Only paper trading demo works
- Everything else needs fixing
- Test-driven development is the only way forward

## No More Lies

If an agent says something works, immediately run:
```bash
poetry run python scripts/verify_capabilities.py
```

If it's not in .knowledge/PROJECT_STATE.json as "working" and "verified": true, it doesn't work.