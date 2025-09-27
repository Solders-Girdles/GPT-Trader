# How to Fix Bugs

## Steps

### 1. Identify the Slice
```bash
# Find which slice contains the issue
grep -r "problematic_function" src/bot_v2/features/
```

### 2. Navigate to Slice
```bash
cd src/bot_v2/features/[slice_name]
```

### 3. Run Slice Test
```bash
# Test the specific slice
poetry run python src/bot_v2/test_[slice_name].py
```

### 4. Fix the Issue
- Make changes ONLY within the slice
- Don't import from other slices as a fix
- Implement missing functionality locally if needed

### 5. Verify Fix
```bash
# Re-run the test
poetry run python src/bot_v2/test_[slice_name].py

# Check isolation still maintained
grep -r "from bot_v2.features" src/bot_v2/features/[slice_name]/
```

### 6. Test Other Slices
```bash
# Ensure no side effects
poetry run python src/bot_v2/test_all_slices.py
```

### 7. Update STATE.json
If the bug fix changes system status, update STATE.json

## Common Issues

### Import Errors
- Solution: Implement functionality locally within the slice

### Missing Functionality
- Solution: Add to the slice, don't import from elsewhere

### Test Failures
- Solution: Fix within slice boundaries only

## Remember
- Changes stay within one slice
- No ripple effects to other slices
- Test in isolation