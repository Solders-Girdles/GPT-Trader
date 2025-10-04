# GitHub Issue: PerpsBot._product_map Dead Code Bug

**Copy this content to create a GitHub issue manually**

---

**Title**: `[BUG] PerpsBot._product_map is dead code - initialized but never used`

**Labels**: `bug`, `refactoring`, `tech-debt`

---

## Bug Description

`PerpsBot._product_map` is initialized as an empty dict in `_init_runtime_state()` but is **never written to**, making it dead code.

**Discovered During**: Phase 0 refactoring characterization (2025-10-01)

## Current Code

**File**: `src/bot_v2/orchestration/perps_bot.py`

```python
def _init_runtime_state(self) -> None:
    # Line 82: Initialized
    self._product_map: dict[str, Product] = {}
    # ... other state ...

def get_product(self, symbol: str) -> Product:
    # Line 323: Only READ, never WRITE
    if symbol in self._product_map:  # ← Always False!
        return self._product_map[symbol]

    # Line 325-339: Always creates new Product on-the-fly
    base, _, quote = symbol.partition("-")
    # ... build Product ...
    return Product(...)  # ← Never cached!
```

## Evidence

Grep results from Phase 0 investigation:
```bash
$ grep -n "_product_map" src/bot_v2/orchestration/perps_bot.py
82:        self._product_map: dict[str, Product] = {}
323:        if symbol in self._product_map:
324:            return self._product_map[symbol]
```

**No writes found** - Never calls `self._product_map[symbol] = ...`

## Expected Behavior

**Option A: Fix Caching** (if intentional)
```python
def get_product(self, symbol: str) -> Product:
    if symbol in self._product_map:
        return self._product_map[symbol]

    product = Product(...)
    self._product_map[symbol] = product  # ← ADD THIS
    return product
```

**Option B: Remove Dead Code** (recommended)
```python
def _init_runtime_state(self) -> None:
    # Remove: self._product_map

def get_product(self, symbol: str) -> Product:
    # Remove cache check, just build on-the-fly
    base, _, quote = symbol.partition("-")
    return Product(...)
```

## Impact

**Current Impact**: Low
- ✅ No functional bug (code works correctly)
- ⚠️ Minor performance impact (recreates Product each call)
- ⚠️ Memory waste (empty dict allocated but unused)

**Refactoring Impact**: Medium
- 🚫 Blocks clean Phase 1 (MarketDataService extraction)
- ❓ Need to decide: should MarketDataService cache products?

## Investigation Needed

1. **Was caching intentional?** Check git history for `_product_map` intent
2. **Is ProductCatalog already caching?** Search for existing product cache
3. **Call frequency?** How often is `get_product()` called?
4. **Should products be fresh or cached?** Depends on use case

## Proposed Solution

**Recommendation**: Remove dead code (Option B)

**Rationale**:
- Product creation is cheap (dataclass construction)
- Products should come from ProductCatalog if caching needed
- Reduces state in PerpsBot (already a god object)
- Simplifies Phase 1 refactoring

**Implementation Steps**:
1. Remove `self._product_map: dict[str, Product] = {}` from line 82
2. Remove cache check from `get_product()` lines 323-324
3. Add docstring comment explaining no cache needed
4. Update characterization tests if needed

## Related Work

- **Phase 0 Status**: `docs/archive/refactoring-2025-q1/REFACTORING_PHASE_0_STATUS.md`
- **Dependencies**: `docs/architecture/perps_bot_dependencies.md`
- **Characterization Tests**: `tests/integration/test_perps_bot_characterization.py`

## Acceptance Criteria

- [ ] Decide: Fix caching or remove code?
- [ ] If removing: Delete `_product_map` references (lines 82, 323-324)
- [ ] If fixing: Add caching logic + unit tests
- [ ] Update characterization tests to verify behavior
- [ ] Document decision in Phase 0 status doc

---

**Priority**: Medium (doesn't block functionality, but blocks clean refactor)
**Effort**: Small (1-2 hours investigation + fix)
**Target**: Resolve before Phase 1 (MarketDataService extraction)
**Discovered By**: Phase 0 open questions investigation
