---
name: Dead Code Bug - _product_map Never Used
about: _product_map is initialized but never written to in PerpsBot
title: "[BUG] PerpsBot._product_map is dead code - initialized but never used"
labels: bug, refactoring, tech-debt
assignees: ''
---

## Bug Description

`PerpsBot._product_map` is initialized as an empty dict in `_init_runtime_state()` but is **never written to**, making it dead code.

**Discovered During**: Phase 0 refactoring characterization (2025-10-01)

## Current Code

```python
# src/bot_v2/orchestration/perps_bot.py

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

## Expected Behavior

**Option A: Fix Caching** (if intentional)
```python
def get_product(self, symbol: str) -> Product:
    if symbol in self._product_map:
        return self._product_map[symbol]

    # Create product
    product = Product(...)

    # CACHE IT
    self._product_map[symbol] = product
    return product
```

**Option B: Remove Dead Code** (if not needed)
```python
def _init_runtime_state(self) -> None:
    # Remove: self._product_map: dict[str, Product] = {}
    # ... other state ...

def get_product(self, symbol: str) -> Product:
    # Remove cache check, just build on-the-fly
    base, _, quote = symbol.partition("-")
    return Product(...)
```

## Impact

**Current Impact**: Low
- No functional bug - code works correctly
- Minor performance impact (recreates Product each call)
- Memory waste (empty dict allocated but never used)

**Refactoring Impact**: Medium
- During Phase 1 (MarketDataService extraction), need to decide:
  - Should MarketDataService cache products?
  - Or delegate to ProductCatalog?
  - Or remove caching entirely?

## Investigation Needed

**Questions to Answer**:
1. Was caching intentional? (Check git history)
2. Is ProductCatalog already caching elsewhere?
3. How often is `get_product()` called? (performance impact)
4. Should products be cached or always fresh?

**Search for Clues**:
```bash
# Check if ProductCatalog exists
git grep "ProductCatalog" src/

# Check get_product call frequency
git grep "\.get_product\(" src/

# Check git history for intent
git log --all -S "_product_map" -- src/bot_v2/orchestration/perps_bot.py
```

## Proposed Solution

**Recommendation**: Remove dead code (Option B)

**Rationale**:
1. Product creation is cheap (just dataclass construction)
2. Products should come from ProductCatalog if caching needed
3. Reduces state in PerpsBot (already a god object)
4. Simplifies Phase 1 refactoring

**Implementation**:
```python
# 1. Remove _product_map from _init_runtime_state
# 2. Remove cache check from get_product
# 3. Add comment explaining why no cache needed
# 4. Consider using ProductCatalog if available
```

## Related Work

- **Phase 0**: Characterization tests (docs/architecture/REFACTORING_PHASE_0_STATUS.md)
- **Dependency Analysis**: docs/architecture/perps_bot_dependencies.md
- **Refactoring Plan**: Blocked on this decision for Phase 1

## Additional Context

**Discovery Method**:
- Grep for `_product_map` usage
- Found only 3 references:
  1. Line 82: `self._product_map: dict[str, Product] = {}` (init)
  2. Line 323: `if symbol in self._product_map:` (check)
  3. Line 324: `return self._product_map[symbol]` (never reached)

**No writes found**: Never calls `self._product_map[symbol] = ...`

## Acceptance Criteria

- [ ] Decide: Fix caching or remove code?
- [ ] If removing: Delete _product_map references
- [ ] If fixing: Add caching logic + tests
- [ ] Update characterization tests to verify behavior
- [ ] Document decision in Phase 0 status

---

**Priority**: Medium (doesn't block functionality, but blocks clean refactor)
**Effort**: Small (1-2 hours investigation + fix)
**Target**: Resolve before Phase 1 (MarketDataService extraction)
