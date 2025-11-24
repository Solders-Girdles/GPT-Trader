# Document Trustworthiness Verification Matrix

**Purpose:** Quick reference for agents to determine which GPT-Trader documents can be trusted and which to avoid.

## Trust Levels

### ✅ GREEN - Fully Trustworthy (Primary Sources)
| Document | Trust Level | Last Verified | Notes |
|----------|-------------|---------------|-------|
| `README.md` | ✅ Current | Frequently updated | Primary system overview, spot-first docs |
| `docs/ARCHITECTURE.md` | ✅ Current | 2025-10-12 | Current architecture with coordinators |
| `docs/agents/Agents.md` | ✅ Current | Recent | Core agent orientation guide |
| `docs/AGENT_CONFUSION_POINTS.md` | ✅ Current | This matrix | Known pitfalls and verification checklist |
| `docs/agents/CLAUDE.md` | ✅ Current | Recent | Model-specific guidance |

### ⚠️ YELLOW - Use With Caution (Secondary Sources)
| Document | Trust Level | Issues | Verification Steps |
|----------|-------------|--------|-------------------|
| `docs/agents/Gemini.md` | ⚠️ Possibly dated | Check dates | Compare with Agents.md |
| `docs/testing/` guides | ⚠️ Partial updates | Verify test counts | Run `pytest --collect-only` |
| `scripts/` documentation | ⚠️ Mixed content | Cross-reference README | Check actual script headers |

### ❌ RED - AVOID (Historical/Source of Confusion)
| Document | Trust Level | Reason | Valid Historical Use |
|----------|-------------|--------|---------------------|
| `docs/reference/system_capabilities.md` | ❌ Stub to archive | Points at Dec 2024 perps-first snapshot | Use archive for history only |
| `docs/archive/` contents | ❌ Archived | Frozen snapshots | Archaeology only |
| `var/legacy/` contents | ❌ Legacy | Retired code/docs | Explicit recovery tasks |
| Dated references in ops docs | ❌ Partial updates | Mixed current/historical | None - seek current sources |

## Quick Trust Check Protocol

### Step 1: Authority Hierarchy Check
```bash
# ✅ TRUST: Primary sources are your first line of defense
README.md                     # Spot-first reality check
docs/ARCHITECTURE.md         # Current architecture (2025-10-12)

# ⚠️ VERIFY: Secondary sources need fact-checking
docs/agents/[model].md        # Compare against Agents.md

# ❌ AVOID: Historical sources are trap doors waiting to spring
docs/archive/2024/system_capabilities.md  # DANGER: Perps-first fiction
```

### Step 2: Date-Based Verification
```bash
# Always check modification dates
git log --oneline -10 -- docs/archive/2024/system_capabilities.md
# Confirms December 2024 freeze point

# For current docs, expect recent commits
git log --oneline -5 -- README.md
```

### Step 3: Cross-Reference Testing
```bash
# ✅ Spot-first reality check
poetry run pytest --collect-only               # ~1484 collected / 1483 selected / 1 deselected
poetry run coinbase-trader run --profile dev         # Spot trading by default

# ⚠️ Perps check (expect failure without INTX)
COINBASE_ENABLE_DERIVATIVES=1 poetry run coinbase-trader run --profile dev
```

## Emergency Trust Reset

When confused about system state:

1. **Clear Cache**: Forget any information older than 2025
2. **Trust Triangulation**: Cross-reference 3 sources, preferring:
   - README.md (current state)
   - ARCHITECTURE.md (system design)
   - AGENT_CONFUSION_POINTS.md (known pitfalls)
3. **Reality Test**: Run commands to verify behavior
4. **Ask Human**: If still uncertain, clarify before proceeding

## Common Document Traps

### Trap 1: "Looks Current, Is Historical"
**Example**: `docs/archive/2024/system_capabilities.md` appears as reference material but contains perps-first assumptions
**Detection**: File begins with "Current System State" but is frozen in December 2024
**Reality**: Spot-first system with ~1484 collected / 1483 selected tests vs document's 220 tests

### Trap 2: "Partial Updates"
**Example**: Testing guides mention legacy suites still active
**Detection**: References to paths/code that no longer exist
**Reality**: Only `src/gpt_trader/` is active, legacy is bundled

### Trap 3: "Authority Ambiguity"
**Example**: Multiple docs with conflicting test counts
**Detection**: Older docs cite higher historical numbers
**Reality**: Current stable count is ~1484 collected, 1483 selected (1 deselected)

## Pro Tip: Agent Trust Protocol

Before trusting any documentation:
1. Check `git log --oneline -- docs/path/to/file.md | head -5` for recency
2. Cross-reference with `AGENT_CONFUSION_POINTS.md` for known issues
3. Verify facts against testable commands (`pytest --collect-only`, etc.)
4. When in doubt, treat document as historical and seek primary sources

## Updates to This Matrix

This matrix should be updated whenever:
- New critical documents are added
- Historical documents receive clarity warnings
- Test counts or system capabilities change significantly

*Last Updated: 2025-10-18*
