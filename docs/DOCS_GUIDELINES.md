# Documentation Guidelines

## Purpose

This document defines the structure and maintenance guidelines for GPT-Trader documentation to prevent clutter and maintain clarity.

---

## Directory Structure

```
docs/
├── README.md                    # Index of all documentation
├── ARCHITECTURE.md              # System architecture (authoritative)
├── QUICK_START.md              # Getting started guide
├── SECURITY.md                 # Security guidelines
│
├── agents/                     # AI agent guides
│   ├── Agents.md              # Master agent guide (authoritative)
│   ├── CLAUDE.md              # Claude-specific notes
│   └── Gemini.md              # Gemini-specific notes
│
├── guides/                     # Feature-specific guides
│   ├── agents.md              # Agent development patterns
│   ├── complete_setup_guide.md
│   ├── paper_trading.md
│   └── testing.md
│
├── reference/                  # Technical reference docs
│   ├── coinbase_complete.md
│   ├── system_capabilities.md
│   └── trading_logic_perps.md
│
├── testing/                    # Test documentation
│   ├── coinbase_coverage_matrix.md
│   └── selective_test_runner.md
│
└── archive/                    # Historical/deprecated docs
    ├── agents/                 # Archived agent initiatives
    └── legacy-deployment/      # Deprecated deployment docs
```

---

## Documentation Types

### 1. **Authoritative Documents** (Keep Updated)

These are the single source of truth:

- `ARCHITECTURE.md` - System design, component structure, data flow
- `README.md` (root + docs/) - Entry points and navigation
- `SECURITY.md` - Security policies and procedures
- `agents/Agents.md` - Master agent guide

**Rules:**
- ✅ Update these when architecture/processes change
- ✅ Include "last-updated" date
- ❌ Don't create alternative versions
- ❌ Don't create temporary "update" docs

### 2. **Feature Guides** (guides/)

Step-by-step instructions for specific features:

- Paper trading setup
- Testing workflows
- Development patterns

**Rules:**
- ✅ Focused on HOW TO do something
- ✅ Include code examples
- ❌ Don't duplicate architecture details
- ❌ Don't create temporary guides

### 3. **Reference Documentation** (reference/)

Technical specifications and API details:

- Coinbase API integration
- Trading logic details
- System capabilities

**Rules:**
- ✅ Technical details and specifications
- ✅ API references and interfaces
- ❌ Don't duplicate guides or architecture
- ❌ Don't include step-by-step tutorials

### 4. **Archive** (archive/)

Historical documents preserved for reference:

- Completed initiatives
- Deprecated features
- Legacy deployment docs

**Rules:**
- ✅ Mark with "ARCHIVED" header
- ✅ Include archival date and reason
- ✅ Keep for historical context
- ❌ Don't update archived docs
- ❌ Don't reference from active docs

---

## Anti-Patterns to Avoid

### ❌ Temporary Status Documents

**Bad:**
```
STREAMLINING_SUMMARY.md
PHASE2_REPORT.md
REFACTORING_STATUS.md
```

**Good:**
Update `ARCHITECTURE.md` with a "Recent Changes" section instead.

### ❌ Redirect Stubs

**Bad:**
```markdown
# Old Document

This has moved to new/location.md
```

**Good:**
Delete the old file and update any references.

### ❌ Duplicate Agent Guides

**Bad:**
```
agents/claude_complete_guide.md
agents/gemini_complete_guide.md
agents/full_agent_reference.md
```

**Good:**
One master `Agents.md` with agent-specific appendices.

### ❌ Versioned Copies

**Bad:**
```
architecture_v1.md
architecture_v2.md
architecture_current.md
```

**Good:**
One `ARCHITECTURE.md` with git history for versions.

---

## Maintenance Checklist

### When Adding New Documentation

- [ ] Check if existing doc can be updated instead
- [ ] Place in correct directory (guides/reference/archive)
- [ ] Update `docs/README.md` index
- [ ] Add "last-updated" date
- [ ] Remove any temporary docs it replaces

### When Archiving Documentation

- [ ] Add "ARCHIVED" header with date
- [ ] Move to `docs/archive/`
- [ ] Remove from `docs/README.md` index
- [ ] Update references in active docs
- [ ] Don't delete (keep for history)

### Monthly Review

- [ ] Check for outdated "last-updated" dates
- [ ] Identify duplicate or redundant docs
- [ ] Archive completed initiative docs
- [ ] Update `docs/README.md` if structure changed

---

## Examples

### ✅ Good: Update in Place

```markdown
# ARCHITECTURE.md

**Recent Changes (2025-09-29):**
- Archived experimental features
- Refactored large files into modules
- Updated test coverage to 443/448 passing
```

### ❌ Bad: Temporary Document

```markdown
# STREAMLINING_SUMMARY.md

This document tracks the streamlining effort...
[temporary status information]
```

### ✅ Good: Clear Archive

```markdown
# docs/archive/agents/ai_workflow_initiative.md

> **ARCHIVED: 2024-09**
> Initiative completed. Tooling integrated.
> Kept for historical reference.

[content...]
```

---

## Document Lifecycle

```
New Need → Check Existing Docs
    ↓
    ├─ Exists → Update in place
    └─ Doesn't exist → Create in correct dir
        ↓
        Active use (kept updated)
        ↓
        Completed/Deprecated → Archive
        ↓
        Historical reference (frozen)
```

---

## Quick Reference

**Adding feature docs?** → `docs/guides/`
**Adding API specs?** → `docs/reference/`
**Updating architecture?** → Edit `docs/ARCHITECTURE.md`
**Initiative complete?** → Archive in `docs/archive/`
**Temporary status doc?** → ❌ Don't create, update existing instead

---

*Last updated: 2025-09-29*
