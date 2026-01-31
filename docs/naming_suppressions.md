# Naming Standards: Strict Mode + Suppression Rules

---
status: current
last-updated: 2026-01-31
---

This document explains **how the naming-standards check works**, what it enforces, and
how (and when) to suppress it.

The goal is to keep naming consistent **without** turning every PR into a rename-fest.

## What enforces naming

CI runs a strict naming scan via:

- `uv run python scripts/agents/naming_inventory.py --strict --quiet`

The banned/flagged patterns are defined in:

- `config/agents/naming_patterns.yaml`

The canonical naming guidelines live in:

- `docs/naming.md`

## Strict mode philosophy

Strict mode exists because:
- short, ambiguous abbreviations create compounding confusion (especially for agents)
- naming drift increases coupling (“cfg”, “config”, “configuration” all referring to the same concept)
- inconsistent naming makes search/grep ineffective

Default behavior: **rename to the clear term** (e.g. `config`, `manager`, `service`) rather than suppress.

## When suppression is allowed

Suppressions are allowed when one of the following is true:

1) **External interface compatibility**
   - Example: a third-party API field name that is canonically abbreviated.

2) **Test-only fixture/data realism**
   - Example: you are asserting against a literal string in a config snapshot and renaming it would
     reduce clarity of the test intent.

3) **Generated code or generated inventories**
   - If the generator output contains a banned token, fix the generator or explicitly document it.

4) **Non-code text where renaming would be misleading**
   - Example: quoting log output, external docs, RFC text.

If it’s production code under `src/`, suppression should be rare.

## How to suppress (supported patterns)

### Inline suppression

Add this comment on the **same line** as the flagged token:

```python
# naming: allow
```

Example:

```python
# naming: allow
legacy_cfg = load_legacy_cfg_format()
```

Use this for one-off cases.

### File-level suppression

Prefer not to do this unless the file is:
- generated
- a compatibility shim
- a legacy migration file that will be deleted

If you truly need file-level suppression, add a short explanation at the top of the file
and reference a tracking issue.

## Tests vs production code guidance

- **Production code (`src/`)**: fix naming (rename) unless there is an external compatibility reason.
- **Tests (`tests/`)**: prefer clarity, but it’s acceptable to suppress in literals/examples.
  - If a test name is tripping strict mode (like `cfg_run`), rename it to something explicit (`config_run`).

## Examples (recommended)

- ❌ `cfg` → ✅ `config`
- ❌ `svc` → ✅ `service`
- ❌ `mgr` → ✅ `manager`
- ✅ `qty` allowed only when clearly trading-domain quantity (otherwise use `quantity`)
