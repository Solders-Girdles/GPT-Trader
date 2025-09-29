# Naming Inventory Tooling – Design Draft

This document defines the scope and output format for the Sprint 0 naming inventory scripts (backlog items T-002 and T-007). It should guide implementation and review of the initial automation pass.

## Objectives
- Detect banned abbreviations and naming exceptions outlined in `docs/agents/naming_standards_outline.md` across `src/`, `tests/`, `scripts/`, and config directories.
- Produce machine-readable output that can seed rename backlog items (`R-***`) and human-readable summaries for quick triage.
- Keep the implementation lightweight (standard library only) so agents can run it without extra dependencies.

## Targets & Patterns
- **Abbreviations:** `cfg`, `svc`, `mgr`, `util`, `qty`, `amt`, `calc`, `upd` (case-sensitive where appropriate). # naming: allow
- **Naming mismatches:** camelCase identifiers inside Python modules, environment variable keys not using `UPPER_SNAKE_CASE` (future enhancement).
- **Opt-out markers:** Lines containing `# naming: allow` should be skipped to enable deliberate exceptions.

## Output
- **JSON report** (`var/agents/naming_inventory.json`, git-ignored): structured list with fields
  - `path`: relative file path.
  - `line`: line number (1-based).
  - `pattern`: matched token.
  - `context`: trimmed line content.
- **Markdown summary** (recommended: `docs/agents/naming_inventory.md`; historical Sprint 0 exports remain accessible in git history)
  - Group results by pattern and subsystem (top-level directory).
  - Include counts per pattern and top offenders for quick review.

## CLI Interface
- Script entry point: `python scripts/agents/naming_inventory.py [--summary docs/agents/naming_inventory.md] [--json var/agents/naming_inventory.json]`.
- Optional flags:
  - `--patterns cfg,svc,...` to override the default search list.
  - `--paths src tests scripts` to limit the scan scope.
  - `--quiet` to suppress stdout summary.

## Implementation Notes
- Use `pathlib` and `re` from the standard library; avoid shelling out to external tools to stay portable.
- Treat binary files defensively (skip if decoding fails).
- When writing JSON, ensure deterministic ordering for reproducible diffs.
- Markdown summary should sort patterns descending by hit count and include total counts.

## Next Steps
1. Scaffold `scripts/agents/` package (ensure `__init__.py`).
2. Implement `naming_inventory.py` according to this design.
3. Update backlog items T-002/T-007 with links to the script once merged.
4. Wire the script into the Sprint 0 status reporting workflow (reference in rename plan).
