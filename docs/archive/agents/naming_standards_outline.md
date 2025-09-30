# GPT-Trader Naming Standards (Draft Outline)

This draft will be socialized during Sprint 0 before landing as policy in `CONTRIBUTING.md`. Use it as the working source of truth while we solicit feedback from maintainers and agents.

## 1. Scope & Goals
- Increase clarity and consistency across code, configuration, CLI surfaces, and docs.
- Reduce rename churn by setting clear casing, terminology, and abbreviation rules.
- Preserve compatibility with external APIs while documenting sanctioned exceptions.

## 2. Naming Categories & Rules

### 2.1 Modules & Packages
- **Format:** `snake_case` for Python modules and packages.
- **Rules:** Descriptive nouns or noun phrases (`risk_limits`, not `rl`). Avoid double abbreviations.
- **Banned abbreviations:** `cfg`, `svc`, `mgr`, `util`; prefer explicit names (`config`, `service`, `manager`, `utilities`). # naming: allow

### 2.2 Classes & Data Structures
- **Format:** `PascalCase`.
- **Rules:** Classes should be nouns; mixins and protocols should carry suffixes (`...Mixin`, `...Protocol`). Avoid `Helper` unless the class truly provides utilities across slices.
- **Banned abbreviations:** `Mgr`, `Cfg`, `Svc`, `Impl`.

### 2.3 Functions & Methods
- **Format:** `snake_case`.
- **Rules:** Start with verbs for actions (`fetch_account_snapshot`); use noun phrases for pure accessors (`risk_limits`). Use suffix `async` only when defining an explicit async variant.
- **Banned abbreviations:** `calc`, `upd`, `cfg`, `mgr`.

### 2.4 Variables & Attributes
- **Format:** `snake_case` for mutable state; `UPPER_SNAKE_CASE` for constants.
- **Rules:** Prefer full words; use domain terms that match docs (`portfolio`, `position`, `exposure`).
- **Quantity terminology:** Spell out `quantity` instead of `qty`; legacy identifiers may remain temporarily with `# naming: allow` markers during migration.
- **Banned abbreviations:** `amt` (prefer `amount`), `qty` (prefer `quantity`), `cfg`.

### 2.5 Configuration Keys & Environment Variables
- **Format:**
  - Config files (JSON/YAML/TOML): `snake_case` keys.
  - Environment variables: `UPPER_SNAKE_CASE`.
- **Rules:** Prefix env vars with `COINBASE_`, `RISK_`, `PERPS_`, etc., to signal subsystem. Document every new key in README + config templates.
- **Banned abbreviations:** `CFG`, `CONF`, `ENV` as suffix-only markers.

### 2.6 CLI Flags & Commands
- **Format:** Long options `--kebab-case`; short options only when already established.
- **Rules:** Align CLI terminology with configuration keys; provide backwards-compatible aliases for at least one sprint when renaming.

### 2.7 External API Exceptions
- When interfacing with third-party APIs (Coinbase, Prometheus), keep their canonical field names.
- Document each exception inline (comment) and in the reference docs to avoid accidental renames.

## 3. Abbreviation Policy
- Maintain a shared glossary in `docs/agents/glossary.md` (to be created) listing approved abbreviations (`PnL`, `MVP`, `API`).
- Any new abbreviation requires maintainer approval; add to glossary when accepted.

## 4. Review & Enforcement
- Incorporate naming checks into `scripts/agents/preflight.py` once implemented.
- Require code review to flag deviations; reference this document in feedback.
- Add a Definition of Done item: "Naming complies with standards or documented exception is provided."

## 5. Adoption Plan
1. Sprint 0: socialization, collect feedback, finalize glossary.
2. Sprint 1: codify in `CONTRIBUTING.md` and integrate into automation.
3. Sprint 2: enforce via governance (rename changelog, dashboards).

## 6. Open Questions
- How to handle legacy third-party conventions embedded in archived tests?
- Should we auto-generate reports for config/env var mismatches or rely on manual review initially?
- Do we need temporary aliases for external integrators (if any) beyond our repo?

Please leave comments in Pull Requests or the shared discussion doc so we can refine this outline before publishing the final standard.
