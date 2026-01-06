# GPT-Trader Naming Standards

This document establishes the naming conventions for the GPT-Trader codebase. Adherence to these standards ensures clarity, consistency, and maintainability across the repository.

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
- **Quantity terminology:** `qty` is approved for trading domain use (see glossary); prefer `quantity` for non-trading contexts.
- **Banned abbreviations:** `amt` (prefer `amount`), `cfg` (prefer `config`).

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
- Incorporate naming checks into `scripts/agents/preflight.py` (planned).
- Require code review to flag deviations; reference this document in feedback.
- Add a Definition of Done item: "Naming complies with standards or documented exception is provided."
