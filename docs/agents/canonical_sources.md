# Canonical Sources: Agent Doctrine & Generated-Artifact Scope

---
status: current
---

Makes the [information-architecture rule](../INFORMATION_ARCHITECTURE.md) — each
fact lives once; generated inventories are derived truth — operational for two
things future agents inherit: **agent doctrine** (the rules a run relies on) and
the committed **`var/agents/**` generated context**.

## 1. Doctrine homing audit

The local `.claude/CLAUDE.md` is **gitignored** (`.gitignore:93`) and therefore a
per-checkout editor convenience mirror, not a source of truth. Every canonical
rule it restates already has a tracked home; no canonical-only rule remains
stranded there. Keep it as a mirror — do **not** point tracked docs *at* it.

| Canonical rule | Tracked home (owner) |
|----------------|----------------------|
| CLI human output (`✓/✗`, exit codes) + `CliResponse`/`CliErrorCode` envelope | [conventions.md](conventions.md) (added in #1074) |
| Trade-idea CLI worked example | [TRADE_IDEA_CLI_SPEC.md](../specs/TRADE_IDEA_CLI_SPEC.md) (surface-specific) |
| `monkeypatch` over `patch()`, mocking local imports, `CliResponse` test asserts | [testing.md](../testing.md), [CONTRIBUTING.md](../../CONTRIBUTING.md) |
| Banned abbreviations / approved terms | [naming.md](../naming.md), [glossary.md](glossary.md) |
| Merge discipline, quality gate, artifact-freshness step | [AGENTS.md](../../AGENTS.md) |

**Owner split (general vs. surface-specific):** [conventions.md](conventions.md)
owns the *general* CLI response/output contract. Command-family specs (e.g.
[TRADE_IDEA_CLI_SPEC.md](../specs/TRADE_IDEA_CLI_SPEC.md)) are *self-canonical*
for their own surface and should reference `conventions.md` rather than restate
the envelope.

## 2. `var/agents/**` generated-artifact scope

### How it is produced and gated

- **Produced by** [`regenerate_all.py`](../../scripts/agents/regenerate_all.py)
  (`uv run agent-regenerate`), one generator per resource group.
- **Validated by** [`agent_artifacts.py`](../../scripts/agents/agent_artifacts.py)
  (`uv run agent-artifacts validate` / `package` / `verify-package`), which
  reads every resource listed in [`var/agents/index.json`](../../var/agents/index.json).
- **Freshness-gated in CI** by the *Agent Artifacts Freshness* job in
  [`ci.yml`](../../.github/workflows/ci.yml) (runs `agent-regenerate --verify`;
  a stale tree fails the build).
- **Packaged/published** by the scheduled
  [`agent-artifacts-refresh.yml`](../../.github/workflows/agent-artifacts-refresh.yml).

Because all ten groups are registered resources in `index.json`, none can be
dropped from the committed tree without also updating the index, the validator
contract, and the freshness gate. **This PR removes nothing.**

### Verdict per resource group

| Resource (generator) | Committed files | Verdict |
|----------------------|-----------------|---------|
| `schemas`, `models`, `logging`, `observability`, `configuration`, `validation`, `broker`, `health` | small JSON/MD inventories | **Keep committed.** Derived truth read as agent context and validated/packaged; low churn. |
| `testing` (`generate_test_inventory.py`) | `index.json`, `markers.json`, `source_test_map.json` | **Keep committed.** The large `test_inventory.json` (~1.8 MB) is *already* gitignored (`.gitignore:150`) and regenerated locally — the working template for advisory/generated-only treatment. |
| `reasoning` (`generate_reasoning_artifacts.py`) | 8 maps × `.md`/`.json`/`.dot` (24 files) | **Keep committed, flagged as the primary churn candidate.** The `.md` summaries are curated in [reasoning_artifacts.md](reasoning_artifacts.md); the `.json`/`.dot` machine forms regenerate on nearly any `src/**` change and dominate diff noise. |

### Churn verdict & next step

The `reasoning/*.{json,dot}` machine forms are the clearest candidates for
future **advisory/generated-only** treatment (gitignore + regenerate/package on
demand, keeping the curated `.md` committed) — mirroring the `test_inventory.json`
precedent. Doing so requires updating `index.json`, the `agent_artifacts.py`
validator expectations, and the freshness gate together, so per the issue's
out-of-scope note it is **deferred to a separate decision record**, not done
here.
