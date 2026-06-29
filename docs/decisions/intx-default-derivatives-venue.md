# INTX default derivatives venue

---
status: proposed
date: 2026-06-28
deciders: rj
supersedes:
superseded-by:
---

## Status

Partial. A.1 has landed; the A.2 default-venue decision is still pending.

This packet records an unresolved owner decision. It is not a live-operation
runbook and does not authorize broker/API calls, canary operations, live trading
commands, money movement, or order submission.

## Authority

- [Direction](../DIRECTION.md): current scope is Coinbase spot plus CFM futures;
  INTX perpetuals are frozen. The accepted record is
  [accept-staged-autonomy-direction](accept-staged-autonomy-direction.md).
- [Deprecations](../DEPRECATIONS.md): `COINBASE_ENABLE_DERIVATIVES` is already
  deprecated in favor of `COINBASE_ENABLE_INTX_PERPS`, so this packet does not
  propose reintroducing that alias.

## Problem

INTX is a frozen venue, yet it remains the default Coinbase derivatives type in
config. The fail-open venue helper has been fixed, but choosing a frozen venue
is still the path of least resistance once derivatives are enabled without an
explicit venue override.

## Current Evidence

| Evidence | Current state |
| --- | --- |
| [bot_config.py](../../src/gpt_trader/app/config/bot_config.py), line 222 | `coinbase_derivatives_type` still defaults to `intx_perps`. |
| [bot_config.py](../../src/gpt_trader/app/config/bot_config.py), line 439 | `COINBASE_DERIVATIVES_TYPE` env parsing still falls back to `intx_perps`. |
| [bot_config.py](../../src/gpt_trader/app/config/bot_config.py), line 375 | `derivatives_enabled` is still parsed from the INTX-named `COINBASE_ENABLE_INTX_PERPS` flag. |
| [symbols.py](../../src/gpt_trader/features/live_trade/symbols.py), line 97 | A.1 landed: `intx_perpetuals_enabled()` now fails closed unless INTX is explicitly selected. |
| [validation.py](../../src/gpt_trader/app/config/validation.py), line 19 | A.1 landed: `coinbase_derivatives_type` is now validated against `intx_perps`, `perpetuals`, and `us_futures`; enabled derivatives require a non-empty type. |

## Severity

This is a default-hygiene and naming problem, not an active incident:

- Derivatives are off by default.
- No shipped profile overrides the derivatives venue keys.
- The venue-selector helpers currently have no production callers.
- The consumed path gates whether derivative symbols are allowed at all; it does
  not currently distinguish INTX from CFM.

The remaining risk is future routing code consuming `coinbase_derivatives_type`
and inheriting a frozen INTX default.

## Decision Prep

### Repo Audit

- Tracked config defaults are conservative at the activation layer:
  `config/environments/.env.template` sets `TRADING_MODES=spot`,
  `CFM_ENABLED=0`, and `COINBASE_ENABLE_INTX_PERPS=0`.
- Tracked YAML profiles under `config/profiles/` do not set derivatives venue
  keys, INTX toggles, CFM toggles, or `TRADING_MODES`.
- `BotConfig` still uses the INTX-named `COINBASE_ENABLE_INTX_PERPS` flag as
  the source for `derivatives_enabled`, while `COINBASE_DERIVATIVES_TYPE`
  separately defaults to `intx_perps`.
- Preflight currently treats CFM and INTX as separate environment concepts:
  CFM uses `TRADING_MODES=cfm` plus `CFM_ENABLED=1`; INTX uses
  `COINBASE_ENABLE_INTX_PERPS=1`.
- The venue-selector helpers are still only directly exercised by tests. There
  are no production callers of `intx_perpetuals_enabled()` or
  `us_futures_enabled()` yet.

Before changing defaults, audit ignored local env files with `rg --hidden`
because `.env` and `.env.*` are intentionally git-ignored. Record only the
decision-relevant shape of that audit; do not commit local operator values,
credentials, or account-specific configuration.

### Current Behavior Matrix

| Scenario | Validation | Helper result |
| --- | --- | --- |
| Default config | Valid; derivatives disabled; type is `intx_perps` | US futures false, INTX false |
| `derivatives_enabled=True` with default type | Valid | US futures false, INTX true |
| `derivatives_enabled=True`, type `us_futures` | Valid | US futures true, INTX false |
| `derivatives_enabled=True`, empty type | Invalid: type must be set | US futures false, INTX false |

### A2 Precondition

A simple A2 default flip is not just a value change from `intx_perps` to
`us_futures`. Because `COINBASE_ENABLE_INTX_PERPS` is still the source for
`derivatives_enabled`, flipping only the type default would make an INTX-named
flag enable the CFM default in any future code path that consumes
`coinbase_derivatives_type`.

Before implementing A2, decide whether that temporary naming mismatch is
acceptable or whether the enablement path must also be aligned with the CFM
controls (`TRADING_MODES` / `CFM_ENABLED`). Option C avoids this silent mismatch
by requiring an explicit venue when derivatives are enabled.

### Decision Criteria

- Choose A2 only if the intended active derivatives lane is CFM and the owner
  accepts either the temporary INTX-named toggle mismatch or a scoped follow-up
  to align enablement with CFM controls.
- Choose C if avoiding implicit venue defaults is more important than minimizing
  config churn. Expect updates to `BotConfig` defaults, env parsing, validation
  tests, env docs, and generated configuration and testing artifacts.
- Do not reintroduce `COINBASE_ENABLE_DERIVATIVES` without reversing the
  documented deprecation decision.
- Do not use broker/API calls, canary commands, or live account checks to settle
  this decision. It is a config semantics decision first.

## Landed A.1

A.1 landed as a small code/test change:

- Changed `intx_perpetuals_enabled()` from fail-open to fail-closed.
- Confirmed a `us_futures` derivatives type no longer reports INTX enabled.
- Added allowed-value validation for `coinbase_derivatives_type`.
- Updated focused tests and generated testing inventory.
- Verification passed with focused tests, `uv run agent-regenerate --verify`,
  and `uv run local-ci --profile quick`.

## Remaining Options

**A2: Default to CFM.** Change the config and env fallback default from
`intx_perps` to `us_futures`. This is lower churn only after the owner accepts
or resolves the `COINBASE_ENABLE_INTX_PERPS` master-toggle mismatch described
above.

**C: Require an explicit venue.** Make missing derivatives venue invalid when
derivatives are enabled. This avoids an implicit default entirely, but it has
more config and preflight churn because current defaults assume a value. It is
the cleaner guardrail if an INTX-named toggle must not silently select CFM.

## Decision Needed

1. Choose the default model: `us_futures` by default, or explicit venue required.
2. If choosing `us_futures`, explicitly decide whether to keep or replace the
   current `COINBASE_ENABLE_INTX_PERPS` source for `derivatives_enabled`.
3. If choosing explicit-required, approve the config and preflight changes
   needed to make a missing derivatives type invalid.

## Verification For The Next Change

```bash
rg --hidden "COINBASE_DERIVATIVES_TYPE|COINBASE_ENABLE_INTX_PERPS|coinbase_derivatives_type" config/ .env
uv run pytest tests/unit/gpt_trader/features/live_trade/test_symbols_derivatives.py -v
uv run pytest tests/unit/gpt_trader/app/config/test_validation.py -v
rg "intx_perpetuals_enabled\(|us_futures_enabled\(" src/ scripts/
```
