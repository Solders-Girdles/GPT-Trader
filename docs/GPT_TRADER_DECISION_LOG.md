# GPT-Trader Decision Log

---
status: current
last-updated: 2026-06-24
---

Durable product and engineering direction for GPT-Trader. Use this log for
decisions that should outlive chat, PR receipts, and local branch state.

## 2026-06-11 — Accept Pre-Migration Direction And Trade-Idea Rails

- **Status:** accepted direction; Stage 0 trade-idea rails implemented where
  noted below
- **Owner:** RJ
- **Decision / direction:** Use the
  [Pre-Migration Decision Framework](PRE_MIGRATION_DECISION_FRAMEWORK.md) as
  the execution-migration gate. The accepted path is approval-gated execution
  (`human_approved_execution`) as the validation phase, with
  `bounded_autonomy` only as a later destination. Coinbase spot plus CFM futures
  research are the current Coinbase-only scope; Robinhood, options execution,
  INTX-only work, and other venues stay out of scope unless a later decision
  packet explicitly reopens them.
- **Implemented evidence:** The trade-idea slice implements broker-neutral
  records (`src/gpt_trader/features/trade_ideas/models.py`), the approval
  workflow (`src/gpt_trader/features/trade_ideas/workflow.py`), append-only
  audit events (`src/gpt_trader/features/trade_ideas/audit.py`), seeded
  versioned risk budgets (`src/gpt_trader/features/trade_ideas/budget.py`),
  eligibility and approval policy (`src/gpt_trader/features/trade_ideas/eligibility.py`,
  `src/gpt_trader/features/trade_ideas/policy.py`), and operator lifecycle
  controls (`src/gpt_trader/features/trade_ideas/service.py`).
- **Safety boundary:** This decision does not authorize real broker/API calls,
  live trading commands, production preflight, canary operations, credential
  reads, money movement, or order submission. Any non-manual execution lane
  still needs a decision packet or approved runbook that names the lane,
  constraints, verification boundary, and rollback/kill-switch expectations.
- **Still pending:** Official venue/API/account capability review for any
  non-manual execution, product-specific migration envelopes, bounded-autonomy
  strategy envelopes, kill-switch drills, and any policy change that would
  allow submission without explicit approval.
- **Rejected alternatives:** Do not migrate unrestricted spot-only automation
  into the new shape. Do not use existing broker adapters or live profiles as
  proof that a product or venue should be automated.

## 2026-06-22 — Continue Trade-Ideas CLI As The Active Discovery Lane

- **Status:** accepted direction, implementation still WIP
- **Owner:** Claw
- **Reviewer:** Edison for decision quality; Hermes for explicit evidence receipts
- **Decision / direction:** Treat `codex/trade-ideas-cli` as the active viable
  GPT-Trader discovery lane. It is the CLI door into the existing
  human-approved trade-idea workflow, not an execution or broker-action lane.
- **Evidence:** The current WIP matches `docs/specs/TRADE_IDEA_CLI_SPEC.md`
  and `docs/specs/TRADE_IDEA_INTERFACES_DESIGN_NOTES.md`; `gpt-trader ideas
  --help` exposes the expected command group; focused trade-ideas domain tests
  passed (`136 passed`); touched files passed `ruff`; import boundaries passed.
- **Safety boundary:** No broker API calls, account actions, credential reads,
  live trading, autonomous order submission, or execution enablement. The v1
  product thesis remains AI-assisted decision support with human-approved
  execution.
- **Trading/account/credential/runtime impact:** None from the direction
  decision. The active WIP is local code only and still needs CLI tests before
  package/PR consideration.
- **Stop condition:** Stop treating the branch as merely speculative once the
  CLI test plan exists and the focused quality gates pass; otherwise park it
  with the exact failing tests or scope gap.
- **Next bounded experiment:** Add focused CLI tests for the `ideas` command
  group, starting with propose/list/show/approve/audit verify against a
  `tmp_path` ideas root. Run black on touched files before any package step.
- **Rejected alternatives:** Do not publish stale local branches. Do not use
  OpenClaw's anchor as GPT-Trader's roadmap. Do not promote momentum receipts
  into product direction without a project-native decision entry.
