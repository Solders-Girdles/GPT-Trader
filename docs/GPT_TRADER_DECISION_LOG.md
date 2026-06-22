# GPT-Trader Decision Log

Durable product and engineering direction for GPT-Trader. Use this log for
decisions that should outlive chat, PR receipts, and local branch state.

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
