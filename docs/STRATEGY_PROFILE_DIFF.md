# Strategy Profile Diff

Operators can now compare the StrategyProfile loaded from disk (baseline) against the active strategy profile emitted by a live runtime.
The diff command produces deterministic entries for **changed**, **unchanged**, and **missing** keys and provides both human-readable and machine-readable outputs.

## Usage

```bash
gpt-trader strategy profile-diff \
  --baseline config/strategy_profiles/btc_momentum.json \
  --profile dev \
  --runtime-root /path/to/repo \
  --format text
```

By default the runtime profile is read from `runtime_data/<profile>/strategy_profile.json`. Provide `--runtime-profile` to point to a different file.

Use `--format json` to receive structured output:

```
{
  "baseline_path": "...",
  "runtime_profile_path": "...",
  "diff": [
    {"path": "risk.max_position_size", "status": "changed", ...},
    {"path": "signals", "status": "unchanged", ...},
  ]
}
```

## Diff semantics

- **changed** – values differ between baseline and runtime (includes runtime-only keys).
- **missing** – the runtime profile lacks a baseline key.
- **unchanged** – values match exactly (lists/dicts are normalized before comparison).
- Entries are emitted in lexicographic order for automation-friendly parsing.

## Ignored fields

- `created_at` is ignored by default because timestamps differ between saved and live profiles.
- Use `--ignore <field>` to omit additional noisy fields (pass the flag multiple times).

## Notes

- Baseline files can be JSON or YAML (PyYAML required for YAML).
- Runtime snapshots are expected to be valid JSON; if they are not present, the command reports `FILE_NOT_FOUND`.
