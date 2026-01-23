# Examples

These scripts are intentionally kept out of the test suite. They are meant to be
small, runnable entrypoints that demonstrate supported patterns.

Run examples with `uv` so imports resolve correctly:

```bash
uv run python examples/composition_root_example.py
uv run python examples/verify_strategy_dev_toolkit.py
```

Notes:
- Some examples require real credentials or network access (e.g., Coinbase REST).
- Any on-disk artifacts should land in ignored directories (see `.gitignore`).
