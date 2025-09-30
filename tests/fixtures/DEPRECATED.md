Legacy JSON fixtures were unused by the active test suite and have been removed.

Notes:
- Search across the repository found no references in active tests or code.
- If a test needs static API samples in the future, place them next to the test
  (e.g., `tests/unit/.../fixtures/`) or model them via small Python helpers
  (see `tests/fixtures/behavioral/`). Avoid central, shared JSON blobs.

Status:
- Removed: `tests/fixtures/coinbase/*.json`, `tests/fixtures/mock_data.json`.
- Retained: Python helpers under `tests/fixtures/behavioral/` used by PnL and
  behavioral tests.

Updates 2025-03:
- Removed test enforcement for deprecated module aliases (`tests/unit/bot_v2/test_removed_aliases.py`). If a compatibility layer reappears, document it here and add a targeted lint instead of a broad test.
