Dev Iteration Test Scripts (Archived)

This folder contains earlier, developer-oriented test scripts and overlapping integration checks that have been superseded by the active test suite under `tests/`.

What moved here and where to look instead:

- src/test_coinbase_api.py -> Covered by:
  - tests/integration/real_api/test_coinbase_connectivity.py
  - tests/integration/test_cdp_comprehensive.py

- src/bot_v2/scripts/test_all_slices.py -> Covered by:
  - tests/integration/bot_v2/test_slice_isolation.py
  - tests/integration/bot_v2/test_vertical_slice.py

- src/bot_v2/test_api_integration.py -> Covered by:
  - tests/integration/bot_v2/test_orchestration.py
  - tests/integration/bot_v2/test_workflows.py

- src/bot_v2/test_workflow_adapter.py -> Covered by:
  - tests/integration/bot_v2/test_workflows.py

- src/bot_v2/test_workflow_engine_fixes.py -> Covered by:
  - tests/integration/bot_v2/test_workflows.py
  - tests/integration/bot_v2/test_orchestration.py

- tests/integration/test_cdp_connection.py (obsolete iteration) -> Consolidated into:
  - tests/integration/test_cdp_comprehensive.py

- tests/integration/test_official_cdp.py (overlapping) -> Consolidated into:
  - tests/integration/test_cdp_comprehensive.py (primary)
  - scripts/test_official_sdk.py (manual check retained under scripts/)

Notes

- Pytest discovery excludes `archived/` by default (see `pytest.ini`).
- If you need any of these for reference, keep them here. If they must be revived, port them into an appropriate file under `tests/` and align them with current interfaces and markers.

