"""
Characterization Tests for PerpsBot

PURPOSE: Freeze current behavior before refactoring
STATUS: Phase 0 - Expand collaboratively
RULE: These tests document WHAT happens, not HOW it should work

⚠️ These tests may be slow, ugly, or use real resources - that's OK.
⚠️ Goal: Catch ANY behavioral change during refactoring.
⚠️ Add assertions as you discover behavior - this is a living test suite.

Reference: docs/architecture/perps_bot_dependencies.md

Module Structure:
- test_initialization.py: Service creation and wiring
- test_update_marks.py: Mark price updates and window trimming
- test_properties.py: Property accessors (broker, risk_manager, etc.)
- test_delegation.py: Service delegation patterns
- test_streaming.py: Streaming service integration and lock sharing
- test_full_cycle.py: End-to-end lifecycle tests
- test_builder.py: Builder pattern and construction
- test_strategy_services.py: Strategy orchestrator extracted services
"""
