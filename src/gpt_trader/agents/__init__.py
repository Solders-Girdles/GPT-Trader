"""Agent tooling entry points.

This module provides CLI entry points for agent tools, enabling them
to be invoked as first-class commands via `uv run agent-*`.

Available commands:
    agent-check      - Run quality gate (lint, types, tests)
    agent-impact     - Analyze change impact and suggest tests
    agent-map        - Generate dependency graph
    agent-tests      - Generate test inventory
    agent-risk       - Query risk configuration
    agent-naming     - Check naming standards
    agent-regenerate - Regenerate all context files
"""
