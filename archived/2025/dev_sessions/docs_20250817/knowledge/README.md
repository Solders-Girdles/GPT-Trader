# V2 Active Knowledge Layer

Current workflow guides and reference documents for effective V2 vertical slice development.

## V2 Workflow & Agent Guides
- **AGENT_WORKFLOW.md** - V2 slice patterns for using Claude Code agents
- **AGENT_DELEGATION_GUIDE.md** - How to delegate V2 slice tasks to specialized agents  
- **TASK_TEMPLATES.md** - V2 slice templates for clear agent delegation
- **WORKFLOW.md** - Core V2 development workflow
- **WORKFLOW_EVALUATION.md** - Results from V2 workflow testing
- **WORKFLOW_STRESS_TEST_RESULTS.md** - V2 stress test findings

## V2 Reference Documents
- **DIAGNOSTICS.md** - V2 slice health check commands
- **IMPORTS.md** - V2 import patterns (local-only, isolation-preserving)
- **TEST_MAP.json** - Which V2 tests validate what slices
- **DEPENDENCIES.json** - V2 slice isolation verification

## V2 Policies & Maintenance
- **NO_NEW_DOCS_POLICY.md** - Policy against creating excessive V2 documentation
- **KNOWLEDGE_LAYER_MAINTENANCE.md** - How to keep V2 knowledge current

## V2 Usage Context

These are **active, proven V2 guides** - not historical documents. Use them to:
- Structure V2 slice agent delegation effectively
- Follow established V2 isolation workflows  
- Troubleshoot common V2 slice issues
- Maintain V2 system knowledge
- Preserve slice independence

## V2 Architecture Focus

All guides updated for:
- **Vertical slice architecture** (src/bot_v2/features/)
- **Complete isolation principles** (no cross-slice imports)
- **Local implementations** (duplicate rather than share)
- **Independent testing** (poetry run python src/bot_v2/test_[slice].py)
- **Ultraclean repository** (230M+ cleanup completed)

## Quick V2 Navigation

| Need | File | Purpose |
|------|------|---------|
| Fix slice test | TASK_TEMPLATES.md | V2 debugging patterns |
| Delegate slice work | AGENT_DELEGATION_GUIDE.md | V2 isolation-aware delegation |
| Follow V2 workflow | AGENT_WORKFLOW.md | V2 slice best practices |
| Maintain knowledge | KNOWLEDGE_LAYER_MAINTENANCE.md | V2 state management |

All knowledge layer files are now V2-native and reflect the current ultraclean repository structure.