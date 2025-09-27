# 🧹 Final Cleanup Plan - Complete V2 Purification

## Executive Summary
Despite our V2 migration, we still have **1GB+ of unnecessary files** including an entire separate GitHub project (!), V1 infrastructure, and extensive cruft.

## 🚨 Priority 1: Critical Issues

### 1. Remove Separate Project
```bash
# This is a completely different GitHub project!
awesome-claude-agents/  (1.1M)
→ Action: DELETE entirely (it's not part of GPT-Trader)
```

### 2. Virtual Environment
```bash
.venv/  (978M!)
→ Action: Ensure it's in .gitignore (shouldn't be in repo)
```

## 📦 Priority 2: Archive V1 Infrastructure

### Move to `archived/v1_infrastructure_20250817/`:
```
deploy/         - Docker, Postgres configs
monitoring/     - Grafana, Prometheus setup  
config/         - V1 configuration files
benchmarks/     - V1 benchmark code
benchmark_results/ - V1 benchmark results
data/           - Experimental/historical data
logs/           - Runtime logs from V1
```

## 🗂️ Priority 3: Clean Remaining V1 Code

### Move to `archived/v1_final_cleanup_20250817/`:
```
tests/          - Remaining V1 unit tests
scripts/        - Shell scripts (if V1-related)
demos/          - Empty directory (just remove)
```

## 📚 Priority 4: Documentation Audit

### Review all 81 files in docs/:
- Keep: V2-relevant guides, README.md
- Archive: V1-specific documentation
- Update: Files with mixed V1/V2 content

## 🎯 Expected Results

### Before Cleanup:
```
Repository Size: 1.1G+
Directories: 20+
V1 References: Hidden in various places
Separate Projects: 1 (awesome-claude-agents)
```

### After Cleanup:
```
Repository Size: ~50M (excluding .venv)
Directories: 8-10 core directories
V1 References: 0 (100% archived)
Separate Projects: 0
```

## 📊 Cleanup Impact

### Storage Freed:
- awesome-claude-agents: 1.1M
- logs: 660K  
- docs (V1 portions): ~500K
- tests: 408K
- data: 344K
- Total: ~3M+ direct, much more with git cleanup

### Clarity Gained:
- No confusion from separate projects
- No V1 infrastructure temptation
- Clean root directory
- Clear V2-only structure

## 🔧 Implementation Steps

1. **Remove awesome-claude-agents/** completely
2. **Archive V1 infrastructure** (deploy/, monitoring/, config/)
3. **Archive remaining V1 code** (tests/, data/, logs/)
4. **Audit docs/** for V1 content
5. **Remove empty directories** (demos/)
6. **Update .gitignore** for .venv/
7. **Final verification scan**

## ⚠️ Backup Reminder
Create a backup branch before this cleanup:
```bash
git checkout -b pre-final-cleanup-backup
git checkout main
```

## 🎯 End Goal
A **pristine V2-only repository** with:
- Zero V1 code or references
- No unrelated projects
- Clean, minimal root structure
- Clear navigation for AI agents
- Optimized for development velocity

**Estimated Completion Time**: 30-45 minutes
**Risk Level**: Low (everything important is already in src/bot_v2/)
**Benefit**: Massive - true 100% V2 purity