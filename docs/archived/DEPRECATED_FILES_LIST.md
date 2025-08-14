# Deprecated Documentation Files

This directory contains documentation files that have been archived as they are no longer current or have been superseded by newer, more comprehensive documentation.

## Files Archived on January 2025

### Reason: Superseded by PHASE3_4_COMPLETION_SUMMARY.md
- `PHASE3_COMPLETION_SUMMARY.md` - Individual Phase 3 summary (superseded)
- `PHASE5_COMPLETION_SUMMARY.md` - References non-existent Phase 5

### Reason: Outdated Planning/Implementation Guides  
- `PHASE1_IMPLEMENTATION_GUIDE.md` - Implementation guide for completed phase
- `PHASE1_RISK_MITIGATION.md` - Risk mitigation for completed phase
- `NEXT_STEPS_ROADMAP.md` - Contains outdated roadmap information
- `ML_FOUNDATION_IMPROVEMENT_PLAN.md` - Outdated improvement plan

### Reason: Redundant Status/Summary Documents
- `DOCUMENTATION_CLEANUP_SUMMARY.md` - Meta-documentation about previous cleanup
- `SYSTEMATIC_IMPROVEMENTS_SUMMARY.md` - Redundant with other status documents

## Current Active Documentation Structure

See `/docs/README.md` for the current, organized documentation structure.

All information from archived files has been consolidated into the current documentation set.

---

# Code Files Archive (SOT Program Phase 1)

**Last Updated**: 2025-08-14  
**SoT Tasks Completed**: SOT-010, SOT-011

## Archived Module Directories (Previously Archived)

### intelligence_20250812_152245/
- **Location**: `../archived/intelligence_20250812_152245/`
- **Archive Date**: 2025-08-12
- **Reason**: Deprecated intelligence module - functionality migrated to `src/bot/ml/` and `src/bot/monitoring/`
- **Replacement**: 
  - ML functionality → `src/bot/ml/integrated_pipeline.py`
  - Monitoring functionality → `src/bot/monitoring/`
- **Files Count**: 19 Python files

### meta_learning_20250812_152245/
- **Location**: `../archived/meta_learning_20250812_152245/`
- **Archive Date**: 2025-08-12
- **Reason**: Deprecated meta-learning module - functionality consolidated into core ML pipeline
- **Replacement**: `src/bot/ml/auto_retraining.py` and `src/bot/ml/integrated_pipeline.py`
- **Files Count**: 4 Python files

### distributed_20250812_152245/ & scaling_20250812_152245/
- **Location**: `../archived/distributed_20250812_152245/` & `../archived/scaling_20250812_152245/`
- **Archive Date**: 2025-08-12
- **Reason**: Premature optimization modules removed
- **Replacement**: Manual scaling in `src/bot/core/deployment.py`
- **Files Count**: 4 Python files total

### knowledge_20250812_152245/
- **Location**: `../archived/knowledge_20250812_152245/`
- **Archive Date**: 2025-08-12
- **Reason**: Scope reduction - knowledge embedded in strategy files
- **Replacement**: Strategy-specific knowledge in individual files
- **Files Count**: 1 Python file

## Files with Deprecated Imports (SOT-010)

### deprecated_imports_20250814/
- **Archive Date**: 2025-08-14
- **Reason**: Files containing imports from archived/non-existent modules
- **SoT Task**: SOT-010

#### Archived Files:
1. **strategy_selector.py**
   - **Original**: `src/bot/live/strategy_selector.py`
   - **Issue**: Imports from `bot.meta_learning.regime_detection`, `bot.meta_learning.temporal_adaptation`
   - **Replacement**: Strategy selection in `src/bot/ml/auto_retraining.py`

2. **production_orchestrator.py**
   - **Original**: `src/bot/live/production_orchestrator.py`
   - **Issue**: Imports from `bot.intelligence.*`, `bot.meta_learning.*`
   - **Replacement**: Production orchestration in `src/bot/monitoring/ops_dashboard.py`

3. **enhanced_evolution_with_knowledge.py**
   - **Original**: `src/bot/optimization/enhanced_evolution_with_knowledge.py`
   - **Issue**: References deprecated knowledge module
   - **Replacement**: Evolution logic in `src/bot/strategy/validation_pipeline.py`

## Migration Notes

- **Import Updates**: All deprecated module imports have been removed from active codebase
- **Functionality Preserved**: Core functionality migrated to appropriate active modules
- **Documentation**: References to archived modules updated in Phase 2 (SOT-020 to SOT-023)