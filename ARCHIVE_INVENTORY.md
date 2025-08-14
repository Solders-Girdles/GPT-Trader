# Archive Inventory

**Date**: August 14, 2025
**Total Size**: 4.9GB
**Recommendation**: Move to external storage or delete

## Contents

### 1. claudia_20250812_151036 (4.9GB) ⚠️
- **What**: Complete Tauri/Rust application (separate project)
- **Why Archived**: Unrelated to GPT-Trader
- **Size Issue**: Contains 4.6GB of Rust build artifacts in src-tauri/target/
- **Recommendation**: DELETE or move to separate repository
- **Contains**: Rust/Tauri desktop app with node_modules and build artifacts

### 2. intelligence_20250812_152245 (588KB)
- **What**: Deprecated ML intelligence module
- **Why Archived**: Functionality migrated to src/bot/ml/
- **Files**: 19 Python files for pattern recognition, anomaly detection
- **Recommendation**: Keep for reference (small size)

### 3. cli_old_20250812_151945 (344KB)
- **What**: Previous CLI implementation
- **Why Archived**: Replaced with unified CLI in src/bot/cli/
- **Files**: Old command structure and utilities
- **Recommendation**: Keep for reference (small size)

### 4. monitor_20250812_152804 (184KB)
- **What**: Old monitoring implementation
- **Why Archived**: Replaced with new monitoring system
- **Files**: Legacy monitoring classes
- **Recommendation**: Keep for reference (small size)

### 5. ml_20250812_152245 (84KB)
- **What**: Original ML module
- **Why Archived**: Integrated into main ML pipeline
- **Files**: Early ML implementations
- **Recommendation**: Keep for reference (small size)

### 6. k8s_20250812_151114 (84KB)
- **What**: Kubernetes deployment configs
- **Why Archived**: Not currently using K8s deployment
- **Files**: YAML configs for K8s deployment
- **Recommendation**: Keep if planning K8s deployment

### 7. meta_learning_20250812_152245 (80KB)
- **What**: Meta-learning experiments
- **Why Archived**: Consolidated into auto_retraining.py
- **Files**: 4 Python files
- **Recommendation**: Keep for reference (small size)

### 8. monitoring_20250812_152804 (64KB)
- **What**: Another monitoring implementation
- **Why Archived**: Duplicate/old monitoring code
- **Files**: Health checker, metrics collector
- **Recommendation**: Can be deleted (duplicated)

### 9. scaling_20250812_152245 (28KB)
- **What**: Premature scaling optimization
- **Why Archived**: Not needed yet
- **Files**: Distributed computing attempts
- **Recommendation**: Delete (premature optimization)

### 10. distributed_20250812_152245 (28KB)
- **What**: Distributed computing module
- **Why Archived**: Premature optimization
- **Files**: Distributed training attempts
- **Recommendation**: Delete (premature optimization)

### 11. realtime_20250812_152936 (24KB)
- **What**: Early real-time data feed
- **Why Archived**: Replaced with current implementation
- **Files**: WebSocket experiments
- **Recommendation**: Delete (superseded)

### 12. knowledge_20250812_152245 (16KB)
- **What**: Knowledge base module
- **Why Archived**: Scope reduction
- **Files**: 1 Python file
- **Recommendation**: Delete (out of scope)

## Action Plan

### Immediate Actions (4.9GB savings):
1. **Delete or move claudia_20250812_151036** - This alone saves 4.9GB
2. This is a completely separate project and should not be in GPT-Trader

### Optional Cleanup (minor savings):
3. Delete duplicated/superseded archives:
   - monitoring_20250812_152804 (duplicate)
   - scaling_20250812_152245 (premature)
   - distributed_20250812_152245 (premature)
   - realtime_20250812_152936 (superseded)
   - knowledge_20250812_152245 (out of scope)

### Keep for Reference (< 1MB total):
4. Keep these small, potentially useful archives:
   - intelligence_20250812_152245
   - cli_old_20250812_151945
   - monitor_20250812_152804
   - ml_20250812_152245
   - k8s_20250812_151114
   - meta_learning_20250812_152245

## Summary
- **Critical Issue**: 4.9GB Claudia project should be removed immediately
- **After Claudia Removal**: Only 316KB of legitimate archives remain
- **Final Recommendation**: Remove Claudia, keep the rest (they're tiny)
