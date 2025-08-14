# Memory Profiling Report

Generated: 2025-08-12 07:16:16

## Summary
- Initial Memory: 70.8 MB
- Final Memory: 250.0 MB
- Total Change: +179.2 MB
- Peak Memory: 250.0 MB

## Snapshots

### initial
- Memory: 70.8 MB
- Objects: 52,648
- Top Types: dict, function, type

### after_dataframe
- Memory: 147.9 MB
- Objects: 52,206
- Top Types: DataFrame, dict, function

### after_optimization
- Memory: 211.0 MB
- Objects: 53,271
- Top Types: Series, DataFrame, dict

### iteration_0
- Memory: 218.8 MB
- Objects: 53,275
- Top Types: Series, DataFrame, dict

### iteration_1
- Memory: 226.4 MB
- Objects: 53,278
- Top Types: Series, DataFrame, dict

### iteration_2
- Memory: 234.7 MB
- Objects: 53,281
- Top Types: Series, DataFrame, dict

### iteration_3
- Memory: 242.3 MB
- Objects: 53,284
- Top Types: Series, DataFrame, dict

### iteration_4
- Memory: 250.0 MB
- Objects: 53,287
- Top Types: Series, DataFrame, dict

## Potential Memory Leaks
- {'type': 'consistent_growth', 'total_growth_mb': 179.21875, 'snapshots': 8, 'severity': 'high'}

## Recommendations
- ⚠️ Potential memory leak detected: 179.2 MB growth