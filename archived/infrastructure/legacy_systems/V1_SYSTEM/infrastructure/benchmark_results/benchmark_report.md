# Serialization Format Benchmark Report

Generated: 2025-08-12 07:13:06

## Executive Summary
- **Fastest Read**: pickle
- **Fastest Write**: pickle
- **Smallest Size**: joblib
- **Best Compression**: joblib
- **Best Throughput**: pickle

## Recommendations by Use Case
- **Real Time Trading**: feather
- **Historical Storage**: parquet
- **Data Archival**: parquet
- **Temporary Cache**: feather
- **Data Exchange**: csv

## Detailed Performance Metrics
```

READ_TIME:
  csv: 0.527
  joblib: 0.108
  json: 0.823
  pickle: 0.006

WRITE_TIME:
  csv: 2.664
  joblib: 0.957
  json: 0.343
  pickle: 0.020

TOTAL_TIME:
  csv: 3.192
  joblib: 1.066
  json: 1.166
  pickle: 0.026

FILE_SIZE_MB:
  csv: 106.303
  joblib: 41.489
  json: 76.145
  pickle: 44.271

COMPRESSION_RATIO:
  csv: 0.426
  joblib: 1.089
  json: 0.567
  pickle: 0.996

THROUGHPUT_MB_S:
  csv: 11.549
  joblib: 39.782
  json: 38.489
  pickle: 1265.078
```

## Performance by Data Size

### 1,000 rows
Best performer: **pickle** (0.000s total)

### 10,000 rows
Best performer: **pickle** (0.001s total)

### 100,000 rows
Best performer: **pickle** (0.021s total)

### 500,000 rows
Best performer: **pickle** (0.082s total)
