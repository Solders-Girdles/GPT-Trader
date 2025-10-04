# Backup System Architecture

## Overview

The backup system provides comprehensive disaster recovery for the Bot V2 trading system through a modular, component-based architecture. It supports multiple backup types (FULL, INCREMENTAL, DIFFERENTIAL, SNAPSHOT, EMERGENCY) with encryption, compression, and multi-tier storage.

## Component Architecture

The system is decomposed into five specialized components with explicit dependency injection:

```
┌─────────────────────────────────────────────────────────────┐
│                      BackupManager                          │
│              (Orchestration & Coordination)                 │
│                                                             │
│  • Scheduling & automated backups                          │
│  • Locking & concurrency control                           │
│  • Retry/error policy                                      │
│  • Public API & backward compatibility                     │
└───────────┬──────────────┬──────────────┬─────────────────┘
            │              │              │
            ▼              ▼              ▼
┌──────────────┐  ┌────────────────┐  ┌──────────────┐
│ DataCollector│  │ BackupCreator  │  │BackupRestorer│
│              │  │                │  │              │
│ State        │  │ Serialization  │  │ Retrieval    │
│ Collection   │  │ Compression    │  │ Decryption   │
│ across Tiers │  │ Encryption     │  │ Decompression│
│              │  │ Storage        │  │ Restoration  │
└──────┬───────┘  └───────┬────────┘  └──────┬───────┘
       │                  │                   │
       └──────────────────┴───────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ BackupMetadataManager │
              │                       │
              │ • Metadata persistence│
              │ • History tracking    │
              │ • Backup queries      │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   BackupContext       │
              │   (Shared State)      │
              │                       │
              │ • backup_history      │
              │ • last_full_state     │
              │ • last_backup_state   │
              └───────────────────────┘
```

## Component Responsibilities

### 1. BackupManager (Orchestrator)
**File:** `src/bot_v2/state/backup/operations.py` (~400 lines)

**Purpose:** Wires components together and provides coordination

**Responsibilities:**
- Initialize all components with shared dependencies
- Schedule automated backups (full, differential, incremental)
- Manage locks and prevent concurrent backups
- Provide sync/async API translation
- Maintain backward compatibility

**Key Dependencies:**
- All other backup components
- Services (encryption, compression, transport, tier strategy)
- State manager instance

### 2. DataCollector
**File:** `src/bot_v2/state/backup/collector.py` (353 lines)

**Purpose:** Collect state data from various sources

**Responsibilities:**
- Implement collection strategies for each backup type
- Access repositories directly (HOT/WARM/COLD tiers) for 99%+ performance gain
- Fall back to StateManager when repositories unavailable
- Filter data by timestamp for incremental/differential backups
- Integrate with performance metrics

**Key Methods:**
- `collect_for_backup(backup_type, override)` - Main entry point
- `_collect_all_data()` - FULL backup collection
- `_collect_changed_data(since)` - INCREMENTAL/DIFFERENTIAL
- `_collect_snapshot_data()` - SNAPSHOT (positions, orders, portfolio)
- `_collect_critical_data()` - EMERGENCY (positions, portfolio, critical_config)

**Dependencies:**
- `state_manager` - For fallback access
- `config` - Backup configuration
- `context` - Shared baseline snapshots
- `metadata_manager` - Last backup timestamps
- `metrics` - Performance tracking

### 3. BackupCreator
**File:** `src/bot_v2/state/backup/creator.py` (218 lines)

**Purpose:** Transform collected data into stored backups

**Responsibilities:**
- Serialize backup data to JSON
- Compress payloads (configurable level)
- Encrypt with key management
- Calculate checksums
- Select storage tier (LOCAL/NETWORK/CLOUD/ARCHIVE)
- Generate comprehensive metadata
- Verify backups post-creation
- Update baseline snapshots via `BackupContext.update_baseline()`

**Key Methods:**
- `create_backup_internal(...)` - Main creation pipeline
- `_serialize_backup_data()` - JSON serialization
- `_prepare_compressed_payload()` - Compression
- `_encrypt_payload()` - Encryption with key tracking
- `_store_backup()` - Storage tier delegation
- `_verify_backup()` - Integrity verification
- `_build_backup_metadata()` - Metadata generation

**Dependencies:**
- `config` - Backup configuration
- `context` - Baseline state management
- `metadata_manager` - History tracking
- `encryption_service` - Encryption/decryption
- `compression_service` - Compression/decompression
- `transport_service` - Storage operations
- `tier_strategy` - Tier selection logic

### 4. BackupRestorer
**File:** `src/bot_v2/state/backup/restorer.py` (232 lines)

**Purpose:** Restore backups to state manager

**Responsibilities:**
- Retrieve backup data from storage
- Verify checksum integrity
- Decrypt encrypted backups
- Decompress compressed data
- Apply state using batch operations (247x faster than sequential)
- Find and restore latest backups with type filtering
- Handle restoration errors gracefully

**Key Methods:**
- `restore_from_backup_internal(backup_id, apply_state)` - Main restoration
- `restore_latest_backup(backup_type)` - Latest backup selection
- `_retrieve_backup()` - Storage retrieval
- `_restore_data_to_state()` - Batch state application with category assignment

**Dependencies:**
- `state_manager` - Target for restored state
- `config` - Configuration (compression/encryption settings)
- `context` - Track restored payloads
- `metadata_manager` - Find backups by ID
- `encryption_service` - Decryption
- `compression_service` - Decompression
- `transport_service` - Storage retrieval

### 5. BackupMetadataManager
**File:** `src/bot_v2/state/backup/metadata.py` (~150 lines)

**Purpose:** Manage backup metadata and history

**Responsibilities:**
- Save/load metadata to/from disk
- Track backup history
- Query backups by ID, type, status
- Provide statistics (total backups, storage used, etc.)
- Manage last backup timestamps for incremental logic

**Key Methods:**
- `save_metadata(metadata)` - Persist metadata
- `find_metadata(backup_id)` - Locate by ID
- `load_history()` - Load from disk
- `add_to_history()` - Add completed backup
- `get_last_backup_time()` - For incremental backups
- `get_stats()` - Statistics

## Data Flow

### Backup Creation Flow

```
1. BackupManager.create_backup(type)
   │
   ├─> Acquire lock
   │
   ├─> DataCollector.collect_for_backup(type)
   │   │
   │   ├─> Repository fast path OR StateManager fallback
   │   │
   │   └─> Returns: raw_state_data
   │
   ├─> BackupManager._normalize_state_payload(raw_data)
   │   │
   │   └─> Returns: normalized_snapshot
   │
   ├─> BackupManager._diff_state() [if INCREMENTAL/DIFFERENTIAL]
   │   │
   │   └─> Returns: diff_payload
   │
   ├─> BackupCreator.create_backup_internal(...)
   │   │
   │   ├─> Serialize → Compress → Encrypt
   │   │
   │   ├─> Store via TransportService
   │   │
   │   ├─> Generate BackupMetadata
   │   │
   │   ├─> Verify (optional)
   │   │
   │   └─> BackupContext.update_baseline()
   │
   ├─> BackupMetadataManager.add_to_history()
   │
   └─> Release lock
```

### Restoration Flow

```
1. BackupManager.restore_from_backup(id)
   │
   ├─> BackupRestorer.restore_from_backup_internal(id)
   │   │
   │   ├─> BackupMetadataManager.find_metadata(id)
   │   │
   │   ├─> TransportService.retrieve(id, tier)
   │   │
   │   ├─> Verify checksum
   │   │
   │   ├─> Decrypt (if encrypted)
   │   │
   │   ├─> Decompress (if compressed)
   │   │
   │   ├─> Parse JSON
   │   │
   │   ├─> StateManager.batch_set_state(items) [247x faster]
   │   │   │
   │   │   └─> Assign StateCategory (HOT/WARM) by key pattern
   │   │
   │   └─> BackupContext.last_restored_payload = payload
   │
   └─> Return restored state
```

## Backup Types

### FULL Backup
- Collects all state data across all patterns
- Updates both `last_full_state` and `last_backup_state`
- Used as baseline for DIFFERENTIAL backups
- Default tier: LOCAL or NETWORK

### INCREMENTAL Backup
- Collects only changes since last backup (any type)
- Uses timestamp filtering
- Updates `last_backup_state` only
- Fastest backup type
- Default tier: LOCAL

### DIFFERENTIAL Backup
- Collects changes since last FULL backup
- Uses timestamp filtering from `last_full_backup`
- Updates `last_backup_state` only
- Balances size and restore complexity
- Default tier: LOCAL or NETWORK

### SNAPSHOT Backup
- Collects current positions, orders, portfolio, performance
- Lightweight, high-frequency capable
- Updates both `last_full_state` and `last_backup_state`
- Default tier: LOCAL

### EMERGENCY Backup
- Collects only critical data (positions, portfolio, critical_config)
- Minimal size for rapid backup
- Used during failures or high-risk operations
- Default tier: ARCHIVE (for compliance)

## State Management via BackupContext

The `BackupContext` dataclass centralizes mutable state mutations to prevent stale references and ensure consistency:

```python
@dataclass
class BackupContext:
    backup_history: list[BackupMetadata]
    backup_metadata: dict[str, BackupMetadata]
    last_full_backup: datetime | None
    last_differential_backup: datetime | None
    last_full_state: dict[str, Any] | None
    last_backup_state: dict[str, Any] | None
    last_restored_payload: dict[str, Any] | None

    def update_baseline(self, backup_type: BackupType, snapshot: dict[str, Any]):
        """Centralized baseline updates for FULL/SNAPSHOT/INCREMENTAL."""
        self.last_backup_state = snapshot
        if backup_type in {BackupType.FULL, BackupType.SNAPSHOT}:
            self.last_full_state = snapshot
```

**Benefits:**
- Single source of truth for baseline snapshots
- Prevents accidental mutations during refactoring
- Clear semantics for which backups update which baselines
- Property descriptors in BackupManager provide backward compatibility

## Performance Optimizations

### Repository Fast Path (DataCollector)
- **Gain:** 99%+ faster than StateManager
- **Method:** Direct access to Redis/PostgreSQL/S3 repositories
- **Fallback:** StateManager when repositories unavailable or not async

### Batch State Operations (BackupRestorer)
- **Gain:** 247x faster than sequential `set_state` calls
- **Method:** `StateManager.batch_set_state({key: (value, category)})`
- **Fallback:** Sequential `set_state` when batch unavailable

### Compression & Encryption
- **Compression:** Configurable level (default: 6)
- **Encryption:** AES with key management
- **Typical ratio:** 40-60% size reduction

## Storage Tiers

```
LOCAL     → Fast access, ephemeral (local disk)
NETWORK   → Shared storage, medium durability (NAS/NFS)
CLOUD     → S3, high durability, slower access
ARCHIVE   → Long-term retention, compliance (Glacier)
```

**Tier Strategy:**
- FULL → NETWORK or CLOUD
- INCREMENTAL → LOCAL
- DIFFERENTIAL → NETWORK
- SNAPSHOT → LOCAL
- EMERGENCY → ARCHIVE

## Backward Compatibility

BackupManager maintains backward compatibility through:

1. **Property Descriptors:** Expose `BackupContext` attributes as instance attributes
   ```python
   @property
   def _last_full_state(self) -> dict[str, Any] | None:
       return self.context.last_full_state
   ```

2. **Metadata Wrappers:** Delegate to `BackupMetadataManager`
   ```python
   def _save_backup_metadata(self, metadata):
       self.metadata_manager.save_metadata(metadata)
   ```

3. **Sync/Async API:** `restore_from_backup()` adapts to caller context
   - Sync: Runs `asyncio.run()` when no event loop active
   - Async: Returns coroutine when loop detected

## Testing Strategy

### Unit Tests (52 tests)
- **test_creator.py:** Creation pipeline, encryption, compression, verification
- **test_data_collector.py:** Collection strategies, repository fallback, timestamp filtering
- **test_restorer.py:** Restoration workflow, batch operations, payload validation

**Benefits:**
- Fast execution (0.09s)
- Independent component verification
- Enable future refactoring

### Integration Tests (182 tests)
- **test_backup_operations.py:** End-to-end workflows, scheduling, concurrency
- **test_backup_operational_scenarios.py:** Error handling, verification, metrics
- **test_backup_smoke.py:** Basic functionality
- **test_backup_storage.py:** Storage tier operations

## Extension Points

### Adding a New Backup Type
1. Add enum to `BackupType` in `models.py`
2. Implement collection strategy in `DataCollector.collect_for_backup()`
3. Define tier strategy in `TierStrategy.determine_tier()`
4. Add retention policy in `RetentionService.get_retention_days()`

### Adding a New Storage Tier
1. Add enum to `StorageTier` in `models.py`
2. Implement transport in `TransportService` in `services/transport.py`
3. Update `TierStrategy.determine_tier()` logic

### Custom Data Collectors
Inject custom `DataCollector` subclass into `BackupManager.__init__()`:
```python
custom_collector = CustomDataCollector(...)
manager.data_collector = custom_collector
```

## Key Design Patterns

1. **Dependency Injection:** All components receive dependencies explicitly
2. **Strategy Pattern:** Different collection/tier strategies per backup type
3. **Repository Pattern:** Direct tier access with StateManager fallback
4. **Template Method:** `BackupCreator.create_backup_internal()` pipeline
5. **Facade:** `BackupManager` simplifies complex subsystem
6. **Context Object:** `BackupContext` encapsulates shared mutable state

## Future Enhancements

1. **Incremental Restoration:** Apply only changed data from incremental backups
2. **Compression Algorithms:** Support multiple algorithms (gzip, zstd, lz4)
3. **Encryption Rotation:** Automatic key rotation with re-encryption
4. **Parallel Backup:** Collect from multiple tiers simultaneously
5. **Deduplication:** Content-addressed storage for space efficiency
6. **Backup Validation:** Periodic test restores to verify integrity

## References

- **Code:**
  - `src/bot_v2/state/backup/operations.py` - BackupManager
  - `src/bot_v2/state/backup/collector.py` - DataCollector
  - `src/bot_v2/state/backup/creator.py` - BackupCreator
  - `src/bot_v2/state/backup/restorer.py` - BackupRestorer
  - `src/bot_v2/state/backup/metadata.py` - BackupMetadataManager
  - `src/bot_v2/state/backup/models.py` - Data models
  - `src/bot_v2/state/backup/services/` - Encryption, compression, transport, etc.

- **Tests:**
  - `tests/unit/bot_v2/state/backup/test_creator.py` - Creator unit tests
  - `tests/unit/bot_v2/state/backup/test_data_collector.py` - Collector unit tests
  - `tests/unit/bot_v2/state/backup/test_restorer.py` - Restorer unit tests
  - `tests/unit/bot_v2/state/backup/test_backup_operations.py` - Integration tests

- **Architecture Docs:**
  - `docs/architecture/EXECUTION_COORDINATOR_REFACTOR.md` - Related state management
  - `docs/archive/refactoring-2025-q1/REFACTORING_PHASE_0_STATUS.md` - Project evolution
