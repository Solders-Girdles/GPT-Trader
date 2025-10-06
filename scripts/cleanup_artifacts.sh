#!/usr/bin/env bash
#
# cleanup_artifacts.sh - Operational directory cleanup with safety features
#
# Purpose: Implements retention policy for backups/, logs/, cache/, data_storage/
# Safety: Dry-run default, backup-before-delete, comprehensive logging
#
# Usage:
#   ./scripts/cleanup_artifacts.sh --dry-run                    # Default: show what would be deleted
#   ./scripts/cleanup_artifacts.sh --confirm                    # Interactive confirmation
#   ./scripts/cleanup_artifacts.sh --auto --age-threshold 30    # Automated with age limit
#   ./scripts/cleanup_artifacts.sh --backup-first               # Archive before delete
#
# See: docs/ops/retention_policy.md

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AUDIT_LOG="${PROJECT_ROOT}/logs/cleanup_audit.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default settings
DRY_RUN=true
CONFIRM=false
AUTO=false
BACKUP_FIRST=false
AGE_THRESHOLD_DAYS=30
EXCLUDE_PATTERNS=""
VERBOSE=false

# Retention periods (days) per retention_policy.md
BACKUPS_RETENTION_DAYS=30
LOGS_RETENTION_DAYS=14
CACHE_RETENTION_DAYS=7
DATA_STORAGE_RETENTION_DAYS=90

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Utility Functions
# ============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Console output with color
    case "$level" in
        INFO)  echo -e "${BLUE}[INFO]${NC} $message" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC} $message" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $message" >&2 ;;
        SUCCESS) echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
    esac

    # Append to audit log (JSON format)
    mkdir -p "$(dirname "$AUDIT_LOG")"
    cat >> "$AUDIT_LOG" <<EOF
{"timestamp":"$timestamp","level":"$level","message":"$message"}
EOF
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Cleanup operational directories per retention policy.

OPTIONS:
    --dry-run                Show what would be deleted (default)
    --confirm                Interactive confirmation before each deletion
    --auto                   Automated mode (no prompts, requires --age-threshold)
    --backup-first           Create archive before deletion
    --age-threshold DAYS     Only delete files older than DAYS (default: 30)
    --exclude-patterns STR   Comma-separated patterns to preserve (e.g., "*.critical,monthly_*")
    --verbose                Verbose output
    -h, --help               Show this help message

EXAMPLES:
    # Dry-run (safe exploration)
    $0 --dry-run

    # Interactive cleanup
    $0 --confirm

    # Automated cleanup with backup
    $0 --auto --age-threshold 30 --backup-first

    # Preserve specific patterns
    $0 --confirm --exclude-patterns "monthly_*,*.critical"

See docs/ops/retention_policy.md for policy details.
EOF
    exit 0
}

confirm_action() {
    local message="$1"
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "[DRY-RUN] Would: $message"
        return 1
    elif [[ "$CONFIRM" == "true" ]]; then
        read -p "$(echo -e "${YELLOW}?${NC} $message [y/N]: ")" -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            return 0
        else
            log INFO "Skipped: $message"
            return 1
        fi
    elif [[ "$AUTO" == "true" ]]; then
        log INFO "Auto-executing: $message"
        return 0
    fi
    return 1
}

should_exclude() {
    local file="$1"
    if [[ -z "$EXCLUDE_PATTERNS" ]]; then
        return 1
    fi

    IFS=',' read -ra patterns <<< "$EXCLUDE_PATTERNS"
    for pattern in "${patterns[@]}"; do
        if [[ "$file" == $pattern ]]; then
            log INFO "Excluded by pattern: $file (matches $pattern)"
            return 0
        fi
    done
    return 1
}

get_file_age_days() {
    local file="$1"
    if [[ ! -e "$file" ]]; then
        echo "0"
        return
    fi

    # macOS vs Linux stat compatibility
    if [[ "$(uname)" == "Darwin" ]]; then
        local mtime=$(stat -f %m "$file")
    else
        local mtime=$(stat -c %Y "$file")
    fi

    local now=$(date +%s)
    local age_seconds=$((now - mtime))
    local age_days=$((age_seconds / 86400))
    echo "$age_days"
}

get_dir_size_mb() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo "0"
        return
    fi

    # macOS vs Linux du compatibility
    if [[ "$(uname)" == "Darwin" ]]; then
        du -sm "$dir" | awk '{print $1}'
    else
        du -sm "$dir" | awk '{print $1}'
    fi
}

# ============================================================================
# Backup Functions
# ============================================================================

create_archive() {
    local source="$1"
    local archive_dir="${PROJECT_ROOT}/backups/archive"
    mkdir -p "$archive_dir"

    local basename=$(basename "$source")
    local archive_file="${archive_dir}/${basename}_${TIMESTAMP}.tar.gz"

    log INFO "Creating archive: $archive_file"
    if tar -czf "$archive_file" -C "$(dirname "$source")" "$basename"; then
        log SUCCESS "Archived: $archive_file ($(get_dir_size_mb "$archive_file") MB)"
        return 0
    else
        log ERROR "Failed to create archive: $archive_file"
        return 1
    fi
}

# ============================================================================
# Cleanup Functions
# ============================================================================

cleanup_backups() {
    local backups_dir="${PROJECT_ROOT}/backups"
    log INFO "Cleaning backups/ (retention: $BACKUPS_RETENTION_DAYS days)"

    if [[ ! -d "$backups_dir" ]]; then
        log WARN "backups/ directory not found, skipping"
        return 0
    fi

    local deleted_count=0
    local freed_mb=0

    # Find timestamped backup directories (pattern: YYYYMMDD_HHMMSS)
    find "$backups_dir" -maxdepth 1 -type d -name "????????_??????" | while read -r backup_dir; do
        local age_days=$(get_file_age_days "$backup_dir")
        local size_mb=$(get_dir_size_mb "$backup_dir")
        local basename=$(basename "$backup_dir")

        # Check exclusions
        if should_exclude "$basename"; then
            continue
        fi

        # Check age threshold
        if [[ $age_days -gt $BACKUPS_RETENTION_DAYS ]]; then
            local action_msg="Delete backup $basename (age: ${age_days}d, size: ${size_mb}MB)"

            if confirm_action "$action_msg"; then
                # Backup first if requested
                if [[ "$BACKUP_FIRST" == "true" ]]; then
                    if ! create_archive "$backup_dir"; then
                        log ERROR "Archive failed, skipping deletion of $basename"
                        continue
                    fi
                fi

                # Delete
                if rm -rf "$backup_dir"; then
                    log SUCCESS "Deleted: $basename (freed ${size_mb}MB)"
                    deleted_count=$((deleted_count + 1))
                    freed_mb=$((freed_mb + size_mb))

                    # Audit trail
                    cat >> "$AUDIT_LOG" <<EOF
{"timestamp":"$(date -u +"%Y-%m-%dT%H:%M:%SZ")","action":"delete","target":"$basename","reason":"age-based (${age_days}d > ${BACKUPS_RETENTION_DAYS}d)","size_freed_mb":$size_mb,"operator":"cleanup_artifacts.sh"}
EOF
                else
                    log ERROR "Failed to delete: $basename"
                fi
            fi
        elif [[ "$VERBOSE" == "true" ]]; then
            log INFO "Keeping backup $basename (age: ${age_days}d ≤ ${BACKUPS_RETENTION_DAYS}d)"
        fi
    done

    if [[ "$DRY_RUN" == "false" ]] && [[ $deleted_count -gt 0 ]]; then
        log SUCCESS "Backups cleanup: deleted $deleted_count items, freed ${freed_mb}MB"
    fi
}

cleanup_logs() {
    local logs_dir="${PROJECT_ROOT}/logs"
    log INFO "Cleaning logs/ (retention: $LOGS_RETENTION_DAYS days)"

    if [[ ! -d "$logs_dir" ]]; then
        log WARN "logs/ directory not found, skipping"
        return 0
    fi

    local deleted_count=0
    local freed_mb=0

    find "$logs_dir" -maxdepth 1 -type f -name "*.log" | while read -r log_file; do
        local age_days=$(get_file_age_days "$log_file")
        local basename=$(basename "$log_file")

        # Skip audit log (self-reference)
        if [[ "$log_file" == "$AUDIT_LOG" ]]; then
            continue
        fi

        # Check exclusions
        if should_exclude "$basename"; then
            continue
        fi

        # Check age threshold
        if [[ $age_days -gt $LOGS_RETENTION_DAYS ]]; then
            # Check for ERROR severity logs (archive before delete)
            if [[ "$basename" == *"ERROR"* ]] && [[ "$BACKUP_FIRST" == "true" ]]; then
                local incident_dir="${logs_dir}/incidents/$(date +%Y-%m-%d)"
                mkdir -p "$incident_dir"
                cp "$log_file" "$incident_dir/" 2>/dev/null || true
                log INFO "Archived error log to $incident_dir/$basename"
            fi

            local action_msg="Delete log $basename (age: ${age_days}d)"

            if confirm_action "$action_msg"; then
                if rm -f "$log_file"; then
                    log SUCCESS "Deleted: $basename"
                    deleted_count=$((deleted_count + 1))
                else
                    log ERROR "Failed to delete: $basename"
                fi
            fi
        elif [[ "$VERBOSE" == "true" ]]; then
            log INFO "Keeping log $basename (age: ${age_days}d ≤ ${LOGS_RETENTION_DAYS}d)"
        fi
    done

    if [[ "$DRY_RUN" == "false" ]] && [[ $deleted_count -gt 0 ]]; then
        log SUCCESS "Logs cleanup: deleted $deleted_count files"
    fi
}

cleanup_cache() {
    local cache_dir="${PROJECT_ROOT}/cache"
    log INFO "Cleaning cache/ (retention: $CACHE_RETENTION_DAYS days)"

    if [[ ! -d "$cache_dir" ]]; then
        log WARN "cache/ directory not found, skipping"
        return 0
    fi

    local deleted_count=0

    find "$cache_dir" -type f -mtime +$CACHE_RETENTION_DAYS | while read -r cache_file; do
        local basename=$(basename "$cache_file")

        # Check exclusions
        if should_exclude "$basename"; then
            continue
        fi

        local action_msg="Delete cache $basename"

        if confirm_action "$action_msg"; then
            if rm -f "$cache_file"; then
                log SUCCESS "Deleted: $basename"
                deleted_count=$((deleted_count + 1))
            else
                log ERROR "Failed to delete: $basename"
            fi
        fi
    done

    if [[ "$DRY_RUN" == "false" ]] && [[ $deleted_count -gt 0 ]]; then
        log SUCCESS "Cache cleanup: deleted $deleted_count files"
    fi
}

cleanup_data_storage() {
    local data_storage_dir="${PROJECT_ROOT}/data_storage"
    log INFO "Cleaning data_storage/ (retention: $DATA_STORAGE_RETENTION_DAYS days, archive to S3)"

    if [[ ! -d "$data_storage_dir" ]]; then
        log WARN "data_storage/ directory not found, skipping"
        return 0
    fi

    # Note: This is a placeholder for S3 archival integration
    # Actual S3 upload requires AWS CLI configuration
    log WARN "data_storage/ archival to S3 not yet implemented"
    log INFO "Manual archival: tar -czf ohlcv_$(date +%Y%m).tar.gz data_storage/ && aws s3 cp ..."
}

cleanup_tool_caches() {
    log INFO "Cleaning tool caches (.mypy_cache, .pytest_cache, .ruff_cache)"

    local caches=(".mypy_cache" ".pytest_cache" ".ruff_cache")
    local deleted_count=0

    for cache in "${caches[@]}"; do
        local cache_dir="${PROJECT_ROOT}/$cache"
        if [[ -d "$cache_dir" ]]; then
            local action_msg="Delete $cache/ (safe, regenerated by tools)"

            if confirm_action "$action_msg"; then
                if rm -rf "$cache_dir"; then
                    log SUCCESS "Deleted: $cache/"
                    deleted_count=$((deleted_count + 1))
                else
                    log ERROR "Failed to delete: $cache/"
                fi
            fi
        fi
    done

    if [[ "$DRY_RUN" == "false" ]] && [[ $deleted_count -gt 0 ]]; then
        log SUCCESS "Tool caches cleanup: deleted $deleted_count directories"
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log INFO "Cleanup started (mode: $([ "$DRY_RUN" == "true" ] && echo "DRY-RUN" || echo "LIVE"))"
    log INFO "Project root: $PROJECT_ROOT"
    log INFO "Audit log: $AUDIT_LOG"

    # Pre-flight checks
    if [[ ! -d "$PROJECT_ROOT" ]]; then
        log ERROR "Project root not found: $PROJECT_ROOT"
        exit 1
    fi

    if [[ "$AUTO" == "true" ]] && [[ "$DRY_RUN" == "true" ]]; then
        log ERROR "--auto requires disabling --dry-run explicitly"
        exit 1
    fi

    # Execute cleanup functions
    cleanup_backups
    cleanup_logs
    cleanup_cache
    cleanup_data_storage
    cleanup_tool_caches

    log SUCCESS "Cleanup complete"
    log INFO "Review audit log: $AUDIT_LOG"
}

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --confirm)
            DRY_RUN=false
            CONFIRM=true
            shift
            ;;
        --auto)
            DRY_RUN=false
            AUTO=true
            CONFIRM=false
            shift
            ;;
        --backup-first)
            BACKUP_FIRST=true
            shift
            ;;
        --age-threshold)
            AGE_THRESHOLD_DAYS="$2"
            shift 2
            ;;
        --exclude-patterns)
            EXCLUDE_PATTERNS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log ERROR "Unknown option: $1"
            usage
            ;;
    esac
done

# Run main
main
