# Documentation Consolidation Priorities

Date: 2025-01-01
Status: Active consolidation plan

## Executive Summary

Repository documentation audit (January 2025) captured the transition toward a perps-first posture. Since then the project has moved to a spot-first model with INTX-gated perps. Many action items remain relevant (e.g., removing Alpaca references), but replace “update to perpetuals” with “update to spot/INTX-ready terminology” going forward.

## Priority 1: Root Directory ✅ MOSTLY COMPLETE

**Status**: Root directory already clean with only essential files
- 12 appropriate files (README, CONTRIBUTING, config files, etc.)
- No urgent action needed

## Priority 2: Eliminate Contradictions 🔴 CRITICAL

### Legacy References to Remove
- **30 Alpaca references** - Replace with Coinbase or remove
- **17 equities references** - Update to perpetuals/futures
- Multiple outdated API documentation files

### Duplicate/Overlapping Files Found
- Coinbase documentation spread across 5+ files
- API setup documentation in multiple locations
- Trading logic documented in multiple places

## Priority 3: Topic Consolidation 🟡 HIGH

### Setup & Configuration (Target: 1 comprehensive guide)
Current files to consolidate:
- docs/guides/api_key_setup.md
- docs/QUICK_START.md
- .env.template comments
- Various setup references

### Trading System (Target: 1 complete reference)
Current files to consolidate:
- docs/reference/trading_logic_perps.md
- docs/guides/paper_trading.md
- src/bot_v2/features/*/README.md files
- Various strategy documentation

### Coinbase Integration (Target: 1 unified reference)
Current files to consolidate:
- docs/reference/coinbase.md
- docs/reference/coinbase_troubleshooting.md
- docs/ARCHIVE/2024_implementation/coinbase/* (2 files)
- src/bot_v2/features/brokerages/coinbase/README.md
- src/bot_v2/features/brokerages/coinbase/COMPATIBILITY.md

## Priority 4: Navigation & Links 🟡 HIGH

### Current Issues
- Multiple broken internal references
- No comprehensive index
- Scattered documentation makes navigation difficult
- Missing cross-references between related topics

## Priority 5: Archive Historical Content 🟢 MEDIUM

### Files to Archive
- All Alpaca-related documentation
- Pre-perpetuals trading documentation
- Outdated architecture documentation
- Legacy test documentation

## Implementation Plan

### Week 1 Goals
1. **Day 1**: Remove/update all Alpaca and equities references
2. **Day 2**: Consolidate Coinbase documentation into single reference
3. **Day 3**: Consolidate setup/configuration documentation
4. **Day 4**: Consolidate trading system documentation
5. **Day 5**: Fix all navigation and create comprehensive index

### Success Metrics
- Zero Alpaca references in active docs
- Zero equities references (replaced with Coinbase spot/perps terminology)
- Reduce 170+ files to ~40 consolidated documents
- 100% functional internal links
- Clear navigation structure

## Next Immediate Actions

1. **Remove Legacy References**
   - Search and replace all Alpaca mentions
   - Update all equities references to perpetuals
   - Remove outdated API documentation

2. **Consolidate Coinbase Docs**
   - Merge 5+ Coinbase files into single comprehensive reference
   - Include all API, authentication, troubleshooting content
   - Archive redundant files

3. **Create Master Setup Guide**
   - Combine all setup/configuration documentation
   - Include environment, API keys, first run
   - Single source of truth for getting started
