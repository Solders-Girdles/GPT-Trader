# 🎯 Final V2 Purification Plan

## Current Problems

### 1. V1 References as Distractions (34 instances)
Even warnings like "don't use V1" are wasting agent context. If V1 is gone, why mention it at all?

### 2. Unclear Document Organization
Currently we have:
- Root: 3 markdown files
- docs/: 3 subdirectories (knowledge/, V2_GUIDES/, WORKFLOW/)
- .knowledge/: 12+ files

**Where should agents put new documents?** It's not clear!

### 3. Fragmented Knowledge
- Some knowledge in docs/knowledge/
- Some in .knowledge/
- Some in root
- No clear hierarchy

## Solution: Crystal Clear Organization

### New Structure
```
GPT-Trader/
├── README.md              # Project overview ONLY
├── CLAUDE.md             # Agent control center ONLY
├── .knowledge/           # ALL agent knowledge
│   ├── NAVIGATION.md    # How to navigate V2
│   ├── ARCHITECTURE.md  # How V2 works
│   ├── OPERATIONS.md    # How to do tasks
│   ├── STATE.json       # Current system state
│   └── WHERE_TO_PUT.md  # Document placement guide
├── src/bot_v2/          # ALL code
│   └── features/        # Vertical slices
└── archived/            # Historical only
```

### Document Placement Rules

**Code**: Always in `src/bot_v2/features/[slice]/`

**Agent Knowledge**: Always in `.knowledge/`
- Navigation guides
- Architecture docs
- Operation patterns
- System state

**Human Docs**: Remove or minimize
- Humans can read README.md
- Everything else is for agents

**Reports/Analysis**: DON'T CREATE
- Use .knowledge/STATE.json for state
- No report files needed

## Implementation Steps

### Phase 1: Remove ALL V1 Mentions
- Strip out all "don't use V1" warnings
- Remove all "V1 archived" notes
- Delete all V1 comparison tables
- Pure V2 documentation only

### Phase 2: Consolidate Documentation
- Move everything from docs/ to .knowledge/
- Delete docs/ directory entirely
- Keep only README.md and CLAUDE.md in root
- Remove CONTRIBUTING.md (outdated)

### Phase 3: Create WHERE_TO_PUT.md
Clear rules for agents about document placement

### Phase 4: Simplify .knowledge/
- Merge similar files
- Remove redundancy
- Create clear, focused documents

## Expected Result

### Before
- 34 V1 distraction references
- 3 different doc locations
- Unclear where to put things
- Fragmented knowledge

### After
- 0 V1 mentions anywhere
- 1 knowledge location (.knowledge/)
- Crystal clear placement rules
- Unified, focused knowledge

## Benefits

1. **No Distractions**: Zero V1 mentions
2. **Clear Navigation**: One place for knowledge
3. **Obvious Placement**: WHERE_TO_PUT.md guide
4. **Minimal Context**: Only essential information
5. **Perfect Clarity**: No ambiguity

## Time Estimate
- Phase 1: 10 minutes (remove V1 mentions)
- Phase 2: 10 minutes (consolidate docs)
- Phase 3: 5 minutes (create placement guide)
- Phase 4: 10 minutes (simplify knowledge)

**Total: 35 minutes to PERFECT state**