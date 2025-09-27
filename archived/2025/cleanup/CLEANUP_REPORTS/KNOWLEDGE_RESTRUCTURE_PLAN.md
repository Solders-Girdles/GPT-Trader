# Knowledge Layer Restructure Plan

## Current Issues

### 1. Flat Knowledge Structure
Current .knowledge/ has 6 files with no hierarchy or workflow guidance.

### 2. Unnecessary Root Directories
```
.idea/        - IDE config (not needed for agents)
.github/      - GitHub workflows (not for agent use)
.venv/        - Virtual env (should be gitignored)
```

### 3. No Clear Agent Workflow
Agents don't know:
- Where to start
- What to read in what order
- Which files are for what purpose

## Proposed Structure

### New .knowledge/ Organization
```
.knowledge/
├── START_HERE.md           # First file agents should read
├── HOW_TO/                 # Task-based guides
│   ├── add_feature.md      # How to add new features
│   ├── fix_bugs.md         # How to fix bugs  
│   ├── run_tests.md        # How to test
│   └── common_tasks.md     # Other common operations
├── REFERENCE/              # Detailed references
│   ├── architecture.md     # System design
│   ├── slices.md          # Slice details
│   └── patterns.md        # Code patterns
├── STATE.json             # Current system state
└── RULES.md               # Critical rules (isolation, etc.)
```

### Agent Workflow Path
```
1. START → START_HERE.md
2. TASK → HOW_TO/[task].md
3. DETAILS → REFERENCE/[topic].md
4. STATUS → STATE.json
5. CONSTRAINTS → RULES.md
```

## Root Cleanup

### Remove/Archive
- `.idea/` → Not needed (IDE config)
- `.github/` → Archive (not for agents)

### Keep
- `.claude/` → Claude configuration
- `.git/` → Version control
- `.venv/` → Virtual environment (gitignored)
- `.knowledge/` → Agent knowledge (restructured)

## Benefits

1. **Clear Entry Point**: START_HERE.md
2. **Task-Oriented**: HOW_TO/ guides
3. **Detailed Reference**: When needed
4. **Clean Root**: No unnecessary directories
5. **Workflow-Based**: Matches how agents work