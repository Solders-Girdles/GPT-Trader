# Workflow Verification & Setup Plan

## Issues Found

### 1. Knowledge Layer Currency
- STATE.json has wrong date (2025-08-17 should be 2025-01-17)
- HOW_TO guides reference poetry but not config/ directory
- No developer setup instructions after removing .venv

### 2. Missing Setup Documentation
- No instructions for creating virtual environment
- No clear path to poetry/config files
- Agent guides assume poetry works but don't explain setup

### 3. Workflow Gaps
- START_HERE.md doesn't mention config/ directory
- No SETUP.md for developers
- Commands assume poetry is available

## Fixes Needed

### 1. Create SETUP.md for Developers
```markdown
# Developer Setup

## First Time Setup
1. Create virtual environment:
   python -m venv .venv
   
2. Activate it:
   source .venv/bin/activate  # Mac/Linux
   .venv\Scripts\activate      # Windows
   
3. Install dependencies:
   cd config
   pip install poetry
   poetry install
   cd ..
   
4. Create your .env:
   cp .env.template .env
   # Edit with your values
```

### 2. Update STATE.json
- Fix date to 2025-01-17
- Add config location
- Update with current status

### 3. Update HOW_TO Guides
- Add "cd config" before poetry commands
- Clarify where pyproject.toml lives
- Make commands work from root

### 4. Create Agent Workflow Check
- Verify each command works
- Test agent navigation path
- Ensure clarity

## For You (The Developer)

Since we removed .venv, you need to:
```bash
# 1. Create new virtual environment
python -m venv .venv

# 2. Activate it
source .venv/bin/activate  # Mac/Linux

# 3. Install poetry and dependencies
pip install poetry
cd config
poetry install
cd ..

# 4. Now you can work with the system
poetry run python src/bot_v2/test_all_slices.py
```