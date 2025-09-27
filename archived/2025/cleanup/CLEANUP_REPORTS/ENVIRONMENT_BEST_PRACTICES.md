# Environment Files Best Practices

## Current Situation

| File | Size | Tracked | Status | Action Needed |
|------|------|---------|--------|---------------|
| `.venv/` | 978MB | No ✅ | Exists locally | DELETE - Never in repo |
| `.env` | Small | No ✅ | Exists locally | DELETE - Security risk |
| `.env.template` | Small | Yes ✅ | Template file | KEEP - Documentation |

## Best Practices

### 1. `.venv/` - Virtual Environment
- **Should exist?** NO - Never in repository
- **Why?** Each developer creates their own
- **Size:** 978MB of unnecessary files
- **Action:** DELETE immediately

### 2. `.env` - Environment Variables
- **Should exist?** NO - Never in repository
- **Why?** Contains secrets (even placeholders are risky)
- **Security:** Could expose sensitive configuration
- **Action:** DELETE immediately

### 3. `.env.template` - Template File
- **Should exist?** YES - Keep in repository
- **Why?** Shows developers what variables are needed
- **Purpose:** Documentation without secrets
- **Action:** KEEP as is

## Recommended Actions

```bash
# 1. Remove .venv directory (978MB savings!)
rm -rf .venv

# 2. Remove .env file (security)
rm .env

# 3. Keep .env.template (documentation)
# No action needed

# 4. Verify .gitignore is correct (already good)
cat .gitignore | grep -E "venv|\.env"
```

## For Developers

### Setting Up Environment
```bash
# 1. Create virtual environment (locally, not tracked)
python -m venv .venv

# 2. Activate it
source .venv/bin/activate  # On Unix/Mac
# or
.venv\Scripts\activate  # On Windows

# 3. Install dependencies
cd config && poetry install

# 4. Create your .env from template
cp .env.template .env
# Edit .env with your actual values
```

## Security Benefits
- No secrets in repository
- No accidental credential exposure
- Clear separation of config template vs actual config

## Space Benefits
- Save 978MB by removing .venv
- Repository stays clean
- Faster cloning

## Final State Should Be
```
Root:
├── .env.template    ✅ (tracked, template only)
├── .gitignore       ✅ (excludes .env and .venv)
NO .env             ❌ (never tracked)
NO .venv/           ❌ (never tracked)
```