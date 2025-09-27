# Project Organization & Maintenance Guidelines

## 📋 Organization Standards

### Directory Structure
```
GPT-Trader/
├── src/bot/           # Source code
├── tests/             # Test suite
├── docs/              # Documentation
├── examples/          # Example scripts and demos
├── scripts/           # Utility scripts
├── benchmarks/        # Performance benchmarks
├── k8s/              # Kubernetes configs
└── .github/          # GitHub configs and workflows
```

### Documentation Organization
```
docs/
├── README.md          # Documentation index
├── *.md              # Active documentation
├── archives/         # Historical documentation
│   ├── phases/      # Phase completion reports
│   └── weeks/       # Weekly progress reports
├── reports/          # Technical analysis reports
└── archived/         # Deprecated documentation
```

### Test Organization
```
tests/
├── unit/             # Fast, isolated tests
├── integration/      # Component interaction tests
├── system/          # System-wide tests
├── acceptance/      # User acceptance tests
├── performance/     # Performance benchmarks
└── production/      # Production readiness tests
```

## 🚫 What NOT to Put in Root Directory

### Never place these in root:
- Test files (belong in `/tests/`)
- Documentation files (belong in `/docs/`)
- Example scripts (belong in `/examples/`)
- Temporary files or reports
- Phase/week completion summaries
- Migration reports

### Acceptable root files:
- README.md
- LICENSE
- CONTRIBUTING.md
- Configuration files (.gitignore, pyproject.toml, etc.)
- .env.template (not .env)

## 📝 Documentation Guidelines

### Creating New Documentation
1. **Location**: Place in `/docs/` directory
2. **Naming**: Use UPPERCASE_WITH_UNDERSCORES.md for major docs
3. **Linking**: Update `/docs/README.md` index
4. **Archiving**: Move outdated docs to `/docs/archived/`

### Phase/Sprint Documentation
- Active phase plans: `/docs/CURRENT_PHASE.md`
- Completed phases: `/docs/archives/phases/`
- Sprint plans: `/docs/SPRINT_X_*.md`
- Sprint reviews: `/docs/archives/weeks/`

### Technical Reports
- Place in `/docs/reports/`
- Include date in filename if relevant
- Archive when outdated

## 🧪 Test File Guidelines

### Test File Placement
```python
# Source file: src/bot/strategy/demo_ma.py
# Test file:   tests/unit/strategy/test_demo_ma.py

# Source file: src/bot/portfolio/allocator.py
# Test file:   tests/unit/portfolio/test_allocator.py
```

### Test Naming Conventions
- Unit tests: `test_<module>_<function>.py`
- Integration tests: `test_<feature>_integration.py`
- System tests: `test_<system>_<aspect>.py`

## 🔄 Regular Maintenance Tasks

### Weekly
- [ ] Review root directory for misplaced files
- [ ] Archive completed phase/week documentation
- [ ] Update documentation index if needed

### Monthly
- [ ] Consolidate duplicate documentation
- [ ] Remove empty directories
- [ ] Archive outdated technical reports
- [ ] Review and update test organization

### Quarterly
- [ ] Major documentation reorganization
- [ ] Archive deprecated features documentation
- [ ] Clean up old migration/temporary files

## 🛠️ Maintenance Scripts

### Check for Misplaced Files
```bash
# Find test files outside of tests/
find . -name "test_*.py" -not -path "./tests/*" -not -path "./.git/*"

# Find .md files in root (except README, LICENSE, CONTRIBUTING)
ls -la *.md | grep -v -E "README|LICENSE|CONTRIBUTING"

# Find empty directories
find . -type d -empty
```

### Quick Cleanup Commands
```bash
# Archive phase documents
mv PHASE*.md docs/archives/phases/

# Archive weekly reports
mv WEEK*.md docs/archives/weeks/

# Move test files to tests/
find . -maxdepth 1 -name "test_*.py" -exec mv {} tests/system/ \;

# Remove empty directories
find . -type d -empty -delete
```

## 📊 Organization Metrics

### Good Organization Indicators
- ✅ Root directory has < 15 files
- ✅ All tests in `/tests/` directory
- ✅ Documentation indexed in `/docs/README.md`
- ✅ No duplicate documentation files
- ✅ Clear separation of concerns

### Warning Signs
- ⚠️ Test files in root directory
- ⚠️ Multiple files with similar names
- ⚠️ Documentation scattered across directories
- ⚠️ Empty directories in project
- ⚠️ Temporary files committed to repo

## 🚀 Automation

### Pre-commit Hooks
Consider adding pre-commit hooks to:
- Prevent test files in root
- Check for empty directories
- Validate documentation structure
- Enforce naming conventions

### CI/CD Checks
Add GitHub Actions to:
- Verify project structure
- Check documentation links
- Ensure test organization
- Monitor file placement

## 📚 References

- [Project Structure Best Practices](DEVELOPMENT_GUIDELINES.md)
- [Documentation Index](README.md)
- [Test Organization](../tests/README.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
