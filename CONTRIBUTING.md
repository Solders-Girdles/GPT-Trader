# Contributing to GPT-Trader

We use a set of automated tools to ensure code quality and consistency. Please follow these steps for your development environment to ensure your contributions pass our automated checks.

## One-Time Setup

### 1. Install `pre-commit`

We use `pre-commit` to run checks before you commit your code. This helps catch issues early. You can install it using `pipx` (recommended) or `pip`.

**With `pipx` (Recommended):**
```bash
pipx install pre-commit
```

**With `pip`:**
```bash
pip install pre-commit
```

### 2. Install the Git Hooks

After installing `pre-commit`, you need to install the hooks into your local git repository.

```bash
pre-commit install
```

That's it! Now, every time you run `git commit`, the pre-commit hooks will run automatically. If they find any issues (like formatting errors), they may fix the files and abort the commit. In that case, just `git add` the modified files and run `git commit` again.

## Testing Requirements

### Current Expectations
- Keep the active suite (≈1484 collected / 1483 selected tests) green.
- Add focused tests for every new feature or regression fix.
- Document any skips or deselections tied to legacy code paths.

### Pre-PR Verification Checklist
1. Review `docs/agents/Document_Verification_Matrix.md` so you reference current documentation.
2. Refresh dependencies: `poetry install`.
3. Snapshot test discovery: `poetry run pytest --collect-only` (expect 1484 collected / 1483 selected / 1 deselected).
4. Run the full unit suite: `poetry run pytest -q`.
5. Execute slice-specific suites relevant to your change set (examples below).

### Recommended Commands

```bash
# Core suites
poetry run pytest tests/unit/gpt_trader -q
poetry run pytest tests/unit/gpt_trader/features/brokerages/coinbase -q
poetry run pytest tests/unit/gpt_trader/orchestration -q

# Coverage snapshot
poetry run pytest --cov=gpt_trader --cov-report=term-missing
```

### Test Metrics (Current)
- **Active Suite**: 1484 collected / 1483 selected / 1 deselected
- **Coverage Goal**: Maintain ~73% overall, >90% on new code paths
- **Integration Paths**: Coordinate with maintainers before toggling derivatives gates

## Running the Bot Locally

To run the spot trading bot for development, use the `coinbase-trader` command (derivatives stay gated behind INTX + `COINBASE_ENABLE_DERIVATIVES=1`):

```bash
poetry run coinbase-trader run --profile dev --dev-fast
```

## Development Workflow

Before branching, make sure to:
- Review `docs/agents/Document_Verification_Matrix.md` and confirm the sources you plan to rely on.
- Sync with the latest `main`.
- Run `poetry install` to pick up dependency changes.

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Write tests** for your changes
   - Unit tests required for all new functions
   - Integration tests for API interactions
   - Must maintain 100% pass rate on active tests
4. **Run the test suite** to ensure nothing is broken
   - `poetry run pytest --collect-only` must report 1484 collected / 1483 selected / 1 deselected
   - `poetry run pytest tests/unit/gpt_trader -q` must pass
   - No new test failures allowed
5. **Follow repository organization standards**
   - Place files in correct directories (see Repository Organization below)
   - Update documentation using consolidated structure
   - Add new documentation to appropriate `/docs` subdirectories
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to your fork** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

## Quality Standards

### Code Quality
- Clean, readable code with meaningful variable names
- Adhere to [Naming Standards](docs/naming.md)
- Comprehensive docstrings for public functions
- Type hints where beneficial
- Maximum line length: 100 characters

### Test Quality
- Descriptive test names that explain what's being tested
- One assertion per test when possible
- Use fixtures for shared setup
- Mock external dependencies

## Repository Organization

### Directory Structure Standards

The repository follows a standardized organization optimized for both human developers and AI agents:

#### Source Code & Configuration
- `/src/gpt_trader/` - Active trading system (vertical slice architecture)
- `/tests/` - Test files organized by component
- `/config/` - Configuration files, trading profiles, and templates
- `/scripts/` - Operational scripts organized by domain:
  - `agents/` - Automation helpers and inventory tooling
  - `analysis/` - Dependency and repository analysis utilities
  - `monitoring/` - Monitoring and dashboards
  - `testing/` - Test runners and fixtures
  - `validation/` - System validation flows

#### Documentation Standards
- `/docs/guides/` - How-to guides and tutorials
- `/docs/reference/` - Technical reference documentation
- `/docs/ops/` - Operations and maintenance procedures
- `/docs/ARCHIVE/` - Historical documentation (read-only)

#### Archive Management
- `/archived/2025/` - Current year development artifacts
- `/archived/experiments/` - Research and exploration work
- `/archived/infrastructure/` - Legacy system architectures
- `/archived/HISTORICAL/` - Long-term preserved data

### File Placement Guidelines

#### New Documentation
- **Tutorials/Guides**: `/docs/guides/`
- **API Reference**: `/docs/reference/`
- **Operations**: `/docs/ops/`
- **Never**: Root directory or legacy locations

#### New Scripts
- **Core Operations**: `/scripts/core/`
- **Testing/Validation**: `/scripts/validation/`
- **Monitoring**: `/scripts/monitoring/`
- **Automation/Agents**: `/scripts/agents/`

#### Deprecated Content
- Move to appropriate `/archived/` subdirectory
- Create redirect stub if high-traffic
- Update all internal references

### Naming Conventions
- **Documentation**: `category_topic.md` (lowercase, underscores)
- **Scripts**: `action_target.py` (clear purpose indication)
- **Directories**: `lowercase_names/` (descriptive, single purpose)

### Link Maintenance
- All documentation links must be functional
- Use relative paths within repository
- Update references when moving files
- Test links before submitting PRs

### Documentation Quality
- Keep README.md updated with current state
- Document breaking changes
- Include examples for complex features
- Update docs/agents/CLAUDE.md for AI context

## Pre-commit Hook Configuration

Our pre-commit hooks enforce:
- **Black**: Code formatting (line length 100)
- **Ruff**: Fast Python linting
- **MyPy**: Type checking (optional types)
- **Poetry Check**: Dependency validation
