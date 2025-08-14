# Claude Code Assistant Guide for GPT-Trader

## Project Overview
GPT-Trader is an advanced ML-powered autonomous portfolio management system for algorithmic trading. It predicts market movements and executes trades with disciplined risk management.

---

## Context Budget Policy
- **Main thread**: goals, current step, brief status, tiny diffs (<40 lines).
- **Subagents**: heavy reads, logs, wide diffs, repo scans.
  - Return a **10-bullet digest** + file paths + artifact links.
- Never paste >200 lines into main; summarize and reference paths.
- If a task lacks a clear 5–8 step plan, call **`planner`** first.

---

## Agentic Playbook (mechanical)
**Explore (read-only)**  
Read {paths}. Do not write code. Output: 10-bullet architecture summary, 5 risks, smallest viable change. If unknowns → `planner`.

**Plan**  
`planner`: For **{TASK-ID}**, output ≤8 steps, affected files, test names, perf checks, rollback.

**Implement**  
Do **step 1 only**. Keep edits ≤5 files or ≤200 LOC.  
Run `test-runner`. If failing, call `debugger` for root cause + minimal patch; apply fix.  
Repeat until step 1 passes, then **STOP** and report.

**Review**  
`agentic-code-reviewer`: Review diff for **{TASK-ID}**. Return CRITICAL/HIGH/NICE-TO-HAVE with file:line. Apply only CRITICAL/HIGH now.

**Doc & PR**  
Update touched docs. Open PR:
- **Title**: `{TASK-ID}: {summary}`
- **Body**: goals, changes, tests, risks, rollback, `git diff --stat`.
Pause for human approval.

---

## Command Registry
# SoT & drift
- `python scripts/generate_filemap.py`
- `rg -n "src/bot/|python -m src\.bot|docker-compose|pytest" docs CLAUDE.md`
- `python scripts/doc_check.py --files CLAUDE.md docs/**/*.md`

# Test / perf
- `pytest -q`
- `pytest tests/performance/benchmark_consolidated.py -q`

# Ops
- `python -m src.bot.cli dashboard`
- `python -m src.bot.cli backtest --symbol AAPL --start 2024-01-01 --end 2024-06-30 --strategy trend_breakout`
- `docker-compose -f deploy/postgres/docker-compose.yml up -d`

---

## Project Structure

GPT-Trader/
├── src/bot/
│   ├── ml/
│   │   ├── integrated_pipeline.py
│   │   ├── auto_retraining.py
│   │   └── deep_learning/
│   │       ├── lstm_architecture.py
│   │       ├── lstm_data_pipeline.py
│   │       ├── lstm_training.py
│   │       ├── attention_mechanisms.py
│   │       ├── transformer_models.py
│   │       └── integrated_lstm_pipeline.py
│   ├── monitoring/
│   ├── risk/
│   ├── data/
│   │   └── sentiment/         # Planned
│   └── strategy/              # Planned
│       ├── multi_asset/
│       └── options/
├── docs/
│   ├── PHASE_3_COMPLETION_REPORT.md
│   ├── PHASE_4_TASK_BREAKDOWN.md
│   └── OPERATIONS_RUNBOOK.md
└── tests/

---

## Core Agents (registered)
| Agent | Purpose | Tools |
|---|---|---|
| `planner` | Turn a request into a ≤8-step plan w/ files & tests | `read`, `grep` |
| `repo-structure-guardian` | SoT scans, drift reports, doc/path checks | `read`, `grep`, `shell` |
| `test-runner` | Run tests, summarize failures, suggest smallest next fix | `read`, `shell` |
| `debugger` | Localize root cause; propose minimal patch hunks | `read`, `grep` |
| `agentic-code-reviewer` | Review diffs for correctness/security/perf/tests | `read`, `grep`, `git_diff` |
| `trading-strategy-consultant` | Validate strategy logic & risk; outline tests | `read`, `grep` |
| `performance-optimizer` | Run perf suites; pinpoint hotspots; suggest fixes | `read`, `grep`, `shell` |

---

## Routing Matrix (GPT-Trader tasks)
| Task prefix | Primary route | Secondary |
|---|---|---|
| **DL-\*** | `planner → trading-strategy-consultant → implement (main) → test-runner → agentic-code-reviewer` | `performance-optimizer` (if slow) |
| **RL-\*** | `planner → trading-strategy-consultant → implement → test-runner` | `performance-optimizer` |
| **SENT-/MICRO-\*** | `planner → implement → test-runner` | `repo-structure-guardian` (schemas/paths) |
| **MULTI-/OPT-\*** | `planner → trading-strategy-consultant → implement → test-runner → agentic-code-reviewer` | — |
| **SOT-\*** | `repo-structure-guardian` | `agentic-code-reviewer` (Phase 5) |

---

## Repository Cleanup & Single-Source-of-Truth (SoT) Program

### Overarching Goals
- Make this `CLAUDE.md` the authoritative SoT for agents.
- Purge or fix deprecated/drifted content across code and docs.
- Add guardrails to keep docs and code in sync.

### How agents use this list
- Pick the highest-priority unchecked task in the current phase, execute it, then check it off and commit with the task ID.
- Prefer CLI/commands shown here; avoid ad-hoc scripts.
- Include task IDs in PR titles and commit messages (e.g., "SOT-021: Align Phase 4 deliverables filenames").

### Phase 0 (Day 1): Baseline Inventory
- [x] **SOT-001**: Generate current file map.
  - Command: `python scripts/generate_filemap.py`
- [x] **SOT-002**: Scan docs for path/command references; collect invalid ones.
  - Command: `rg -n "src/bot/|python -m src\\.bot|docker-compose|pytest" docs CLAUDE.md`
- [x] **SOT-003**: Scan code for absolute imports/paths that may drift.
  - Command: `rg -n "from\\s+src\\.|import\\s+src\\." src`
- [x] **SOT-004**: Emit drift report artifact at `docs/reports/doc_drift.json`.

### Phase 1 (Day 2): Deprecation Sweep
- [ ] **SOT-010**: Identify orphaned/unused files (>60d) and archive to `docs/archived/` with reason/date.
- [ ] **SOT-011**: Update `docs/archived/DEPRECATED_FILES_LIST.md`.
- [ ] **SOT-012**: Remove `CLAUDE.md.bak` after verifying parity.

### Phase 2 (Day 3): Normalize Structure & References
- [x] **SOT-020**: Standardize references to `src/bot/strategy/` (singular) across docs.
- [x] **SOT-021**: Align Phase 4 deliverables filenames in `docs/PHASE_4_TASK_BREAKDOWN.md` to actual files:
  - `lstm_architecture.py`, `lstm_data_pipeline.py`, `lstm_training.py`, `attention_mechanisms.py`, `transformer_models.py`.
- [x] **SOT-022**: Refresh `docs/ARCHITECTURE_FILEMAP.md` from generated file map.
- [x] **SOT-023**: Fix monitoring/risk module paths in docs to match `src/bot/monitoring/*` and `src/bot/risk/*`.

### Phase 3 (Day 4): Automate SoT Generation
- [ ] **SOT-030**: Extend `scripts/update_claude_md.py` to auto-populate:
  - Project Structure (from file map) and validated Commands.
- [ ] **SOT-031**: Generator validates imports/commands before insertion; skip invalid entries.
- [ ] **SOT-032**: Add `--dry-run` and `--diff` flags; write out only on success.
- [ ] **SOT-033**: Add auto-generated markers in `CLAUDE.md` sections.

### Phase 4 (Day 5): Validation Guards
- [ ] **SOT-040**: Create `scripts/doc_check.py` to parse `CLAUDE.md`/`docs/*.md`, verify file paths, `python -m` commands, and imports.
- [ ] **SOT-041**: Add a pre-commit hook to run `doc_check.py` on changed docs.
- [ ] **SOT-042**: Add CI step to run `doc_check.py` and tests.
  - Example: `python scripts/doc_check.py --files CLAUDE.md docs/**/*.md && pytest -q`
- [ ] **SOT-043**: Optional: nightly job to run generator + doc check and open a PR on drift.

### Phase 5 (Day 6-7): Finalization & Lock
- [ ] **SOT-050**: Run all commands from `CLAUDE.md` top-to-bottom; fix residual issues.
- [ ] **SOT-051**: Mark auto-generated blocks with "Do not edit manually" notes.
- [ ] **SOT-052**: Document the SoT process in `docs/OPERATIONS_RUNBOOK.md`.
- [ ] **SOT-053**: Update `CHANGELOG.md` with doc cleanup and SoT adoption summary.

### Agent runbook for this program
- **Selection**: Start at the current phase; choose the first unchecked task.
- **Execution**: Use the listed commands; prefer CLI over direct scripts.
- **Validation**: If a command fails, fix the root cause or open an issue linking the task ID.
- **Completion**: Check off the task in `CLAUDE.md` and reference the task ID in commits/PRs.


## Important Technical Details

## AI Subagent Reference Guide

### Overview
This section documents all available AI subagents, their specialized expertise, and appropriate use cases. Agents should be called proactively when their expertise matches the task at hand.

### Core Analysis & Configuration Agents

#### 1. **project-analyst**
- **Expertise**: Codebase analysis, framework detection, tech stack identification
- **When to use**: MUST BE USED for any new or unfamiliar codebase. Use PROACTIVELY to detect frameworks, tech stacks, and architecture before routing to specialists
- **Tools**: LS, Read, Grep, Glob, Bash

#### 2. **team-configurator**
- **Expertise**: AI team setup and configuration
- **When to use**: MUST BE USED to set up or refresh AI development team. Use PROACTIVELY on new repos, after major tech stack changes, or when user asks to configure the AI team
- **Tools**: LS, Read, WriteFile, Bash, Glob, Grep

#### 3. **tech-lead-orchestrator**
- **Expertise**: Strategic technical analysis and task planning
- **When to use**: MUST BE USED for multi-step development tasks, feature implementation, or architectural decisions. Returns structured findings and task breakdowns
- **Tools**: Read, Grep, Glob, LS, Bash

### Backend Development Agents

#### 4. **backend-developer**
- **Expertise**: General backend development across any language/stack
- **When to use**: MUST BE USED for server-side code when no framework-specific agent exists. Use PROACTIVELY for production-ready features
- **Tools**: Full access

#### 5. **rails-backend-expert**
- **Expertise**: Ruby on Rails backend development
- **When to use**: MUST BE USED for Rails backend tasks, ActiveRecord models, controllers, or Rails-specific implementation
- **Tools**: Full access

#### 6. **rails-api-developer**
- **Expertise**: Rails API development (RESTful and GraphQL)
- **When to use**: MUST BE USED for Rails API development, API controllers, serializers, or GraphQL implementations
- **Tools**: Full access

#### 7. **rails-activerecord-expert**
- **Expertise**: Rails ActiveRecord ORM and database optimization
- **When to use**: Complex queries, database performance, migrations in Rails projects
- **Tools**: Full access

#### 8. **laravel-backend-expert**
- **Expertise**: Laravel backend architecture (MVC, Inertia.js, Livewire, API-only)
- **When to use**: MUST BE USED for Laravel backend tasks, controllers, services, or Eloquent models
- **Tools**: Full access

#### 9. **laravel-eloquent-expert**
- **Expertise**: Laravel Eloquent ORM, schemas, migrations, query optimization
- **When to use**: MUST BE USED for data modeling, persistence, or query optimization in Laravel projects
- **Tools**: Read, Grep, Glob, LS, Bash, WebFetch

#### 10. **django-backend-expert**
- **Expertise**: Django backend development (models, views, services)
- **When to use**: MUST BE USED for Django backend development tasks following Django best practices
- **Tools**: Full access

#### 11. **django-orm-expert**
- **Expertise**: Django ORM optimization and database performance
- **When to use**: Complex queries, database design, migrations for Django applications
- **Tools**: Full access

#### 12. **django-api-developer**
- **Expertise**: Django REST Framework and GraphQL APIs
- **When to use**: MUST BE USED for Django API development, DRF serializers, viewsets, or GraphQL schemas
- **Tools**: Full access

### Frontend Development Agents

#### 13. **frontend-developer**
- **Expertise**: Responsive, accessible, high-performance UIs (vanilla JS/TS, React, Vue, Angular, Svelte)
- **When to use**: MUST BE USED for user-facing code when no framework-specific agent exists. Use PROACTIVELY for UI requirements
- **Tools**: LS, Read, Grep, Glob, Bash, Write, Edit, WebFetch

#### 14. **react-component-architect**
- **Expertise**: Modern React patterns, hooks, component design
- **When to use**: MUST BE USED for React component development, hooks implementation, or React architecture decisions
- **Tools**: Full access

#### 15. **react-nextjs-expert**
- **Expertise**: Next.js framework (SSR, SSG, ISR, full-stack React)
- **When to use**: Next.js applications requiring SSR, SSG, or ISR implementation
- **Tools**: Full access

#### 16. **vue-component-architect**
- **Expertise**: Vue 3 Composition API, scalable component architecture
- **When to use**: MUST BE USED for Vue components, composables, or Vue architecture decisions
- **Tools**: Full access

#### 17. **vue-nuxt-expert**
- **Expertise**: Nuxt.js framework (SSR, SSG, full-stack Vue)
- **When to use**: Nuxt applications requiring SSR, SSG, or full-stack Vue solutions
- **Tools**: Full access

#### 18. **tailwind-frontend-expert**
- **Expertise**: Tailwind CSS styling, utility-first CSS, responsive components
- **When to use**: MUST BE USED for Tailwind CSS styling, utility-first refactors, or responsive component work. Use PROACTIVELY for UI tasks involving Tailwind
- **Tools**: LS, Read, Grep, Glob, Bash, Write, Edit, MultiEdit, WebFetch

### API & Architecture Agents

#### 19. **api-architect**
- **Expertise**: RESTful design, GraphQL schemas, API contracts (OpenAPI/GraphQL specs)
- **When to use**: MUST BE USED PROACTIVELY for new or revised API contracts. Produces resource models, specs, and guidance on auth, versioning, pagination
- **Tools**: Read, Grep, Glob, Write, WebFetch, WebSearch

### Quality & Documentation Agents

#### 20. **documentation-specialist**
- **Expertise**: Project documentation (READMEs, API specs, architecture guides)
- **When to use**: MUST BE USED for documentation. Use PROACTIVELY after major features, API changes, or when onboarding developers
- **Tools**: LS, Read, Grep, Glob, Bash, Write

#### 21. **code-reviewer**
- **Expertise**: Security-aware code review
- **When to use**: MUST BE USED after every feature, bug-fix, or pull-request. Use PROACTIVELY before merging to main
- **Tools**: LS, Read, Grep, Glob, Bash

#### 22. **agentic-code-reviewer**
- **Expertise**: Detecting AI-assisted development pitfalls (over-engineering, incomplete implementations, unnecessary complexity)
- **When to use**: After logical chunks of AI-generated code or when reviewing architectural decisions
- **Tools**: Full access

### Performance & Analysis Agents

#### 23. **performance-optimizer**
- **Expertise**: System performance optimization, bottleneck identification
- **When to use**: MUST BE USED for slowness, high cloud costs, or scaling concerns. Use PROACTIVELY before traffic spikes
- **Tools**: LS, Read, Grep, Glob, Bash

#### 24. **code-archaeologist**
- **Expertise**: Legacy/complex codebase exploration and documentation
- **When to use**: MUST BE USED for unfamiliar, legacy, or complex codebases. Use PROACTIVELY before refactors, onboarding, audits
- **Tools**: LS, Read, Grep, Glob, Bash

### Specialized Agents

#### 25. **repo-structure-guardian**
- **Expertise**: Project organization standards and file placement verification
- **When to use**: When adding new components, moving files, adding tests/documentation, or reviewing project organization
- **Tools**: Full access

#### 26. **trading-strategy-consultant**
- **Expertise**: Financial trading strategies, risk management, trading tools
- **When to use**: For trading strategy validation, tool recommendations, technical indicators, portfolio management, or backtesting methodologies
- **Tools**: Full access

### Hybrid Agents

#### 27. **gemini-gpt-hybrid**
- **Expertise**: Analysis using Gemini and GPT models, returning insights to Claude
- **When to use**: Complex problem identification requiring multiple AI perspectives
- **Tools**: Read, Edit, Bash, Grep, Glob

#### 28. **gemini-gpt-hybrid-hard**
- **Expertise**: AGGRESSIVE code generation using Gemini and GPT
- **When to use**: Rapid development and automation tasks
- **Tools**: Bash

### Language-Specific Agents

#### 29. **python-pro**
- **Expertise**: Modern Python 3.11+ with type safety, async programming, data science, web frameworks
- **When to use**: Python development requiring Pythonic patterns and production-ready code quality
- **Tools**: Read, Write, MultiEdit, Bash, pip, pytest, black, mypy, poetry, ruff, bandit

#### 30. **javascript-pro**
- **Expertise**: Modern ES2023+ features, async programming, full-stack JavaScript
- **When to use**: JavaScript development for browser APIs and Node.js with emphasis on performance
- **Tools**: Read, Write, MultiEdit, Bash, node, npm, eslint, prettier, jest, webpack, rollup

#### 31. **typescript-pro**
- **Expertise**: Advanced TypeScript type system, full-stack development, build optimization
- **When to use**: Type-safe patterns for frontend and backend with developer experience focus
- **Tools**: Read, Write, MultiEdit, Bash, tsc, eslint, prettier, jest, webpack, vite, tsx

#### 32. **rust-engineer**
- **Expertise**: Systems programming, memory safety, zero-cost abstractions
- **When to use**: Mission-critical applications requiring ownership patterns and async programming
- **Tools**: Read, Write, MultiEdit, Bash, cargo, rustc, clippy, rustfmt, miri, rust-analyzer

#### 33. **golang-pro**
- **Expertise**: High-performance systems, concurrent programming, cloud-native microservices
- **When to use**: Idiomatic Go patterns with emphasis on simplicity and efficiency
- **Tools**: Read, Write, MultiEdit, Bash, go, gofmt, golint, delve, golangci-lint

#### 34. **java-architect**
- **Expertise**: Enterprise-grade applications, Spring ecosystem, cloud-native development
- **When to use**: Modern Java features, reactive programming, microservices patterns
- **Tools**: Read, Write, MultiEdit, Bash, maven, gradle, javac, junit, spotbugs, jmh, spring-cli

#### 35. **csharp-developer**
- **Expertise**: Modern .NET development, ASP.NET Core, cloud-native applications
- **When to use**: C# 12 features, Blazor, cross-platform development with clean architecture
- **Tools**: Read, Write, MultiEdit, Bash, dotnet, msbuild, nuget, xunit, resharper, dotnet-ef

#### 36. **php-pro**
- **Expertise**: Modern PHP 8.3+ with strong typing, async programming, enterprise frameworks
- **When to use**: Laravel, Symfony, modern PHP patterns with performance focus
- **Tools**: Read, Write, MultiEdit, Bash, php, composer, phpunit, phpstan, php-cs-fixer, psalm

#### 37. **swift-expert**
- **Expertise**: Swift 5.9+ with async/await, SwiftUI, protocol-oriented programming
- **When to use**: Apple platforms development, server-side Swift with safety emphasis
- **Tools**: Read, Write, MultiEdit, Bash, swift, swiftc, xcodebuild, instruments, swiftlint, swift-format

#### 38. **kotlin-specialist**
- **Expertise**: Coroutines, multiplatform development, Android applications
- **When to use**: Functional programming patterns, DSL design, modern Kotlin features
- **Tools**: Read, Write, MultiEdit, Bash, kotlin, gradle, detekt, ktlint, junit5, kotlinx-coroutines

#### 39. **cpp-pro**
- **Expertise**: Modern C++20/23, systems programming, high-performance computing
- **When to use**: Template metaprogramming, zero-overhead abstractions, low-level optimization
- **Tools**: Read, Write, MultiEdit, Bash, g++, clang++, cmake, make, gdb, valgrind, clang-tidy

#### 40. **sql-pro**
- **Expertise**: Complex query optimization across PostgreSQL, MySQL, SQL Server, Oracle
- **When to use**: Advanced SQL features, indexing strategies, data warehousing patterns
- **Tools**: Read, Write, MultiEdit, Bash, psql, mysql, sqlite3, sqlplus, explain, analyze

### Framework-Specific Agents

#### 41. **spring-boot-engineer**
- **Expertise**: Spring Boot 3+ with cloud-native patterns, microservices, reactive programming
- **When to use**: Spring Cloud integration, enterprise Java solutions
- **Tools**: maven, gradle, spring-cli, docker, kubernetes, intellij, git, postgresql

#### 42. **nextjs-developer**
- **Expertise**: Next.js 14+ with App Router, server components, full-stack features
- **When to use**: Server components, server actions, performance optimization, production deployment
- **Tools**: next, vercel, turbo, prisma, playwright, npm, typescript, tailwind

#### 43. **dotnet-core-expert**
- **Expertise**: .NET 8 with modern C# features, cross-platform development
- **When to use**: Minimal APIs, cloud-native applications, microservices
- **Tools**: dotnet-cli, nuget, xunit, docker, azure-cli, visual-studio, git, sql-server

#### 44. **angular-architect**
- **Expertise**: Angular 15+ with enterprise patterns, RxJS, NgRx state management
- **When to use**: Micro-frontend architecture, performance optimization for enterprise apps
- **Tools**: angular-cli, nx, jest, cypress, webpack, rxjs, npm, typescript

#### 45. **flutter-expert**
- **Expertise**: Flutter 3+ with modern architecture patterns, cross-platform development
- **When to use**: Custom animations, native integrations, performance optimization
- **Tools**: flutter, dart, android-studio, xcode, firebase, fastlane, git, vscode

#### 46. **react-specialist**
- **Expertise**: React 18+ with modern patterns, performance optimization, server components
- **When to use**: Advanced hooks, production-ready architectures, scalable applications
- **Tools**: vite, webpack, jest, cypress, storybook, react-devtools, npm, typescript

#### 47. **vue-expert**
- **Expertise**: Vue 3 with Composition API, reactivity system, performance optimization
- **When to use**: Nuxt 3 development, enterprise patterns, elegant reactive applications
- **Tools**: vite, vue-cli, vitest, cypress, vue-devtools, npm, typescript, pinia

#### 48. **rails-expert**
- **Expertise**: Rails 7+ with modern conventions, Hotwire/Turbo, Action Cable
- **When to use**: Convention over configuration, rapid application development
- **Tools**: rails, rspec, sidekiq, redis, postgresql, bundler, git, rubocop

#### 49. **django-developer**
- **Expertise**: Django 4+ with modern Python practices, scalable web applications
- **When to use**: REST API development, async views, enterprise patterns
- **Tools**: django-admin, pytest, celery, redis, postgresql, docker, git, python

#### 50. **laravel-specialist**
- **Expertise**: Laravel 10+ with modern PHP practices, Eloquent ORM, queue systems
- **When to use**: Enterprise features, scalable web applications and APIs
- **Tools**: artisan, composer, pest, redis, mysql, docker, git, php

### Data & ML Agents

#### 51. **data-scientist**
- **Expertise**: Statistical analysis, machine learning, business insights
- **When to use**: Exploratory data analysis, predictive modeling, data storytelling
- **Tools**: python, jupyter, pandas, sklearn, matplotlib, statsmodels

#### 52. **data-engineer**
- **Expertise**: Scalable data pipelines, ETL/ELT processes, data infrastructure
- **When to use**: Big data technologies, cloud platforms, reliable data platforms
- **Tools**: spark, airflow, dbt, kafka, snowflake, databricks

#### 53. **data-analyst**
- **Expertise**: Business intelligence, data visualization, statistical analysis
- **When to use**: SQL, Python, BI tools for actionable insights and business impact
- **Tools**: Read, Write, MultiEdit, Bash, sql, python, tableau, powerbi, looker, dbt, excel

#### 54. **ml-engineer**
- **Expertise**: ML model lifecycle, production deployment, system optimization
- **When to use**: Traditional ML and deep learning for scalable, reliable ML systems
- **Tools**: mlflow, kubeflow, tensorflow, sklearn, optuna

#### 55. **machine-learning-engineer**
- **Expertise**: Production model deployment, serving infrastructure, scalable ML systems
- **When to use**: Model optimization, real-time inference, edge deployment
- **Tools**: Read, Write, MultiEdit, Bash, tensorflow, pytorch, onnx, triton, bentoml, ray, vllm

#### 56. **mlops-engineer**
- **Expertise**: ML infrastructure, platform engineering, operational excellence
- **When to use**: CI/CD for ML, model versioning, scalable ML platforms
- **Tools**: mlflow, kubeflow, airflow, docker, prometheus, grafana

#### 57. **ai-engineer**
- **Expertise**: AI system design, model implementation, production deployment
- **When to use**: Multiple AI frameworks for scalable, efficient, ethical AI solutions
- **Tools**: python, jupyter, tensorflow, pytorch, huggingface, wandb

#### 58. **nlp-engineer**
- **Expertise**: Natural language processing, transformer models, text pipelines
- **When to use**: Multilingual support, real-time NLP performance
- **Tools**: Read, Write, MultiEdit, Bash, transformers, spacy, nltk, huggingface, gensim, fasttext

#### 59. **llm-architect**
- **Expertise**: Large language model architecture, deployment, optimization
- **When to use**: LLM system design, fine-tuning strategies, production serving
- **Tools**: transformers, langchain, llamaindex, vllm, wandb

#### 60. **prompt-engineer**
- **Expertise**: Prompt design, optimization, and management for LLMs
- **When to use**: Prompt architecture, evaluation frameworks, production prompt systems
- **Tools**: openai, anthropic, langchain, promptflow, jupyter

### Database & Infrastructure Agents

#### 61. **database-administrator**
- **Expertise**: High-availability systems, performance optimization, disaster recovery
- **When to use**: PostgreSQL, MySQL, MongoDB, Redis operational excellence
- **Tools**: Read, Write, MultiEdit, Bash, psql, mysql, mongosh, redis-cli, pg_dump, percona-toolkit, pgbench

#### 62. **database-optimizer**
- **Expertise**: Query optimization, performance tuning, scalability
- **When to use**: Execution plan analysis, index strategies, peak database performance
- **Tools**: explain, analyze, pgbench, mysqltuner, redis-cli

#### 63. **postgres-pro**
- **Expertise**: PostgreSQL administration, performance optimization, high availability
- **When to use**: PostgreSQL internals, advanced features, enterprise deployment
- **Tools**: psql, pg_dump, pgbench, pg_stat_statements, pgbadger

### DevOps & Cloud Agents

#### 64. **devops-engineer**
- **Expertise**: CI/CD, containerization, cloud platforms, automation
- **When to use**: Bridging development and operations with culture and collaboration focus
- **Tools**: Read, Write, MultiEdit, Bash, docker, kubernetes, terraform, ansible, prometheus, jenkins

#### 65. **sre-engineer**
- **Expertise**: Site Reliability Engineering, SLOs, automation, operational excellence
- **When to use**: Reliability engineering, chaos testing, toil reduction
- **Tools**: Read, Write, MultiEdit, Bash, prometheus, grafana, terraform, kubectl, python, go, pagerduty

#### 66. **platform-engineer**
- **Expertise**: Internal developer platforms, self-service infrastructure
- **When to use**: Platform APIs, GitOps workflows, golden path templates
- **Tools**: Read, Write, MultiEdit, Bash, kubectl, helm, argocd, crossplane, backstage, terraform, flux

#### 67. **cloud-architect**
- **Expertise**: Multi-cloud strategies, scalable architectures, cost-effective solutions
- **When to use**: AWS, Azure, GCP with security, performance, compliance focus
- **Tools**: Read, Write, MultiEdit, Bash, aws-cli, azure-cli, gcloud, terraform, kubectl, draw.io

#### 68. **kubernetes-specialist**
- **Expertise**: Container orchestration, cluster management, cloud-native architectures
- **When to use**: Production-grade deployments, security hardening, performance optimization
- **Tools**: Read, Write, MultiEdit, Bash, kubectl, helm, kustomize, kubeadm, k9s, stern, kubectx

#### 69. **terraform-engineer**
- **Expertise**: Infrastructure as code, multi-cloud provisioning, modular architecture
- **When to use**: Terraform best practices, state management, enterprise patterns
- **Tools**: Read, Write, MultiEdit, Bash, terraform, terragrunt, tflint, terraform-docs, checkov, infracost

#### 70. **deployment-engineer**
- **Expertise**: CI/CD pipelines, release automation, deployment strategies
- **When to use**: Blue-green, canary, rolling deployments with zero-downtime releases
- **Tools**: Read, Write, MultiEdit, Bash, ansible, jenkins, gitlab-ci, github-actions, argocd, spinnaker

### Security & Testing Agents

#### 71. **security-engineer**
- **Expertise**: DevSecOps, cloud security, compliance frameworks
- **When to use**: Security automation, vulnerability management, zero-trust architecture
- **Tools**: Read, Write, MultiEdit, Bash, nmap, metasploit, burp, vault, trivy, falco, terraform

#### 72. **security-auditor**
- **Expertise**: Security assessments, compliance validation, risk management
- **When to use**: Security frameworks, audit methodologies, regulatory adherence
- **Tools**: Read, Grep, nessus, qualys, openvas, prowler, scout, suite, compliance, checker

#### 73. **penetration-tester**
- **Expertise**: Ethical hacking, vulnerability assessment, security testing
- **When to use**: Offensive security techniques, exploit development, security assessments
- **Tools**: Read, Grep, nmap, metasploit, burpsuite, sqlmap, wireshark, nikto, hydra

#### 74. **compliance-auditor**
- **Expertise**: Regulatory frameworks, data privacy laws, security standards
- **When to use**: GDPR, HIPAA, PCI DSS, SOC 2, ISO certifications
- **Tools**: Read, Write, MultiEdit, Bash, prowler, scout, checkov, terrascan, cloudsploit, lynis

#### 75. **qa-expert**
- **Expertise**: Comprehensive quality assurance, test strategy, quality metrics
- **When to use**: Manual and automated testing, test planning, quality processes
- **Tools**: Read, Grep, selenium, cypress, playwright, postman, jira, testrail, browserstack

#### 76. **test-automator**
- **Expertise**: Test frameworks, CI/CD integration, comprehensive test coverage
- **When to use**: Maintainable, scalable, efficient automated testing solutions
- **Tools**: Read, Write, selenium, cypress, playwright, pytest, jest, appium, k6, jenkins

#### 77. **accessibility-tester**
- **Expertise**: WCAG compliance, inclusive design, universal access
- **When to use**: Screen reader compatibility, keyboard navigation, assistive technology
- **Tools**: Read, Write, MultiEdit, Bash, axe, wave, nvda, jaws, voiceover, lighthouse, pa11y

### Performance & Debugging Agents

#### 78. **performance-engineer**
- **Expertise**: System optimization, bottleneck identification, scalability engineering
- **When to use**: Performance testing, profiling, tuning for optimal response times
- **Tools**: Read, Grep, jmeter, gatling, locust, newrelic, datadog, prometheus, perf, flamegraph

#### 79. **performance-monitor**
- **Expertise**: System-wide metrics collection, analysis, optimization
- **When to use**: Real-time monitoring, anomaly detection, performance insights
- **Tools**: Read, Write, MultiEdit, Bash, prometheus, grafana, datadog, elasticsearch, statsd

#### 80. **debugger**
- **Expertise**: Complex issue diagnosis, root cause analysis, systematic problem-solving
- **When to use**: Debugging tools and techniques across multiple languages/environments
- **Tools**: Read, Grep, Glob, gdb, lldb, chrome-devtools, vscode-debugger, strace, tcpdump

#### 81. **error-detective**
- **Expertise**: Complex error pattern analysis, correlation, root cause discovery
- **When to use**: Distributed system debugging, error tracking, anomaly detection
- **Tools**: Read, Grep, Glob, elasticsearch, datadog, sentry, loggly, splunk

#### 82. **error-coordinator**
- **Expertise**: Distributed error handling, failure recovery, system resilience
- **When to use**: Error correlation, cascade prevention, automated recovery strategies
- **Tools**: Read, Write, MultiEdit, Bash, sentry, pagerduty, error-tracking, circuit-breaker

#### 83. **chaos-engineer**
- **Expertise**: Controlled failure injection, resilience testing, antifragile systems
- **When to use**: Chaos experiments, game day planning, continuous resilience improvement
- **Tools**: Read, Write, MultiEdit, Bash, chaostoolkit, litmus, gremlin, pumba, powerfulseal, chaosblade

### Business & Product Agents

#### 84. **product-manager**
- **Expertise**: Product strategy, user-centric development, business outcomes
- **When to use**: Roadmap planning, feature prioritization, cross-functional leadership
- **Tools**: jira, productboard, amplitude, mixpanel, figma, slack

#### 85. **business-analyst**
- **Expertise**: Requirements gathering, process improvement, data-driven decisions
- **When to use**: Stakeholder management, business process modeling, solution design
- **Tools**: excel, sql, tableau, powerbi, jira, confluence, miro

#### 86. **sales-engineer**
- **Expertise**: Technical pre-sales, solution architecture, proof of concepts
- **When to use**: Technical demonstrations, competitive positioning, business value translation
- **Tools**: Read, Write, MultiEdit, Bash, salesforce, demo-tools, docker, postman, zoom

#### 87. **customer-success-manager**
- **Expertise**: Customer retention, growth, and advocacy
- **When to use**: Account health monitoring, strategic relationship building, value realization
- **Tools**: Read, Write, MultiEdit, Bash, salesforce, zendesk, intercom, gainsight, mixpanel

#### 88. **content-marketer**
- **Expertise**: Content strategy, SEO optimization, engagement-driven marketing
- **When to use**: Multi-channel content creation, analytics, conversion optimization
- **Tools**: wordpress, hubspot, buffer, canva, semrush, analytics

#### 89. **scrum-master**
- **Expertise**: Agile transformation, team facilitation, continuous improvement
- **When to use**: Scrum framework implementation, impediment removal, high-performing teams
- **Tools**: Read, Write, MultiEdit, Bash, jira, confluence, miro, slack, zoom, azure-devops

#### 90. **project-manager**
- **Expertise**: Project planning, execution, and delivery
- **When to use**: Resource management, risk mitigation, stakeholder communication
- **Tools**: jira, asana, monday, ms-project, slack, zoom

### Specialized Domain Agents

#### 91. **fintech-engineer**
- **Expertise**: Financial systems, regulatory compliance, secure transaction processing
- **When to use**: Banking integrations, payment systems, regulatory requirements
- **Tools**: Read, Write, MultiEdit, Bash, python, java, kafka, redis, postgresql, kubernetes

#### 92. **blockchain-developer**
- **Expertise**: Smart contract development, DApp architecture, DeFi protocols
- **When to use**: Solidity, Web3 integration, blockchain security
- **Tools**: truffle, hardhat, web3, ethers, solidity, foundry

#### 93. **payment-integration**
- **Expertise**: Payment gateway integration, PCI compliance, transaction processing
- **When to use**: Secure payment flows, multi-currency support, fraud prevention
- **Tools**: stripe, paypal, square, razorpay, braintree

#### 94. **iot-engineer**
- **Expertise**: Connected device architectures, edge computing, IoT platforms
- **When to use**: IoT protocols, device management, data pipelines
- **Tools**: mqtt, aws-iot, azure-iot, node-red, mosquitto

#### 95. **embedded-systems**
- **Expertise**: Microcontroller programming, RTOS development, hardware optimization
- **When to use**: Low-level programming, real-time constraints, resource-limited environments
- **Tools**: gcc-arm, platformio, arduino, esp-idf, stm32cube

#### 96. **game-developer**
- **Expertise**: Game engine programming, graphics optimization, multiplayer systems
- **When to use**: Game design patterns, performance optimization, cross-platform development
- **Tools**: unity, unreal, godot, phaser, pixi, three.js

### Financial & Analytics Agents

#### 97. **risk-manager**
- **Expertise**: Risk assessment, mitigation strategies, compliance frameworks
- **When to use**: Risk modeling, stress testing, regulatory compliance
- **Tools**: python, R, matlab, excel, sas, sql, tableau

#### 98. **quant-analyst**
- **Expertise**: Financial modeling, algorithmic trading, risk analytics
- **When to use**: Statistical methods, derivatives pricing, high-frequency trading
- **Tools**: python, numpy, pandas, quantlib, zipline, backtrader

### Mobile & Cross-Platform Agents

#### 99. **mobile-developer**
- **Expertise**: Cross-platform mobile development with React Native and Flutter
- **When to use**: Optimized mobile applications with platform-specific excellence
- **Tools**: Read, Write, MultiEdit, Bash, adb, xcode, gradle, cocoapods, fastlane

#### 100. **mobile-app-developer**
- **Expertise**: Native and cross-platform development for iOS and Android
- **When to use**: Performance optimization, platform guidelines, exceptional mobile experiences
- **Tools**: Read, Write, MultiEdit, Bash, xcode, android-studio, flutter, react-native, fastlane

#### 101. **electron-pro**
- **Expertise**: Desktop application development with Electron
- **When to use**: Cross-platform desktop apps with native OS integration
- **Tools**: Read, Write, MultiEdit, Bash, electron-forge, electron-builder, node-gyp, codesign, notarytool

### Developer Tools & Workflow Agents

#### 102. **dx-optimizer**
- **Expertise**: Build performance, tooling efficiency, workflow automation
- **When to use**: Development environment optimization, reducing friction, maximizing productivity
- **Tools**: webpack, vite, turbo, nx, rush, lerna, bazel

#### 103. **git-workflow-manager**
- **Expertise**: Branching strategies, automation, team collaboration
- **When to use**: Git workflows, merge conflict resolution, repository management
- **Tools**: git, github-cli, gitlab, gitflow, pre-commit

#### 104. **cli-developer**
- **Expertise**: Command-line interface design, developer tools, terminal applications
- **When to use**: User experience, cross-platform compatibility, CLI tools
- **Tools**: Read, Write, MultiEdit, Bash, commander, yargs, inquirer, chalk, ora, blessed

#### 105. **dependency-manager**
- **Expertise**: Package management, security auditing, version conflict resolution
- **When to use**: Dependency optimization, supply chain security, automated updates
- **Tools**: npm, yarn, pip, maven, gradle, cargo, bundler, composer

#### 106. **build-engineer**
- **Expertise**: Build system optimization, compilation strategies, developer productivity
- **When to use**: Modern build tools, caching mechanisms, fast build pipelines
- **Tools**: Read, Write, MultiEdit, Bash, webpack, vite, rollup, esbuild, turbo, nx, bazel

#### 107. **tooling-engineer**
- **Expertise**: Developer tool creation, CLI development, productivity enhancement
- **When to use**: Tool architecture, plugin systems, user experience design
- **Tools**: node, python, go, rust, webpack, rollup, esbuild

### Documentation & Writing Agents

#### 108. **technical-writer**
- **Expertise**: Clear, accurate documentation and content creation
- **When to use**: API documentation, user guides, technical content
- **Tools**: markdown, asciidoc, confluence, gitbook, mkdocs

#### 109. **documentation-engineer**
- **Expertise**: Technical documentation systems, API documentation
- **When to use**: Documentation-as-code, automated generation, maintainable documentation
- **Tools**: Read, Write, MultiEdit, Bash, markdown, asciidoc, sphinx, mkdocs, docusaurus, swagger

#### 110. **api-documenter**
- **Expertise**: Comprehensive, developer-friendly API documentation
- **When to use**: OpenAPI/Swagger specifications, interactive documentation portals
- **Tools**: swagger, openapi, postman, insomnia, redoc, slate

### Architecture & Design Agents

#### 111. **architect-reviewer**
- **Expertise**: System design validation, architectural patterns, technical decisions
- **When to use**: Scalability analysis, technology stack evaluation, evolutionary architecture
- **Tools**: Read, plantuml, structurizr, archunit, sonarqube

#### 112. **microservices-architect**
- **Expertise**: Distributed systems, scalable microservice ecosystems
- **When to use**: Service boundaries, communication patterns, operational excellence
- **Tools**: Read, Write, MultiEdit, Bash, kubernetes, istio, consul, kafka, prometheus

#### 113. **api-designer**
- **Expertise**: Scalable, developer-friendly API interfaces
- **When to use**: REST and GraphQL APIs with comprehensive documentation
- **Tools**: Read, Write, MultiEdit, Bash, openapi-generator, graphql-codegen, postman, swagger-ui, spectral

#### 114. **fullstack-developer**
- **Expertise**: End-to-end feature development across entire stack
- **When to use**: Complete solutions from database to UI
- **Tools**: Read, Write, MultiEdit, Bash, Docker, database, redis, postgresql, magic, context7, playwright

#### 115. **graphql-architect**
- **Expertise**: GraphQL schema design, efficient API graphs
- **When to use**: Federation, subscriptions, query optimization, type safety
- **Tools**: Read, Write, MultiEdit, Bash, apollo-rover, graphql-codegen, dataloader, graphql-inspector, federation-tools

#### 116. **websocket-engineer**
- **Expertise**: Real-time communication, scalable WebSocket architectures
- **When to use**: Bidirectional protocols, event-driven systems, low-latency messaging
- **Tools**: Read, Write, MultiEdit, Bash, socket.io, ws, redis-pubsub, rabbitmq, centrifugo

### Network & Infrastructure Agents

#### 117. **network-engineer**
- **Expertise**: Cloud and hybrid network architectures, security, performance
- **When to use**: Network design, troubleshooting, automation, zero-trust principles
- **Tools**: Read, Write, MultiEdit, Bash, tcpdump, wireshark, nmap, iperf, netcat, dig, traceroute

#### 118. **incident-responder**
- **Expertise**: Security and operational incident management
- **When to use**: Evidence collection, forensic analysis, coordinated response
- **Tools**: Read, Write, MultiEdit, Bash, pagerduty, opsgenie, victorops, slack, jira, statuspage

#### 119. **devops-incident-responder**
- **Expertise**: Rapid detection, diagnosis, resolution of production issues
- **When to use**: Observability tools, root cause analysis, automated remediation
- **Tools**: Read, Write, MultiEdit, Bash, pagerduty, slack, datadog, kubectl, aws-cli, jq, grafana

### Research & Analysis Agents

#### 120. **market-researcher**
- **Expertise**: Market analysis, consumer insights, competitive intelligence
- **When to use**: Market sizing, segmentation, trend analysis
- **Tools**: Read, Write, WebSearch, survey-tools, analytics, statista, similarweb

#### 121. **search-specialist**
- **Expertise**: Advanced information retrieval, query optimization, knowledge discovery
- **When to use**: Finding needle-in-haystack information across diverse sources
- **Tools**: Read, Write, WebSearch, Grep, elasticsearch, google-scholar, specialized-databases

#### 122. **trend-analyst**
- **Expertise**: Identifying emerging patterns, forecasting developments, strategic foresight
- **When to use**: Trend detection, impact analysis, scenario planning
- **Tools**: Read, Write, WebSearch, google-trends, social-listening, data-visualization

#### 123. **competitive-analyst**
- **Expertise**: Competitor intelligence, strategic analysis, market positioning
- **When to use**: Competitive benchmarking, SWOT analysis, strategic recommendations
- **Tools**: Read, Write, WebSearch, WebFetch, similarweb, semrush, crunchbase

#### 124. **data-researcher**
- **Expertise**: Discovering, collecting, analyzing diverse data sources
- **When to use**: Data mining, statistical analysis, pattern recognition
- **Tools**: Read, Write, sql, python, pandas, WebSearch, api-tools

#### 125. **research-analyst**
- **Expertise**: Comprehensive information gathering, synthesis, insight generation
- **When to use**: Research methodologies, data analysis, report creation
- **Tools**: Read, Write, WebSearch, WebFetch, Grep

### Legal & Compliance Agents

#### 126. **legal-advisor**
- **Expertise**: Technology law, compliance, risk mitigation
- **When to use**: Contract drafting, intellectual property, data privacy, regulatory compliance
- **Tools**: markdown, latex, docusign, contract-tools

### UX & Design Agents

#### 127. **ux-researcher**
- **Expertise**: User insights, usability testing, data-driven design decisions
- **When to use**: Qualitative and quantitative research methods for user needs validation
- **Tools**: Read, Write, MultiEdit, Bash, figma, miro, usertesting, hotjar, maze, airtable

### Code Refactoring & Legacy Agents

#### 128. **refactoring-specialist**
- **Expertise**: Safe code transformation, design pattern application
- **When to use**: Improving code structure, reducing complexity, enhancing maintainability
- **Tools**: ast-grep, semgrep, eslint, prettier, jscodeshift

#### 129. **legacy-modernizer**
- **Expertise**: Incremental migration strategies, risk-free modernization
- **When to use**: Transforming legacy systems into modern architectures
- **Tools**: ast-grep, jscodeshift, rector, rubocop, modernizr

### Multi-Agent Coordination Agents

#### 130. **agent-organizer**
- **Expertise**: Multi-agent orchestration, team assembly, workflow optimization
- **When to use**: Task decomposition, agent selection, coordination strategies
- **Tools**: Read, Write, agent-registry, task-queue, monitoring

#### 131. **workflow-orchestrator**
- **Expertise**: Complex process design, state machine implementation, business process automation
- **When to use**: Workflow patterns, error compensation, transaction management
- **Tools**: Read, Write, workflow-engine, state-machine, bpmn

#### 132. **multi-agent-coordinator**
- **Expertise**: Complex workflow orchestration, inter-agent communication, distributed coordination
- **When to use**: Parallel execution, dependency management, fault tolerance
- **Tools**: Read, Write, message-queue, pubsub, workflow-engine

#### 133. **task-distributor**
- **Expertise**: Intelligent work allocation, load balancing, queue management
- **When to use**: Priority scheduling, capacity tracking, fair distribution
- **Tools**: Read, Write, task-queue, load-balancer, scheduler

#### 134. **context-manager**
- **Expertise**: Information storage, retrieval, synchronization across multi-agent systems
- **When to use**: State management, version control, data lifecycle
- **Tools**: Read, Write, redis, elasticsearch, vector-db

#### 135. **knowledge-synthesizer**
- **Expertise**: Extracting insights from multi-agent interactions, pattern identification
- **When to use**: Cross-agent learning, best practice extraction, continuous improvement
- **Tools**: Read, Write, MultiEdit, Bash, vector-db, nlp-tools, graph-db, ml-pipeline

### Agent Selection Best Practices

1. **Use framework-specific agents** when available (e.g., rails-backend-expert for Rails, django-backend-expert for Django)
2. **Chain agents appropriately**: project-analyst → team-configurator → tech-lead-orchestrator → specific implementation agents
3. **Use PROACTIVELY** when agent descriptions indicate proactive use
4. **Launch multiple agents concurrently** when tasks are independent
5. **Trust agent outputs** - they are optimized for their specific domains
6. **For this GPT-Trader project specifically**:
   - Use backend-developer for Python ML/trading logic
   - Use trading-strategy-consultant for strategy validation
   - Use performance-optimizer for backtest optimization
   - Use code-archaeologist before major refactors

## Code Style & Best Practices

1. **Always use type hints** for function parameters and returns
2. **Document with docstrings** (Google style)
3. **Handle errors gracefully** with try/except blocks
4. **Log important events** using structured logging
5. **Write comprehensive tests** (target 90% coverage)
6. **Use task IDs** for tracking (e.g., DL-001, RL-015)
7. **Implement in phases** with validation checkpoints
8. **Shadow mode first** before production deployment


### Repository Cleanup & Single-Source-of-Truth (SoT) Program Focus

**Current Priority**: Before continuing Phase 4 ML development, the repository requires systematic cleanup and standardization through the SoT Program (SOT-001 to SOT-053).
