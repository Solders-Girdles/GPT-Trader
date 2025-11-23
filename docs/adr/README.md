# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for GPT-Trader.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences. ADRs help future maintainers understand **why** certain design choices were made.

## When to Write an ADR

Create an ADR when making decisions about:
- System architecture and module organization
- Technology choices and patterns
- API design and interfaces
- Data models and persistence strategies
- Security and compliance approaches
- Trade-offs between competing concerns

**Rule of thumb:** If you're having a significant discussion about "should we do X or Y?", write an ADR.

## ADR Format

Each ADR follows this structure:

```markdown
# ADR-NNN: Short Title

**Status:** [Proposed | Accepted | Deprecated | Superseded]
**Date:** YYYY-MM-DD
**Deciders:** Who was involved in the decision
**Technical Story:** Context or issue that triggered this decision

## Context
What is the issue we're addressing? What are the constraints?

## Decision
What did we decide to do?

## Rationale
Why did we make this decision? What were the key factors?

## Consequences
What are the positive and negative outcomes?

## Alternatives Considered
What other options did we evaluate and why did we reject them?

## Related Decisions
Links to other ADRs or documentation

## References
Links to code, docs, or external resources
```

## ADR Lifecycle

- **Proposed:** Decision under discussion, not yet implemented
- **Accepted:** Decision implemented and in use
- **Deprecated:** Decision no longer recommended but still present in codebase
- **Superseded:** Decision replaced by a newer ADR (link to replacement)

## ADR Index

- [ADR-001: Dual Execution Engine Pattern](./ADR-001-dual-execution-engine-pattern.md) - Feature flag-based execution engine selection (Accepted, 2025-10-06)

## Further Reading

- [Michael Nygard's ADR Template](https://github.com/joelparkerhenderson/architecture-decision-record)
- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
