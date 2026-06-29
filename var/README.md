# Runtime Workspace

Most of this directory is local runtime state created by development, test, and
operator commands. Keep generated logs, databases, status files, dashboards, and
temporary outputs out of commits.

`var/agents/` is the tracked exception: it contains generated agent handoff
artifacts and must be refreshed with `uv run agent-regenerate` when its inputs
change.

Use `make clean-dry-run` to inspect safe local cleanup actions and `make clean`
to apply them.
