# Gemini Agent Notes

`docs/agents/Agents.md` is the canonical reference—use it for architecture, commands, and workflows.

## Gemini-Specific Tips
- Keep responses concise and enumerate follow-up actions so the human maintainer can respond with a single number when possible.
- Include the exact commands you ran (or recommend) with `rg`/`fd` snippets for context gathering; this keeps the assistant workflow reproducible.
- Call out environment prerequisites (`poetry install`, credentials) whenever you suggest running tests or scripts.

Check the Agent guide’s *Agent-Specific Notes* section for any new expectations that apply to all assistants.
