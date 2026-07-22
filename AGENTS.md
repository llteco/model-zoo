# Model Zoo Instructions

## Workflow

- Check relevant skills in `.claude/skills/` before handling specialized tasks.
- Prefer minimal, targeted changes that preserve existing project structure and style.
- Validate the files you change with the smallest relevant command before finishing.

## Python Changes

- Follow the existing Python style already used in `src/`.
- When Python files are modified, use the lint skill workflow and prefer `uv run ruff check src`.
- Format Python changes with `uv run ruff format src` when formatting is needed.

## Commits

- If asked to create a commit, use a clear and concise commit message that describes the actual change.
