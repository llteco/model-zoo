---
name: "code-lint"
description: "Use when linting or formatting Python code, fixing ruff issues, or validating code quality after edits."
---

# Code Linter

## Overview

Use this skill after Python code changes when you need to check style, catch lint errors, or format files.

## Preparation

Before linting codes, ensure that the following tools are installed:

- ruff

The tools can be installed using uv (preferred) or pip:

```bash
# If tools are defined in pyproject.toml
uv sync --dev
# If tools are not defined
uv add ruff --dev
```

## Steps:

1. Check code quality with ruff:

   ```bash
   uv run ruff check src
   ```

2. Fix reported issues in the changed files and rerun the check until it passes.

3. Format code when needed:

   ```bash
   uv run ruff format src
   ```
