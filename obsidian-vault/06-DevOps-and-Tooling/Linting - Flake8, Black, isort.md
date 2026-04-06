# Linting - Flake8, Black, isort

## What Is Linting?

Linting is static code analysis that checks for style violations, potential errors, and code quality issues without running the code.

## Tools in This Project

### Flake8 (Linter)
Checks code against PEP 8 style guide:

```toml
# pyproject.toml
[tool.flake8]
max-line-length = 100
ignore = ["E203", "W503"]
exclude = [".venv", "data", "models"]
```

| Ignored Rule | What It Is | Why Ignored |
|-------------|-----------|-------------|
| E203 | Whitespace before `:` | Conflicts with Black's formatting |
| W503 | Line break before binary operator | Outdated rule, W504 preferred |

### Black (Formatter)
Opinionated code formatter - one way to format code, no debates:

```toml
[tool.black]
line-length = 100
target-version = ["py310"]
```

Black handles:
- Indentation, spacing, quotes (double)
- Line wrapping, trailing commas
- Consistent style across the entire codebase

### isort (Import Sorter)
Sorts and groups imports:

```toml
[tool.isort]
profile = "black"          # Compatible with Black
line_length = 100
```

Import order:
1. Standard library (`os`, `pathlib`)
2. Third-party (`pandas`, `sklearn`)
3. Local (`from churn.config import ...`)

## Why These Tools Matter

| Tool | Catches | Autofix? |
|------|---------|----------|
| Flake8 | Style violations, undefined names, unused imports | No (reports only) |
| Black | Inconsistent formatting | Yes (rewrites files) |
| isort | Unsorted imports | Yes (rewrites files) |

## Running Them

```bash
flake8 src tests           # Check style (CI does this)
black src tests            # Auto-format
isort src tests            # Sort imports
```

## Interview Relevance

Using linting tools demonstrates:
- **Code quality awareness** - Consistent, readable code
- **Team collaboration** - Automated style enforcement, no bikeshedding
- **CI integration** - Style checked automatically on every PR

---

**Related:** [[06-DevOps-and-Tooling/GitHub Actions CI-CD]] | [[06-DevOps-and-Tooling/Project Configuration - pyproject.toml]]
