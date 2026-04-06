# Project Configuration - pyproject.toml

## What Is It?

`pyproject.toml` is the standard Python project configuration file (PEP 518/621). It consolidates tool configuration that previously lived in separate files (`setup.cfg`, `.flake8`, `.isort.cfg`, etc.).

## This Project's Configuration

```toml
[tool.black]
line-length = 100
target-version = ["py310"]

[tool.isort]
profile = "black"
line_length = 100

[tool.flake8]
max-line-length = 100
ignore = ["E203", "W503"]
exclude = [".venv", "data", "models"]
```

## Key Sections

| Section | Purpose |
|---------|---------|
| `[tool.black]` | Black formatter config |
| `[tool.isort]` | Import sorting config |
| `[tool.flake8]` | Linting config |
| `[project]` | Package metadata (not used here) |
| `[build-system]` | Build backend (not used here) |

## Configuration Alignment

All three tools use `line-length = 100` for consistency. isort uses `profile = "black"` to avoid conflicts with Black's formatting.

## pyproject.toml vs Other Config Files

| Old Way | New Way (pyproject.toml) |
|---------|-------------------------|
| `setup.py` | `[project]` |
| `setup.cfg` | `[tool.setuptools]` |
| `.flake8` | `[tool.flake8]` |
| `.isort.cfg` | `[tool.isort]` |
| `MANIFEST.in` | `[tool.setuptools.package-data]` |

Single source of truth for all tool configuration.

## requirements.txt vs pyproject.toml

| | requirements.txt | pyproject.toml |
|--|-----------------|----------------|
| Purpose | Pin exact dependency versions | Declare project metadata + tool config |
| Used by | `pip install -r` | `pip install .` |
| Versions | Pinned (`pandas==2.2.2`) | Ranges (`pandas>=2.2`) |
| Best for | Reproducible environments | Library distribution |

This project uses `requirements.txt` for dependency pinning (application-style) and `pyproject.toml` for tool configuration.

---

**Related:** [[06-DevOps-and-Tooling/Linting - Flake8, Black, isort]] | [[00-Project-Overview/Project Architecture]]
