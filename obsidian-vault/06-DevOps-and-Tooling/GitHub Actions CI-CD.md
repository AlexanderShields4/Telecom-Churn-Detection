# GitHub Actions CI/CD

## What Is It?

GitHub Actions is a CI/CD (Continuous Integration / Continuous Deployment) platform built into GitHub. It automates workflows like testing, linting, and deployment triggered by events (push, PR, schedule).

## This Project's Pipeline

`.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push:
    branches: [main, master]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: pip install flake8 pytest
      - run: flake8 src tests        # Linting
      - run: pytest -q || true        # Tests (non-blocking)
```

## Key Concepts

### Triggers (`on:`)
| Trigger | When |
|---------|------|
| `push` to main | Code merged to main branch |
| `pull_request` | PR opened, updated, or reopened |
| `schedule` | Cron-based (e.g., nightly builds) |
| `workflow_dispatch` | Manual trigger via GitHub UI |

### Jobs and Steps
- **Job**: A set of steps that run on a fresh VM (`ubuntu-latest`)
- **Step**: Individual command or action
- Jobs run in parallel by default; use `needs:` for dependencies

### Actions
Pre-built steps from the marketplace:
- `actions/checkout@v4` - Clones your repo
- `actions/setup-python@v5` - Installs Python

### Pipeline Steps in This Project
1. **Checkout** code
2. **Setup** Python 3.10
3. **Install** dependencies
4. **Lint** with flake8 (catches style issues before review)
5. **Test** with pytest (validates preprocessing logic)

### `|| true` on Tests
```yaml
- run: pytest -q || true  # Non-blocking
```
Tests won't fail the pipeline. In production, you'd remove `|| true` to enforce passing tests.

## CI/CD Best Practices

| Practice | This Project |
|----------|-------------|
| Run on every PR | Yes |
| Lint before test | Yes |
| Fast feedback loop | Yes (~1 min) |
| Pin Python version | Yes (3.10) |
| Pin dependencies | Yes (requirements.txt) |

## Common Interview Questions

**Q: What's the difference between CI and CD?**
A: **CI (Continuous Integration)**: Automatically build, lint, and test every code change. **CD (Continuous Deployment/Delivery)**: Automatically deploy to staging/production after CI passes. This project implements CI.

**Q: Why run linting in CI?**
A: Catches style inconsistencies and basic errors before code review. Enforces team standards automatically. Prevents "style nit" comments in PRs.

**Q: How would you extend this pipeline for ML?**
A: Add model validation steps: (1) run the full pipeline (preprocess → train → evaluate), (2) assert metrics meet thresholds, (3) compare against baseline model, (4) deploy model artifact.

---

**Related:** [[06-DevOps-and-Tooling/Linting - Flake8, Black, isort]] | [[06-DevOps-and-Tooling/Pytest]]
