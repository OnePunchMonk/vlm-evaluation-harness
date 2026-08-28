# Contributing to vlm-evaluation-harness

## Dev setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10+ is required.

## Running tests and lint

```bash
pytest                 # full test suite (offline, no API keys/GPU needed)
ruff check .            # lint
mypy src/vlm_evaluation_harness   # type check
```

All three must pass before a PR is merged. CI runs the same checks on
Python 3.10, 3.11, and 3.12.

Optionally, install [pre-commit](https://pre-commit.com) hooks to run ruff
automatically on every commit:

```bash
pip install pre-commit
pre-commit install
```

## Making a change

- Keep PRs focused: one logical change per PR.
- Add or update tests for any behavior change — the test suite is expected
  to run fully offline via the `mock` adapters, no network or API keys.
- Run `ruff check .` and `pytest` locally before opening a PR.
- If you're adding a new benchmark or adapter, see `docs/new-benchmark-guide.md`
  or `docs/new-adapter-guide.md`.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/`. Bug reports should
include the command you ran and the full traceback; feature requests should
describe the use case, not just the desired API.
