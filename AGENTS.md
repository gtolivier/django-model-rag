# AGENTS.md

Instructions for coding agents working in this repository. Read
[README.md](README.md) first for what this package is and is not.

## Commands

- Install: `uv sync`
- Test: `uv run pytest`
- Lint: `uv run ruff check` — format: `uv run ruff format`
- Type check: `uv run mypy` (strict, with the django-stubs plugin)

Always go through `uv run`; do not rely on an activated virtualenv.

## Layout

- `src/django_model_rag/` — the package (src layout: tests run against the
  installed package, not the working directory).
- `tests/` — pytest tests (`test_*.py`), run with pytest-django.
- `tests/settings.py` — minimal Django settings (in-memory SQLite).
- `tests/testapp/` — the test bench: models copied from the prototype
  (`Category` / `Product` for a plain model, `Page` / `TextPlugin` /
  `AccordionItem` for content scattered across related models). After
  changing them, regenerate the migration, then format it (Django's output
  does not pass ruff): `uv run django-admin makemigrations testapp
  --settings=tests.settings --pythonpath=. && uv run ruff format tests/testapp/migrations`.
  A test fails while the migration and the models disagree.

## Test-driven development

Features are built with the `/tdd:feature` skill of the
[`tdd` plugin](https://github.com/gtolivier/agent-workflows). Its conventions
for this repository:

- **Test, lint, type-check and format commands:** those of the Commands
  section above.
- **Test files:** everything under `tests/` — test modules and the test bench
  (`tests/settings.py`, `tests/testapp/`). Nothing outside `tests/` is a
  test file.

## Rules

- **Dependencies:** add or remove them with `uv add` / `uv remove`, never by
  editing `pyproject.toml` by hand — a manual edit leaves `uv.lock` stale.
- **Generated files:** when an official tool can produce a file
  (`django-admin startapp`, `manage.py makemigrations`, `uv init`…), use it
  instead of writing the file.
- **Test-first:** every behavior in `src/` is introduced by a failing test.
  An earlier prototype serves as the behavioral reference; do not copy its
  code — re-derive each function from a red test.
- **Clean code, enforced where a tool can:** ruff flags magic values in
  comparisons (numbers and strings, outside `tests/`), complexity, naming,
  unused arguments and commented-out code; mypy runs in strict mode, so
  every function is annotated. A `# noqa` or `# type: ignore` must name
  the error it silences (both tools enforce it) and come with a comment
  saying why.
- **Supported versions:** Python 3.11+, Django 5.2 LTS / 6.0 / 6.1. Keep the
  CI matrix in `.github/workflows/ci.yml` in sync when this changes.
- **No `CLAUDE.md`:** this file is the single source of agent instructions.
  A `CLAUDE.md` next to it would make Claude Code ignore it.
