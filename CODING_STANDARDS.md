# Coding Standards

## Packaging & dependency management

- [uv](https://docs.astral.sh/uv/) is the only tool used to manage dependencies, virtual environments, and running scripts/tests.
- Add dependencies with `uv add <package>` (or `uv add --dev <package>` for dev-only tools). Never hand-edit `dependencies`/`dependency-groups` in `pyproject.toml` unless uv can't express the change.
- Run project code and tools through `uv run ...` so the locked environment (`uv.lock`) is always what executes.
- Commit `uv.lock`.

## Testing

- [pytest](https://docs.pytest.org/) is the only test runner.
- Run tests with `uv run pytest`.
- Every new piece of code must ship with test coverage. No PR adds a function, class, or behavior without a corresponding test.

## Project structure

- Code lives in a package (not loose top-level modules), e.g. `src/jev_plays_pokemon/`.
- The `tests/` directory mirrors the package's internal structure 1:1: a module at `src/jev_plays_pokemon/foo/bar.py` is tested by `tests/foo/test_bar.py`.

## Linting & formatting

- [ruff](https://docs.astral.sh/ruff/) is the only linter and formatter.
- Run `uv run ruff check .` and `uv run ruff format .` before committing.

## Type checking

- [ty](https://github.com/astral-sh/ty) is the only type checker.
- Run `uv run ty check` before committing.

## Configuration

- All project and tool configuration lives in `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.ruff]`, `[tool.ty]`, etc.). Avoid separate config files (`pytest.ini`, `setup.cfg`, `ruff.toml`, ...) unless a tool cannot be configured from `pyproject.toml`.

## Comments

- Limit comments to cases where the code is not self-explanatory: a non-obvious constraint, a subtle invariant, or a workaround for a specific bug.
- Don't restate what the code already says through naming and structure.

## Repowise

- This repo's codebase health and documentation are monitored by [Repowise](https://repowise.dev). Keep code structured and named clearly enough for Repowise's index to stay accurate, and re-run `repowise update` after structural changes so docs/health scores stay current.
