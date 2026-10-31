# potakt (Python package)

Python core of Potakt. See the [repository README](../README.md) and
[`docs/DESIGN.md`](../docs/DESIGN.md).

Development (from this directory): `uv sync` once, then the six gates, equivalent to CI. CI runs
`uv sync --locked` and a separate Python 3.12 job; locally `uv lock --check` and `--isolated` give
the same checks.

```
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -q --cov=potakt
uv run --locked --isolated --python 3.12 pytest -q --cov=potakt   # as in CI; leaves .venv on 3.14
```

Both test runs fail below 95% line coverage.
