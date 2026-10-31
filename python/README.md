# potakt (Python package)

Python core of Potakt. See the [repository README](../README.md) and
[`docs/DESIGN.md`](../docs/DESIGN.md).

Development (from this directory):

```
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -q --cov=potakt            # fails below 95% line coverage
uv run --python 3.12 pytest -q          # oldest supported Python, as in CI
```
