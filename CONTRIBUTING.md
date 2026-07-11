# Contributing

Thanks for helping make fit estimates more useful without making them sound more certain.

1. Create a focused branch and add a test that demonstrates the behavior.
2. Keep metadata reads bounded and never download model weights.
3. Install [uv](https://docs.astral.sh/uv/), run `uv sync --group dev`, then run
   `uv run ruff check .`, `uv run mypy src`, `uv run pytest`, and `uv build`.
4. Explain estimator/policy changes and their assumptions in the pull request.

Use Python 3.11+, type hints, small functions, and fixture-driven tests that work without macOS or
network access. Do not add tokens/second claims without reproducible hardware/runtime benchmarking;
that is deliberately outside this project's scope. By contributing, you agree your work is
licensed under MIT.