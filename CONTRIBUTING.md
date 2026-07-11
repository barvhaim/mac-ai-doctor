# Contributing

Thanks for helping make fit estimates more useful without making them sound more certain.

1. Create a focused branch and add a test that demonstrates the behavior.
2. Keep metadata reads bounded and never download model weights.
3. Run `ruff check .`, `mypy src`, `pytest`, and `python -m build`.
4. Explain estimator/policy changes and their assumptions in the pull request.

Use Python 3.11+, type hints, small functions, and fixture-driven tests that work without macOS or
network access. Do not add tokens/second claims without reproducible hardware/runtime benchmarking;
that is deliberately outside this project's scope. By contributing, you agree your work is
licensed under MIT.