# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses [uv](https://docs.astral.sh/uv/). All commands run through `uv run`.

```bash
uv sync --group dev                          # install deps (including dev group)
uv run maid check <model> --memory-gb 16     # run the CLI in dev
uv run ruff check .                          # lint (also serves as import sorting via rule "I")
uv run mypy src                              # type-check (strict mode)
uv run pytest                                # run tests
uv run pytest tests/test_core.py::test_cli_check_json   # run a single test
uv run pytest --cov=mac_ai_doctor            # tests with coverage (as CI runs them)
uv build                                     # build wheel/sdist
```

The full CI gate (see `.github/workflows/ci.yml`) is: `ruff check` → `mypy src` → `pytest --cov` → `uv build`, across Python 3.11/3.12 on Linux and macOS. `uv sync --locked` is used in CI, so keep `uv.lock` committed and current.

## Architecture

A Typer CLI (entry point `mac_ai_doctor.cli:app`, exposed as both `maid` and `mac-ai-doctor`) that estimates whether a local AI model fits in a Mac's unified memory. The data flows through three layers, all pure and stateless:

1. **`metadata.py` — resolve a model reference into a `ModelInfo`.** `resolve_model()` dispatches on the input string: existing `.gguf` path → `resolve_gguf` (reads only the 24-byte header + file size), existing dir/file → local `config.json`, otherwise → `resolve_hf` (HTTP to Hugging Face). A hidden `--fixture` path routes to `resolve_fixture` for deterministic offline tests. All config-based sources funnel through `_from_config`, which normalizes the many naming conventions (`hidden_size`/`d_model`/`dim`, etc.) via the `_first()` helper into a single `ModelInfo`.

2. **`estimate.py` — turn `ModelInfo` + workload into an `Estimate`.** `estimate()` is the single source of truth for the memory formula and verdict policy (documented in README "Formula and policy"). It falls back to estimating weights from parameter count when byte size is unavailable, and returns `Verdict.UNKNOWN` when weight size cannot be determined at all. Any change to the numbers here should be reflected in the README formula block and the JSON assumptions strings.

3. **`cli.py` — commands and rendering.** Five commands: `check`, `compare`, `recommend`, `system`, `web`. `_evaluate()` wraps resolve+estimate and converts all exceptions into `typer.BadParameter`. Every command supports `--json`; human output uses Rich tables/panels. `_memory()` resolves the memory budget from `--memory-gb` or `detect_system()`. `build_check_renderables()` builds the shared table/verdict-panel/disclaimer for a single estimate. `VERDICT_GUIDANCE` and `DISCLAIMER` are the copy strings the web UI also reads. The `web` command shells out to `streamlit run` on `webapp.py`.

**`webapp.py`** is an optional Streamlit app (`maid web`) that mirrors `maid check` interactively — a form (model ref + memory/context/concurrency/kv-dtype) whose results reuse the same `resolve_model` → `estimate` path. It renders with native Streamlit widgets (Rich renderables don't display in a browser) but reads the shared `VERDICT_GUIDANCE`/`DISCLAIMER` copy from `cli.py` so wording stays in sync. `streamlit` is an optional `[web]` extra, not a core dependency, so the CLI stays lightweight; the `web` command launches it via `subprocess` → `streamlit run` and gives an install hint if the extra is missing.

**`models.py`** holds the three dataclasses (`SystemInfo`, `ModelInfo`, `Estimate`) and the `Verdict` enum. `to_dict()` stamps `schema_version: "1.0"` — this is the stable JSON contract; fields that can't be computed are `null`, never invented.

**`system.py`** detects Apple Silicon and unified memory via `sysctl` on Darwin only; returns a friendly non-macOS `SystemInfo` (with `memory_gb=None`) elsewhere, which is why `--memory-gb` is required off-macOS.

## Conventions and invariants

- **Never download model weights.** Only bounded metadata: `config.json`, the safetensors index, the HF API file-size listing, or a GGUF header. This is a core promise of the tool (see README and CONTRIBUTING) — keep reads bounded.
- **No tokens/second predictions.** Deliberately out of scope; do not add throughput/latency claims without reproducible hardware benchmarking.
- Tests must run without macOS or network access — use fixtures (`tests/fixtures/*.json`, shape defined by `resolve_fixture`) and `tmp_path`, and drive the CLI via Typer's `CliRunner`.
- `strict` mypy and `ruff` line length 100 are enforced; all modules use `from __future__ import annotations`.
