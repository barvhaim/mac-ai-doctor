# Mac AI Doctor Implementation Plan

**Goal:** Build an extremely easy-to-use CLI that tells Apple Silicon users whether a Hugging Face or local GGUF/MLX model is likely to fit in unified memory.

**Architecture:** Separate system detection, model metadata resolution, transparent memory estimation, verdict policy, and rendering. Support dependency injection and fixtures so all logic is testable on Linux while macOS CI verifies the user path.

## Tasks

1. Scaffold a Python 3.11+ package with `mac-ai-doctor` and short `maid` commands.
2. Implement Apple Silicon/system memory detection with graceful non-Mac diagnostics.
3. Resolve Hugging Face `config.json` and local GGUF metadata without downloading model weights.
4. Estimate weight, KV-cache, runtime overhead, safety margin, and peak-memory ranges with explicit assumptions and confidence.
5. Provide `system`, `check`, `compare`, and `recommend` commands with Rich output plus JSON mode.
6. Add tests, fixtures, macOS/Linux CI, README quickstart, formulas, limitations, examples, contributing guide, and MIT license.
7. Verify installation, lint, tests, CLI examples, wheel build, and remote CI.
