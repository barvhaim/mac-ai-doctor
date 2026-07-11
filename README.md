# 🩺 Mac AI Doctor

**Will this AI model fit in your Apple Silicon Mac's unified memory?**

`maid` reads small metadata files—never model weights—and estimates weight, KV-cache, and runtime
memory as a range, then gives a plain verdict.

```text
$ maid check ibm-granite/granite-switch-4.1-3b-preview --memory-gb 16

     ibm-granite/granite-switch-4.1-3b-preview
┌────────────┬────────────────┐
│ Weights    │        8.80 GB │
│ KV cache   │        0.34 GB │
│ Runtime    │        1.31 GB │
│ Peak range │ 11.49–13.06 GB │
│ Available  │        16.0 GB │
└────────────┴────────────────┘
╭───────────────────── TIGHT · high confidence ──────────────────────╮
│ May fit, but close memory-heavy apps or reduce context/concurrency. │
╰─────────────────────────────────────────────────────────────────────╯
```

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv tool install mac-ai-doctor          # CLI only
uv tool install 'mac-ai-doctor[web]'    # also enables `maid web`
```

## Use

```bash
maid web                                                 # interactive web UI (needs the 'web' extra)
maid system                                              # your chip and memory
maid check meta-llama/Llama-3.1-8B-Instruct              # a Hugging Face model
maid check ~/Models/model-q4.gguf --context 8192         # a local GGUF
maid compare org/model-a org/model-b --concurrency 2     # side by side
maid recommend --memory-gb 16 --task coding              # a starting point
maid check MODEL --json                                  # machine-readable
```

Accepts a Hugging Face ID, a local `.gguf` file, or a local/MLX directory with `config.json`.

**Options:** `--memory-gb N` (required off macOS), `--context`, `--concurrency`,
`--kv-dtype` (`fp32`/`fp16`/`bf16`/`int8`/`q8`), `--json`. Tasks: `coding`, `chat`, `vision`.

## Verdicts

| Verdict | Meaning |
| --- | --- |
| **COMFORTABLE** | High estimate ≤ 80% of memory. |
| **TIGHT** | Fits, but leaves < 20% headroom. |
| **UNLIKELY** | Exceeds memory. |
| **UNKNOWN** | Weight size unavailable. |

Confidence reflects metadata completeness, not prediction accuracy.

## What it reads

Only bounded metadata: `config.json`, the safetensors index, and the model API's file-size
listing—or the 24-byte header of a local GGUF. **Weight contents are never downloaded.** No
credentials are collected.

## How it estimates

Decimal GB. This is screening, not a benchmark—verify with your actual runtime. **No tokens/second
prediction is made.**

```text
weights = stored_weight_bytes × 1.06
KV      = 2 × layers × KV_heads × head_dim × context × concurrency × dtype_bytes
runtime = max(1 GB, weights × 12%) + 0.25 GB × concurrency
range   = subtotal × 1.10 .. subtotal × 1.25
```

Unified memory is shared with macOS, apps, and the GPU. Memory mapping can help; multimodal image
encoders may add memory not shown in text config metadata.

## Troubleshooting

- **Can't detect memory** — pass `--memory-gb N` (auto-detection needs macOS).
- **401/403 or gated** — authenticate, or point at a downloaded `config.json`.
- **No weight size** — use a repo with a safetensors index/API sizes, or a local GGUF (v2/v3).

## Develop

```bash
git clone https://github.com/barvhaim/mac-ai-doctor.git && cd mac-ai-doctor
uv sync --group dev
uv run ruff check . && uv run mypy src && uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.
