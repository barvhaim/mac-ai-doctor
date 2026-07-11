# 🩺 Mac AI Doctor

**A transparent first opinion on whether a local AI model fits your Apple Silicon Mac.**

Mac AI Doctor (`maid`) reads small metadata files—never model weights—and estimates weight,
KV-cache, runtime, and safety-margin memory. It reports a range rather than fake precision.

```text
              demo/llama-8b
┌────────────────┬──────────────┐
│ Weights        │      4.77 GB │
│ KV cache       │      0.54 GB │
│ Runtime        │      1.25 GB │
│ Peak range     │ 7.21–8.20 GB │
│ Available      │     16.0 GB  │
└────────────────┴──────────────┘
╭──────── COMFORTABLE · high confidence ────────╮
│ Good fit. Keep normal apps modest.             │
╰────────────────────────────────────────────────╯
```

## Install

This project uses [uv](https://docs.astral.sh/uv/). Install it first, then use either a packaged
release or a checkout.

```bash
# Package install (once published)
uv tool install mac-ai-doctor
```

```bash
# Development checkout
git clone https://github.com/barvhaim/mac-ai-doctor.git
cd mac-ai-doctor
uv sync --group dev
```

## Use

```bash
uv run maid system
uv run maid check meta-llama/Llama-3.1-8B-Instruct
uv run maid check ~/Models/model-q4.gguf --memory-gb 16 --context 8192
uv run maid compare org/model-a org/model-b --memory-gb 32 --concurrency 2
uv run maid recommend --memory-gb 16 --task coding
uv run maid check MODEL --json
```

On Linux or when detection fails, pass `--memory-gb`. `--kv-dtype` accepts `fp32`, `fp16`,
`bf16`, `int8`, or `q8`. For deterministic offline demos/tests, `check` and `compare` include a
hidden `--fixture FILE` option using the documented fixture shape in `tests/fixtures/`.

## What it reads (and what it does not)

For Hugging Face and MLX repositories, the tool requests `config.json`, an optional
`model.safetensors.index.json`, and the model API's file-size listing. **It never downloads
weight contents.** For local GGUF files it reads only the 24-byte fixed header and uses file size.
Local directories/config files are also supported. Dense and MoE architecture fields are parsed;
MoE active parameters are informational because all expert weights still need memory.

No credentials are collected. Normal Hugging Face HTTP requests reveal the same network metadata
as any web request (IP and user-agent) to Hugging Face.

## Formula and policy

Decimal GB are used. When metadata permits:

```text
weights = stored_weight_bytes × 1.06
KV = 2 × layers × KV_heads × head_dim × context × concurrency × dtype_bytes
runtime = max(1 GB, weights × 12%) + 0.25 GB × concurrency
low..high = subtotal × 1.10 .. subtotal × 1.25
```

`COMFORTABLE` means the high estimate is at most 80% of unified memory; `TIGHT` fits but leaves
less than 20%; `UNLIKELY` exceeds memory; `UNKNOWN` means weight size is unavailable. Confidence
reflects metadata completeness—not prediction accuracy. Unified memory is shared by macOS, apps,
GPU, and model runtime. Implementations differ, memory mapping can help, and multimodal image
encoders may add memory not exposed in text config metadata. Treat this as screening, then verify
with your actual runtime. **The tool intentionally makes no tokens/second prediction.**

## JSON

`--json` emits schema version `1.0`, model metadata, inputs, component estimates, range, headroom,
verdict, confidence, and assumptions. Fields that cannot be supported are `null`, never invented.

## Troubleshooting

- **Cannot detect memory:** pass `--memory-gb N`; automatic detection requires macOS.
- **401/403 or gated model:** authenticate/access the model or use a downloaded `config.json`.
- **No weight size:** use a repository with safetensors index/API sizes or a local GGUF.
- **GGUF rejected:** versions 2 and 3 are supported; verify the file is complete.
- **Very long context looks large:** KV memory grows linearly with context and concurrency.

## Development

```bash
uv sync --group dev
uv run ruff check . && uv run mypy src && uv run pytest
uv build
```

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.