# Validation

Mac AI Doctor is a screening tool, not a benchmark. This document defines how estimates should be checked against real Apple Silicon runs.

## Method

1. Record the Mac model, chip, unified memory, macOS version, runtime and runtime version.
2. Close unrelated memory-heavy applications and record baseline memory pressure.
3. Run `maid check MODEL --context N --concurrency N --kv-dtype TYPE --json`.
4. Load the same model and settings in MLX, llama.cpp, Ollama or LM Studio.
5. Record the highest observed process/resident memory after model load and one representative generation.
6. Submit the result using the validation issue template. Include failures and underestimates.

Do not compare the estimate with model file size alone. Runtime allocation, KV cache, memory mapping and shared system use affect observed memory.

## Public results

| Mac | Model | Runtime | Settings | Predicted peak | Measured peak | Result |
| --- | --- | --- | --- | ---: | ---: | --- |
| Contributions welcome | | | | | | |

No measured values are published until a reproducible run is submitted. This avoids presenting synthetic examples as evidence.

## Interpretation

- **Inside range:** measured peak falls between the low and high estimate.
- **Conservative:** measured peak is below the low estimate.
- **Underestimate:** measured peak exceeds the high estimate. This is the most important case to report.
- A correct fit verdict does not imply a useful tokens-per-second rate.
