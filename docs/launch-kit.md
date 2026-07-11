# Launch kit

## Show HN

**Title:** Show HN: Mac AI Doctor - check if a Hugging Face model fits before downloading it

I built Mac AI Doctor because parameter count and download size do not account for KV cache, context length, concurrency and runtime overhead on unified-memory Macs.

It reads bounded metadata, never model weights, and returns a transparent range instead of claiming benchmark precision. It accepts Hugging Face URLs/IDs, MLX directories and local GGUF files.

I would especially value measured-versus-estimated results from different Apple Silicon configurations, including cases where the estimate is wrong.

## Reddit / r/LocalLLaMA

**Title:** I kept downloading models that did not fit my Mac, so I built a metadata-only fit checker

File size alone did not answer the question because context length, KV cache, concurrency and runtime allocations matter. Mac AI Doctor inspects the exact model repository without downloading its weights and shows every assumption behind the verdict.

The project deliberately does not predict tokens per second. I am looking for falsification: if you have an Apple Silicon Mac, please compare an estimate with a real MLX, llama.cpp, Ollama or LM Studio run and report the result using the validation template.

## LinkedIn hook

A model's parameter count does not tell you whether it will fit on your Mac.

The missing pieces are KV cache, context length, concurrency, runtime overhead and the fact that macOS shares the same unified memory. I built Mac AI Doctor to inspect the exact model metadata and give a transparent range before the weights are downloaded.

## Launch rule

Lead with the technical insight and demonstration. Ask for measured feedback, not stars.
