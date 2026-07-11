from __future__ import annotations

from .models import Estimate, ModelInfo, Verdict

GB = 1_000_000_000
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "q8": 1}


def estimate(
    model: ModelInfo,
    memory_gb: float,
    context: int = 4096,
    concurrency: int = 1,
    kv_dtype: str = "fp16",
) -> Estimate:
    assumptions = list(model.assumptions)
    weights = model.weight_bytes / GB * 1.06 if model.weight_bytes else None
    if weights is None and model.parameters:
        bits = 16
        q = (model.quantization or "").lower()
        for marker, value in (("4", 4), ("8", 8), ("fp32", 32)):
            if marker in q:
                bits = value
                break
        weights = model.parameters * bits / 8 / GB * (1.10 if bits <= 8 else 1.03)
        assumptions.append(f"Estimated weights from parameter count at {bits}-bit.")
    kv = None
    if model.layers and model.kv_heads and model.head_dim:
        kv = (
            2
            * model.layers
            * model.kv_heads
            * model.head_dim
            * context
            * concurrency
            * DTYPE_BYTES[kv_dtype]
            / GB
        )
    else:
        assumptions.append("KV cache unknown: architecture lacks layers/KV heads/head dimension.")
    if weights is None:
        return Estimate(
            model,
            memory_gb,
            context,
            concurrency,
            kv_dtype,
            None,
            kv,
            None,
            None,
            None,
            None,
            Verdict.UNKNOWN,
            "low",
            assumptions + ["Weight size is unavailable."],
        )
    kv_value = kv or 0
    runtime = max(1.0, weights * 0.12) + 0.25 * concurrency
    base = weights + kv_value + runtime
    low, high = base * 1.10, base * 1.25
    headroom = memory_gb - high
    verdict = (
        Verdict.COMFORTABLE
        if high <= memory_gb * 0.80
        else Verdict.TIGHT
        if high <= memory_gb
        else Verdict.UNLIKELY
    )
    confidence = (
        "high"
        if model.weight_bytes and kv is not None
        else "medium"
        if model.weight_bytes
        else "low"
    )
    assumptions += [
        "Weight storage includes 6% format/allocator overhead.",
        "Runtime reserve is max(1 GB, 12% of weights) plus 0.25 GB per concurrent request.",
        "Low/high totals apply 10%/25% safety margins; macOS and other apps share memory.",
    ]
    return Estimate(
        model,
        memory_gb,
        context,
        concurrency,
        kv_dtype,
        round(weights, 2),
        round(kv, 2) if kv is not None else None,
        round(runtime, 2),
        round(low, 2),
        round(high, 2),
        round(headroom, 2),
        verdict,
        confidence,
        assumptions,
    )
