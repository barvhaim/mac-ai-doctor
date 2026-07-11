from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    COMFORTABLE = "COMFORTABLE"
    TIGHT = "TIGHT"
    UNLIKELY = "UNLIKELY"
    UNKNOWN = "UNKNOWN"


@dataclass
class SystemInfo:
    platform: str
    chip: str | None
    model: str | None
    memory_gb: float | None
    supported: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelInfo:
    source: str
    model_id: str
    architecture: str | None = None
    parameters: int | None = None
    active_parameters: int | None = None
    weight_bytes: int | None = None
    layers: int | None = None
    attention_heads: int | None = None
    kv_heads: int | None = None
    hidden_size: int | None = None
    head_dim: int | None = None
    quantization: str | None = None
    context_limit: int | None = None
    assumptions: list[str] = field(default_factory=list)


@dataclass
class Estimate:
    model: ModelInfo
    available_gb: float
    context: int
    concurrency: int
    kv_dtype: str
    weights_gb: float | None
    kv_cache_gb: float | None
    runtime_gb: float | None
    low_gb: float | None
    high_gb: float | None
    headroom_gb: float | None
    verdict: Verdict
    confidence: str
    assumptions: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        data["schema_version"] = "1.0"
        return data
