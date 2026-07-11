from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import httpx

from .models import ModelInfo

GGUF_TYPES = {0: 4, 1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 1, 8: 8, 10: 8, 11: 8, 12: 8}


def _first(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if config.get(key) is not None:
            return config[key]
    return None


def _from_config(
    model_id: str, source: str, config: dict[str, Any], weight_bytes: int | None
) -> ModelInfo:
    hidden = _first(config, "hidden_size", "d_model", "dim")
    heads = _first(config, "num_attention_heads", "n_head")
    layers = _first(config, "num_hidden_layers", "n_layer")
    kv = _first(config, "num_key_value_heads", "num_kv_heads") or heads
    params = _first(config, "num_parameters", "parameter_count")
    experts = config.get("num_local_experts") or config.get("num_experts")
    active_experts = config.get("num_experts_per_tok")
    active = (
        int(params * active_experts / experts) if params and experts and active_experts else params
    )
    quant = config.get("quantization_config", {})
    quant_name = quant.get("quant_method") if isinstance(quant, dict) else None
    architectures = config.get("architectures") or []
    return ModelInfo(
        source,
        model_id,
        architectures[0] if architectures else config.get("model_type"),
        int(params) if params else None,
        int(active) if active else None,
        weight_bytes,
        int(layers) if layers else None,
        int(heads) if heads else None,
        int(kv) if kv else None,
        int(hidden) if hidden else None,
        int(hidden // heads) if hidden and heads else None,
        quant_name,
        _first(config, "max_position_embeddings", "max_sequence_length"),
    )


def _normalize_hf_id(value: str) -> str:
    """Accept a bare ``owner/name`` ID or a pasted huggingface.co URL and return the ID."""
    text = value.strip()
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    # Drop any /tree/... /blob/... /resolve/... path suffix, query, or fragment.
    text = text.split("?", 1)[0].split("#", 1)[0]
    parts = text.strip("/").split("/")
    for marker in ("tree", "blob", "resolve"):
        if marker in parts:
            parts = parts[: parts.index(marker)]
            break
    return "/".join(parts)


def resolve_hf(model_id: str, timeout: float = 10) -> ModelInfo:
    model_id = _normalize_hf_id(model_id)
    base = f"https://huggingface.co/{model_id}/resolve/main"
    with httpx.Client(
        timeout=timeout, follow_redirects=True, headers={"User-Agent": "mac-ai-doctor/0.1"}
    ) as client:
        response = client.get(f"{base}/config.json")
        response.raise_for_status()
        config = response.json()
        weight_bytes = None
        idx = client.get(f"{base}/model.safetensors.index.json")
        if idx.status_code == 200:
            weight_bytes = idx.json().get("metadata", {}).get("total_size")
        if weight_bytes is None:
            api = client.get(f"https://huggingface.co/api/models/{model_id}")
            if api.status_code == 200:
                siblings = api.json().get("siblings", [])
                sizes = [
                    x.get("size", 0)
                    for x in siblings
                    if str(x.get("rfilename", "")).endswith((".safetensors", ".gguf"))
                ]
                weight_bytes = sum(sizes) or None
    return _from_config(model_id, "huggingface", config, weight_bytes)


def resolve_fixture(path: Path) -> ModelInfo:
    payload = json.loads(path.read_text())
    config = payload.get("config", payload)
    return _from_config(
        payload.get("model_id", path.stem), "fixture", config, payload.get("weight_bytes")
    )


def resolve_gguf(path: Path) -> ModelInfo:
    size = path.stat().st_size
    with path.open("rb") as fh:
        header = fh.read(24)
        if len(header) < 24 or header[:4] != b"GGUF":
            raise ValueError("not a valid GGUF file")
        version, tensors, kv_count = struct.unpack("<IQQ", header[4:])
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    return ModelInfo(
        "gguf",
        str(path),
        weight_bytes=size,
        quantization="GGUF",
        assumptions=[
            f"GGUF v{version}; {tensors} tensors, {kv_count} metadata entries",
            "Architecture metadata was not scanned to keep reads bounded.",
        ],
    )


def resolve_model(value: str, fixture: Path | None = None) -> ModelInfo:
    if fixture:
        return resolve_fixture(fixture)
    path = Path(value).expanduser()
    if path.exists():
        if path.suffix.lower() == ".gguf":
            return resolve_gguf(path)
        config_path = path / "config.json" if path.is_dir() else path
        return _from_config(str(path), "local", json.loads(config_path.read_text()), None)
    return resolve_hf(value)
