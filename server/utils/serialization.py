"""NumPy-aware serialization helpers for MCP responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from server.schemas.models import ArrayOutputConfig


def json_safe(value: Any) -> Any:
    """Recursively convert common scientific Python objects into JSON-safe data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def array_bundle(name: str, values: np.ndarray, output: ArrayOutputConfig) -> Dict[str, Any]:
    """Return metadata and optionally inline values for a scientific array."""
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        magnitude = np.abs(arr)
        min_value = float(np.min(magnitude)) if arr.size else None
        max_value = float(np.max(magnitude)) if arr.size else None
    else:
        min_value = float(np.min(arr)) if arr.size else None
        max_value = float(np.max(arr)) if arr.size else None
    payload: Dict[str, Any] = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": min_value,
        "max": max_value,
    }
    if output.mode == "inline" and output.include_values:
        payload["values"] = json_safe(arr)
    return payload


def write_npz_payload(file_path: str, arrays: Dict[str, Any]) -> str:
    """Persist arrays to an npz archive and return the absolute file path."""
    target = Path(file_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **arrays)
    return str(target)
