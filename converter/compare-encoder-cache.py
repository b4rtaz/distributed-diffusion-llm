#!/usr/bin/env python3
"""Compare persistent encoder caches produced with different chunk sizes."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


def load_trace_reader():
    path = Path(__file__).with_name("checkpoint-trace-test.py")
    spec = importlib.util.spec_from_file_location("ddllm_checkpoint_trace", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.read_trace


def compare(name, left, right, tolerance):
    if left.shape != right.shape:
        raise ValueError(f"{name} shape mismatch: {left.shape} != {right.shape}")
    max_error = max(abs(a - b) for a, b in zip(left.values, right.values))
    if not math.isfinite(max_error) or max_error > tolerance:
        raise ValueError(f"{name} differs across chunk sizes: max_abs={max_error:.9g}")
    qualification = "exact" if max_error == 0.0 else f"max_abs={max_error:.9g}"
    print(f"{name}: {qualification} ({len(left.values)} values)")


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} FIRST.ddrf SECOND.ddrf", file=sys.stderr)
        return 1
    read_trace = load_trace_reader()
    left = read_trace(Path(sys.argv[1]))
    right = read_trace(Path(sys.argv[2]))
    for fixture in (
        "fixture.encoder_tokens", "fixture.layer_type", "fixture.weight_type",
        "fixture.cache_capacity",
    ):
        if left.get(fixture) != right.get(fixture):
            raise ValueError(f"{fixture} mismatch")
    if left.get("fixture.encoder_chunk") == right.get("fixture.encoder_chunk"):
        raise ValueError("traces use the same encoder chunk size")
    weight_type = int(left["fixture.weight_type"].values[0])
    if weight_type not in (0, 2):
        raise ValueError("unsupported checkpoint weight type")
    tolerance = 1e-5 if weight_type == 2 else 0.0
    compare("encoder.key_cache", left["encoder.key_cache"], right["encoder.key_cache"], tolerance)
    compare("encoder.value_cache", left["encoder.value_cache"], right["encoder.value_cache"], tolerance)
    print("Encoder cache chunk-equivalence passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
