#!/usr/bin/env python3
"""Validate the self-describing DDRF checkpoint trace fixture."""

from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


MAGIC = 0x46524444
VERSION = 1


@dataclass(frozen=True)
class Record:
    shape: tuple[int, ...]
    values: tuple[float, ...]


def read_exact(source, size: int) -> bytes:
    data = source.read(size)
    if len(data) != size:
        raise ValueError("truncated DDRF checkpoint trace")
    return data


def read_trace(path: Path) -> dict[str, Record]:
    records: dict[str, Record] = {}
    with path.open("rb") as source:
        magic, version, count = struct.unpack("<III", read_exact(source, 12))
        if magic != MAGIC or version != VERSION:
            raise ValueError("unsupported DDRF checkpoint trace")
        for _ in range(count):
            name_length, rank, value_count = struct.unpack("<IIQ", read_exact(source, 16))
            if not name_length or not rank or rank > 8 or value_count == 0:
                raise ValueError("invalid DDRF record header")
            name = read_exact(source, name_length).decode("utf-8")
            shape = struct.unpack(f"<{rank}I", read_exact(source, rank * 4))
            if math.prod(shape) != value_count:
                raise ValueError(f"invalid DDRF shape for {name}")
            values = struct.unpack(f"<{value_count}f", read_exact(source, value_count * 4))
            if name in records:
                raise ValueError(f"duplicate DDRF record {name}")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite DDRF value in {name}")
            records[name] = Record(shape, values)
        if source.read(1):
            raise ValueError("trailing data in DDRF checkpoint trace")
    return records


def require(records: dict[str, Record], name: str, shape: tuple[int, ...] | None = None) -> Record:
    if name not in records:
        raise ValueError(f"missing DDRF record {name}")
    record = records[name]
    if shape is not None and record.shape != shape:
        raise ValueError(f"{name} has shape {record.shape}, expected {shape}")
    return record


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} TRACE.ddrf", file=sys.stderr)
        return 1
    records = read_trace(Path(sys.argv[1]))
    encoder_tokens = require(records, "fixture.encoder_tokens", (3,))
    require(records, "fixture.canvas_tokens", (3,))
    require(records, "fixture.sampling_coins", (3,))
    require(records, "fixture.layer_type", (1,))
    weight_type = require(records, "fixture.weight_type", (1,))
    if weight_type.values[0] not in (0.0, 2.0):
        raise ValueError("unexpected checkpoint trace weight type")
    if encoder_tokens.values != (2.0, 17.0, 42.0):
        raise ValueError("unexpected checkpoint trace fixture tokens")

    for prefix in ("encoder", "decoder0", "decoder1"):
        hidden = require(records, f"{prefix}.hidden")
        if hidden.shape[0] != 3:
            raise ValueError(f"{prefix}.hidden has the wrong query length")
        require(records, f"{prefix}.hard_embedding", hidden.shape)
        require(records, f"{prefix}.query")
        require(records, f"{prefix}.key")
        require(records, f"{prefix}.value")
        require(records, f"{prefix}.attention")
        require(records, f"{prefix}.attention_projection", hidden.shape)
        require(records, f"{prefix}.router_probabilities")
        require(records, f"{prefix}.expert_indexes")
        require(records, f"{prefix}.expert_weights")
        require(records, f"{prefix}.dense_output", hidden.shape)
        require(records, f"{prefix}.moe_output", hidden.shape)
        require(records, f"{prefix}.normalized_branches", (2,) + hidden.shape)
        require(records, f"{prefix}.combined_branches", hidden.shape)
        require(records, f"{prefix}.key_cache")
        require(records, f"{prefix}.value_cache")
        require(records, f"{prefix}.logits")
        require(records, f"{prefix}.self_conditioning", hidden.shape)
    require(records, "decoder0.samples", (3, 3))
    require(records, "decoder1.samples", (3, 3))

    encoder_key_cache = records["encoder.key_cache"]
    encoder_value_cache = records["encoder.value_cache"]
    if records["decoder0.key_cache"] != encoder_key_cache or records["decoder1.key_cache"] != encoder_key_cache:
        raise ValueError("decoder modified the persistent key cache")
    if records["decoder0.value_cache"] != encoder_value_cache or records["decoder1.value_cache"] != encoder_value_cache:
        raise ValueError("decoder modified the persistent value cache")
    print(f"Checkpoint trace tests passed: {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
