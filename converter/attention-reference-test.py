#!/usr/bin/env python3
"""Compare randomized CPU diffusion attention with an independent PyTorch reference."""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch


ENCODER_CAUSAL = 0
DECODER_BIDIRECTIONAL = 1


@dataclass(frozen=True)
class Case:
    mode: int
    query_length: int
    query_heads: int
    kv_heads: int
    global_query_heads: int
    global_kv_heads: int
    query_head_start: int
    kv_head_start: int
    head_dim: int
    sliding_window: int
    cache_capacity: int
    prefix_length: int
    score_scale: float
    positions: torch.Tensor
    query: torch.Tensor
    current_key: torch.Tensor
    current_value: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor


def tensor_values(generator: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.4


def build_case(index: int, scenario: tuple[int, int], generator: torch.Generator) -> Case:
    sliding_window, prefix_length = scenario
    query_length = 1 + index % 4
    head_dim = (4, 8, 12)[index % 3]
    local_group = index % 3 != 0
    if local_group:
        query_heads = 2
        kv_heads = 1
        query_head_start = 2 if index % 2 else 0
        kv_head_start = query_head_start // 2
    else:
        query_heads = 4
        kv_heads = 2
        query_head_start = 0
        kv_head_start = 0
    cache_capacity = 8 if sliding_window == 0 else sliding_window
    position_start = prefix_length
    score_scale = (1.0, 0.5, 1.0 / math.sqrt(head_dim), 1.7)[index % 4]
    positions = torch.arange(
        position_start, position_start + query_length, dtype=torch.float32)
    query_dim = query_heads * head_dim
    kv_dim = kv_heads * head_dim
    return Case(
        mode=index % 2,
        query_length=query_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        global_query_heads=4,
        global_kv_heads=2,
        query_head_start=query_head_start,
        kv_head_start=kv_head_start,
        head_dim=head_dim,
        sliding_window=sliding_window,
        cache_capacity=cache_capacity,
        prefix_length=prefix_length,
        score_scale=score_scale,
        positions=positions,
        query=tensor_values(generator, query_length, query_dim),
        current_key=tensor_values(generator, query_length, kv_dim),
        current_value=tensor_values(generator, query_length, kv_dim),
        key_cache=tensor_values(generator, cache_capacity, kv_dim),
        value_cache=tensor_values(generator, cache_capacity, kv_dim),
    )


def reference(case: Case) -> torch.Tensor:
    output = torch.empty_like(case.query)
    group = case.global_query_heads // case.global_kv_heads
    for query_index in range(case.query_length):
        query_position = int(case.positions[query_index].item())
        for local_query_head in range(case.query_heads):
            global_query_head = case.query_head_start + local_query_head
            local_kv_head = global_query_head // group - case.kv_head_start
            q_start = local_query_head * case.head_dim
            kv_start = local_kv_head * case.head_dim
            query = case.query[query_index, q_start:q_start + case.head_dim]
            scores: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            prefix_start = 0
            if case.sliding_window:
                window_end = (
                    query_position + 1
                    if case.mode == ENCODER_CAUSAL
                    else case.prefix_length
                )
                prefix_start = max(0, window_end - case.sliding_window)
            prefix_end = case.prefix_length
            if case.mode == ENCODER_CAUSAL:
                prefix_end = min(prefix_end, query_position + 1)
            for absolute_position in range(prefix_start, prefix_end):
                slot = (
                    absolute_position
                    if case.sliding_window == 0
                    else absolute_position % case.cache_capacity
                )
                key = case.key_cache[slot, kv_start:kv_start + case.head_dim]
                value = case.value_cache[slot, kv_start:kv_start + case.head_dim]
                scores.append(torch.dot(query, key) * case.score_scale)
                values.append(value)
            for key_index in range(case.query_length):
                key_position = int(case.positions[key_index].item())
                if case.mode == ENCODER_CAUSAL and key_position > query_position:
                    continue
                if (
                    case.mode == ENCODER_CAUSAL
                    and case.sliding_window
                    and key_position < prefix_start
                ):
                    continue
                key = case.current_key[key_index, kv_start:kv_start + case.head_dim]
                value = case.current_value[key_index, kv_start:kv_start + case.head_dim]
                scores.append(torch.dot(query, key) * case.score_scale)
                values.append(value)
            probabilities = torch.softmax(torch.stack(scores), dim=0)
            result = torch.sum(probabilities[:, None] * torch.stack(values), dim=0)
            output[query_index, q_start:q_start + case.head_dim] = result
    return output


def flatten(tensor: torch.Tensor) -> list[float]:
    return tensor.contiguous().view(-1).tolist()


def write_case(path: Path, case: Case) -> None:
    header = [
        case.mode,
        case.query_length,
        case.query_heads,
        case.kv_heads,
        case.global_query_heads,
        case.global_kv_heads,
        case.query_head_start,
        case.kv_head_start,
        case.head_dim,
        case.sliding_window,
        case.cache_capacity,
        case.prefix_length,
        case.score_scale,
    ]
    values = (
        flatten(case.positions)
        + flatten(case.query)
        + flatten(case.current_key)
        + flatten(case.current_value)
        + flatten(case.key_cache)
        + flatten(case.value_cache)
    )
    path.write_text(
        "DDAF1\n"
        + " ".join(str(value) for value in header)
        + "\n"
        + " ".join(format(value, ".9g") for value in values)
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", type=Path)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--repetitions", type=int, default=8)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    probe = args.probe.resolve()
    if not probe.is_file():
        parser.error(f"probe does not exist: {probe}")

    scenarios = (
        (0, 0),
        (0, 2),
        (0, 6),
        (2, 0),
        (4, 2),
        (4, 4),
        (4, 10),
        (6, 13),
    )
    generator = torch.Generator().manual_seed(args.seed)
    maximum_error = 0.0
    case_count = len(scenarios) * args.repetitions * 2
    with tempfile.TemporaryDirectory(prefix="ddllm-attention-reference-") as directory:
        root = Path(directory)
        case_index = 0
        for repetition in range(args.repetitions):
            for scenario in scenarios:
                for mode in (ENCODER_CAUSAL, DECODER_BIDIRECTIONAL):
                    case = build_case(case_index, scenario, generator)
                    case = Case(**{**case.__dict__, "mode": mode})
                    input_path = root / "input.ddaf"
                    output_path = root / "output.txt"
                    write_case(input_path, case)
                    result = subprocess.run(
                        [str(probe), str(input_path), str(output_path)],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"attention probe failed for case {case_index}: {result.stderr.strip()}"
                        )
                    actual = torch.tensor(
                        [float(value) for value in output_path.read_text().split()],
                        dtype=torch.float32,
                    ).reshape(case.query.shape)
                    expected = reference(case)
                    error = torch.max(torch.abs(actual - expected)).item()
                    maximum_error = max(maximum_error, error)
                    if not torch.allclose(actual, expected, atol=3e-6, rtol=2e-5):
                        raise AssertionError(
                            f"case {case_index} failed: mode={mode}, scenario={scenario}, "
                            f"max_abs={error:.9g}"
                        )
                    case_index += 1
    assert case_index == case_count
    print(
        f"PyTorch attention reference passed: {case_count} cases, "
        f"max_abs={maximum_error:.9g}, seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
