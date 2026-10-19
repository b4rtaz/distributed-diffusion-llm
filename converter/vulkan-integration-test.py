#!/usr/bin/env python3
"""Compare tiny DiffusionGemma CPU, Vulkan, and mixed-device CLI runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30.0,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}\n{result.stdout}"
        )
    return result.stdout


def root_command(executable: str, mode: str, model: str, tokenizer: str, sync_type: str) -> list[str]:
    return [
        executable,
        mode,
        "--model", model,
        "--tokenizer", tokenizer,
        "--nthreads", "1",
        "--nbatches", "4",
        "--max-seq-len", "128",
        "--seed", "42",
        "--denoising-steps", "1",
        "--buffer-float-type", sync_type,
    ]


def results_block(output: str) -> str:
    marker = "\nResults\n"
    start = output.rfind(marker)
    if start < 0:
        raise AssertionError(f"perplexity output is missing its results block\n{output}")
    return "\n".join(output[start:].splitlines()[:5])


def generated_text(output: str, prompt: str) -> str:
    start_marker = prompt + "\n"
    end_marker = "\n\nGeneration\n"
    start = output.rfind(start_marker)
    end = output.rfind(end_marker)
    if start < 0 or end < start:
        raise AssertionError(f"inference output is missing generation markers\n{output}")
    return output[start + len(start_marker):end]


def qualify_model(executable: str, model: str, tokenizer: str, sync_type: str) -> None:
    cpu_perplexity = root_command(executable, "perplexity", model, tokenizer, sync_type)
    cpu_perplexity.extend(["--prompt", "hello hello"])
    expected_results = results_block(run(cpu_perplexity))

    for device_args in (["--gpu-index", "0"], ["--gpu-index", "0", "--gpu-segments", "0:3"]):
        output = run(cpu_perplexity + device_args)
        if results_block(output) != expected_results:
            raise AssertionError(f"CPU and Vulkan perplexity results differ\n{output}")
        if "Device: " not in output:
            raise AssertionError(f"Vulkan execution did not report a device\n{output}")
        if "--gpu-segments" in device_args and "CPU:" not in output:
            raise AssertionError(f"mixed execution did not initialize the CPU backend\n{output}")

    cpu_inference = root_command(executable, "inference", model, tokenizer, sync_type)
    cpu_inference.extend(["--prompt", "hello", "--steps", "10"])
    cpu_output = run(cpu_inference)
    expected_text = generated_text(cpu_output, "hello")
    for device_args in (["--gpu-index", "0"], ["--gpu-index", "0", "--gpu-segments", "0:11"]):
        output = run(cpu_inference + device_args)
        if generated_text(output, "hello") != expected_text:
            raise AssertionError(f"CPU and Vulkan inference emitted different text\n{output}")
        for expected in ("tokens: 9", "canvases: 3", "decoder forwards: 3"):
            if expected not in output:
                raise AssertionError(f"incomplete Vulkan generation metrics\n{output}")


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <ddllm> <f32-model.ddlm> <q40-model.ddlm> <tokenizer.ddlt>"
        )
    executable, f32_model, q40_model, tokenizer = (
        str(Path(value).resolve()) for value in sys.argv[1:]
    )
    qualify_model(executable, f32_model, tokenizer, "f32")
    qualify_model(executable, q40_model, tokenizer, "q80")
    print("ddllm Vulkan integration tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
