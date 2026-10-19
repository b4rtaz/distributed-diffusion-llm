#!/usr/bin/env python3
"""Exercise every ddllm CLI mode with matching tiny DDLM/DDLT fixtures."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path


def run(command: list[str], *, input_text: str | None = None, timeout: float = 30.0) -> str:
    result = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}\n{result.stdout}"
        )
    if "Critical error:" in result.stdout:
        raise AssertionError(f"command reported a critical error: {' '.join(command)}\n{result.stdout}")
    return result.stdout


def root_command(executable: str, mode: str, model: str, tokenizer: str) -> list[str]:
    return [
        executable,
        mode,
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--nthreads",
        "1",
        "--nbatches",
        "4",
        "--max-seq-len",
        "128",
        "--seed",
        "42",
        "--denoising-steps",
        "1",
        "--buffer-float-type",
        "f32",
    ]


def generated_text(output: str, prompt: str) -> str:
    start_marker = prompt + "\n"
    end_marker = "\n\nGeneration\n"
    start = output.rfind(start_marker)
    end = output.rfind(end_marker)
    if start < 0 or end < start:
        raise AssertionError(f"inference output is missing generation markers\n{output}")
    return output[start + len(start_marker) : end]


def results_block(output: str) -> str:
    marker = "\nResults\n"
    start = output.rfind(marker)
    if start < 0:
        raise AssertionError(f"output is missing a results block\n{output}")
    lines = output[start:].splitlines()
    if len(lines) < 5:
        raise AssertionError(f"output has an incomplete results block\n{output}")
    return "\n".join(lines[:5]) + "\n"


def chat_texts(output: str) -> list[str]:
    start_marker = "\nAssistant\n"
    end_marker = "\n\nUser\n> "
    result: list[str] = []
    cursor = 0
    while True:
        start = output.find(start_marker, cursor)
        if start < 0:
            break
        end = output.find(end_marker, start + len(start_marker))
        if end < start:
            raise AssertionError(f"chat output has an unterminated assistant turn\n{output}")
        result.append(output[start + len(start_marker) : end])
        cursor = end + len(end_marker)
    if not result:
        raise AssertionError(f"chat output is missing assistant markers\n{output}")
    return result


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def distributed_modes(
    executable: str,
    model: str,
    tokenizer: str,
    expected_inference_text: str,
    expected_perplexity: str,
    expected_chat_texts: list[str],
) -> None:
    port = free_port()
    worker = subprocess.Popen(
        [executable, "worker", "--host", "127.0.0.1", "--port", str(port), "--nthreads", "1"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        time.sleep(0.25)
        worker_address = f"127.0.0.1:{port}"

        inference_command = root_command(executable, "inference", model, tokenizer)
        inference_command.extend(["--prompt", "hello", "--steps", "10", "--workers", worker_address])
        inference_output = run(inference_command)
        if "2 node(s)" not in inference_output:
            raise AssertionError(f"distributed inference did not initialize two nodes\n{inference_output}")
        if ("tokens: 9" not in inference_output or "canvases: 3" not in inference_output or
                "decoder forwards: 3" not in inference_output):
            raise AssertionError(f"distributed inference did not execute three canvases\n{inference_output}")
        if generated_text(inference_output, "hello") != expected_inference_text:
            raise AssertionError("one-node and two-node inference emitted different text")
        if "sent: 0 kB" in inference_output and "received: 0 kB" in inference_output:
            raise AssertionError(f"distributed inference reported no network traffic\n{inference_output}")
        if worker.poll() is not None:
            raise AssertionError("worker exited instead of reconnecting after distributed inference")

        perplexity_command = root_command(executable, "perplexity", model, tokenizer)
        perplexity_command.extend(["--prompt", "hello hello", "--workers", worker_address])
        perplexity_output = run(perplexity_command)
        if "2 node(s)" not in perplexity_output:
            raise AssertionError(f"distributed perplexity did not initialize two nodes\n{perplexity_output}")
        if results_block(perplexity_output) != expected_perplexity:
            raise AssertionError(
                "one-node and two-node perplexity results differ\n"
                f"one node:{expected_perplexity}\n"
                f"two nodes:{results_block(perplexity_output)}"
            )
        if worker.poll() is not None:
            raise AssertionError("worker exited instead of reconnecting after distributed perplexity")

        chat_command = root_command(executable, "chat", model, tokenizer)
        chat_command.extend(["--steps", "1", "--workers", worker_address])
        chat_output = run(chat_command, input_text="\nhello\nagain\n")
        if "2 node(s)" not in chat_output:
            raise AssertionError(f"distributed chat did not initialize two nodes\n{chat_output}")
        if chat_texts(chat_output) != expected_chat_texts:
            raise AssertionError("one-node and two-node chat turns emitted different text")
        if worker.poll() is not None:
            raise AssertionError("worker exited instead of reconnecting after distributed chat")
    finally:
        worker.terminate()
        try:
            worker.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.communicate(timeout=5.0)


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(f"Usage: {sys.argv[0]} <ddllm> <model.ddlm> <tokenizer.ddlt>")
    executable, model, tokenizer = (str(Path(value).resolve()) for value in sys.argv[1:])

    help_output = run([executable, "--help"])
    for option in (
        "--buffer-float-type",
        "--net-turbo",
        "--gpu-index",
        "--gpu-segments",
        "--denoising-steps",
        "--entropy-bound",
        "--topp",
    ):
        if option not in help_output:
            raise AssertionError(f"help output does not document {option}\n{help_output}")
    if "unsupported by the entropy-bound diffusion sampler" not in help_output:
        raise AssertionError(f"help output does not explain --topp compatibility\n{help_output}")
    topp = subprocess.run(
        [executable, "inference", "--topp", "0.9"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if topp.returncode == 0 or "--topp is not supported" not in topp.stdout:
        raise AssertionError(f"--topp was not rejected explicitly\n{topp.stdout}")

    inference_command = root_command(executable, "inference", model, tokenizer)
    inference_command.extend(["--prompt", "hello", "--steps", "10"])
    inference_output = run(inference_command)
    if ("tokens: 9" not in inference_output or "canvases: 3" not in inference_output or
            "decoder forwards: 3" not in inference_output):
        raise AssertionError(f"inference did not execute three deterministic canvases\n{inference_output}")
    one_node_text = generated_text(inference_output, "hello")

    perplexity_command = root_command(executable, "perplexity", model, tokenizer)
    perplexity_command.extend(["--prompt", "hello hello"])
    perplexity_output = run(perplexity_command)
    if "Results\n  perplexity: 512" not in perplexity_output or "prob=" not in perplexity_output:
        raise AssertionError(f"perplexity output is incomplete or unexpected\n{perplexity_output}")

    chat_command = root_command(executable, "chat", model, tokenizer)
    chat_command.extend(["--steps", "1"])
    chat_output = run(chat_command, input_text="\nhello\nagain\n")
    one_node_chat = chat_texts(chat_output)
    if len(one_node_chat) != 2 or "(end of context)" not in chat_output:
        raise AssertionError(f"chat did not complete two interactive turns\n{chat_output}")

    distributed_modes(
        executable,
        model,
        tokenizer,
        one_node_text,
        results_block(perplexity_output),
        one_node_chat,
    )
    print("ddllm CLI integration tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
