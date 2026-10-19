#!/usr/bin/env python3
"""Build a small ordinary DDLM model for C++ loader tests."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

from ddllm_writer import TensorSpec, stored_size, write_model


def load_converter_test():
    path = Path(__file__).with_name("converter-test.py")
    spec = importlib.util.spec_from_file_location("ddllm_converter_test_fixture", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(f"Usage: {sys.argv[0]} <output.ddlm> [f32|q40]")
    tests = load_converter_test()
    output = Path(sys.argv[1])
    weight_type = sys.argv[2] if len(sys.argv) == 3 else "f32"
    if weight_type not in ("f32", "q40"):
        raise SystemExit("weight type must be f32 or q40")
    with tempfile.TemporaryDirectory(prefix="ddllm-model-loader-") as directory:
        tests.create_checkpoint(directory)
        if weight_type == "q40":
            metadata, manifest, _, _ = tests.CONVERTER_MODULE.build_manifest(directory, weight_type, "0")

            def zero_writer(size):
                def write(output):
                    remaining = size
                    zeros = b"\x00" * min(size, 65536)
                    while remaining:
                        chunk = zeros[: min(remaining, len(zeros))]
                        output.write(chunk)
                        remaining -= len(chunk)
                    return size
                return write

            specs = [
                TensorSpec(tensor.output_name, tensor.shape, tensor.tensor_type,
                           zero_writer(stored_size(tensor.shape, tensor.tensor_type)))
                for tensor in manifest
            ]
            write_model(str(output), metadata, specs)
            print(f"Model loader fixture: {output}")
            return 0
        result = subprocess.run(
            [sys.executable, tests.CONVERTER, directory, str(output), "--layers", "0", "--type", weight_type],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
    print(f"Model loader fixture: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
