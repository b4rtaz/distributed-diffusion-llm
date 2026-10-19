#!/usr/bin/env python3
"""Compatibility entry point for inspecting DDLM containers."""

import argparse
import json

from ddllm_writer import TYPE_NAMES, inspect_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    args = parser.parse_args()
    metadata, tensors = inspect_model(args.model)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    for tensor in tensors:
        print(f"{tensor.name}: {list(tensor.shape)} {TYPE_NAMES[tensor.tensor_type]} @{tensor.offset}+{tensor.length}")


if __name__ == "__main__":
    main()
