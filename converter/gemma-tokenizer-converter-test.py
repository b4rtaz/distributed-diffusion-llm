#!/usr/bin/env python3
"""Self-contained tests and tiny fixture generator for the DDLT converter."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path


def load_converter():
    path = Path(__file__).with_name("convert-diffusion-gemma-tokenizer.py")
    spec = importlib.util.spec_from_file_location("gemma_tokenizer_converter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def create_source(path: Path) -> dict:
    special = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|turn>", "<turn|>"]
    tokens = list(special)
    tokens.extend(f"<0x{byte:02X}>" for byte in range(256))
    tokens.extend(["a", "b", "h", "e", "l", "o", "▁", "he", "hel", "hell", "hello", "▁hello"])
    tokens.extend(f"<unused{i}>" for i in range(512 - len(tokens)))
    vocab = {token: token_id for token_id, token in enumerate(tokens)}
    merges = [["h", "e"], ["he", "l"], ["hel", "l"], ["hell", "o"], ["▁", "hello"]]
    added = [
        {
            "id": vocab[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token in special
    ]
    tokenizer = {
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithPrevious",
            "invert": False,
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ],
        },
        "added_tokens": added,
    }
    config = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "mask_token": "<mask>",
        "model_specific_special_tokens": {"sot_token": "<|turn>", "eot_token": "<turn|>"},
    }
    generation = {"eos_token_id": [vocab["<eos>"], vocab["<turn|>"]]}
    write_json(path / "tokenizer.json", tokenizer)
    write_json(path / "tokenizer_config.json", config)
    write_json(path / "generation_config.json", generation)
    (path / "chat_template.jinja").write_text("{{ bos_token }} gemma4 canonical test", encoding="utf-8")
    return vocab


def main() -> int:
    converter = load_converter()
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.gettempdir()) / "ddllm-gemma-tokenizer-test.ddlt"
    with tempfile.TemporaryDirectory(prefix="ddllm-tokenizer-source-") as directory:
        source = Path(directory)
        vocab = create_source(source)
        summary = converter.convert(source, output)
        assert summary["vocab"] == len(vocab)
        assert summary["merges"] == 5
        assert summary["generation_eos_ids"] == [vocab["<eos>"], vocab["<turn|>"]]
        data = output.read_bytes()
        header = converter.HEADER.unpack_from(data)
        assert header[0] == converter.MAGIC
        assert header[1] == converter.VERSION
        assert header[3] == len(data)
        assert header[4] == converter.HEADER.size

        broken = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))
        broken["model"]["byte_fallback"] = False
        write_json(source / "tokenizer.json", broken)
        try:
            converter.convert(source, output.with_suffix(".broken.ddlt"))
        except converter.TokenizerConversionError as error:
            assert "byte fallback" in str(error)
        else:
            raise AssertionError("converter accepted tokenizer without byte fallback")
    print(f"Tokenizer converter tests passed; fixture: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
