#!/usr/bin/env python3
"""Convert a Hugging Face DiffusionGemma tokenizer to the DDLT format."""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import BinaryIO, Dict, List, Sequence, Tuple


MAGIC = b"DDLT"
VERSION = 1
ENDIAN_MARKER = 0x01020304

FLAG_BYTE_FALLBACK = 1 << 0
FLAG_REPLACE_SPACE = 1 << 1
FLAG_ADD_PREFIX_SPACE = 1 << 2
FLAG_ADD_BOS = 1 << 3

HEADER = struct.Struct("<4sIIQIIIIIIIIII")
U32 = struct.Struct("<I")
I32 = struct.Struct("<i")
MERGE = struct.Struct("<III")


class TokenizerConversionError(RuntimeError):
    pass


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except FileNotFoundError as exc:
        raise TokenizerConversionError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TokenizerConversionError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TokenizerConversionError(f"expected a JSON object in {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TokenizerConversionError(message)


def _ordered_vocab(raw_vocab: dict) -> Tuple[List[bytes], Dict[str, int]]:
    _require(isinstance(raw_vocab, dict), "tokenizer model vocab must be an object")
    size = len(raw_vocab)
    tokens: List[bytes | None] = [None] * size
    ids: Dict[str, int] = {}
    for token, token_id in raw_vocab.items():
        _require(isinstance(token, str), "vocabulary token must be a string")
        _require(isinstance(token_id, int), f"token id for {token!r} must be an integer")
        _require(0 <= token_id < size, f"token id {token_id} is outside 0..{size - 1}")
        _require(tokens[token_id] is None, f"duplicate vocabulary id {token_id}")
        encoded = token.encode("utf-8")
        _require(encoded, f"empty vocabulary token at id {token_id}")
        tokens[token_id] = encoded
        ids[token] = token_id
    missing = [index for index, token in enumerate(tokens) if token is None]
    if missing:
        raise TokenizerConversionError(f"vocabulary ids are not contiguous; first missing id is {missing[0]}")
    return [token for token in tokens if token is not None], ids


def _resolve_merges(raw_merges: object, vocab: Dict[str, int]) -> List[Tuple[int, int, int]]:
    _require(isinstance(raw_merges, list), "tokenizer model merges must be an array")
    result: List[Tuple[int, int, int]] = []
    seen = set()
    for rank, raw_merge in enumerate(raw_merges):
        if isinstance(raw_merge, list) and len(raw_merge) == 2:
            left, right = raw_merge
        elif isinstance(raw_merge, str):
            parts = raw_merge.split(" ")
            _require(len(parts) == 2, f"merge rank {rank} is not a pair")
            left, right = parts
        else:
            raise TokenizerConversionError(f"merge rank {rank} is not a two-token pair")
        _require(isinstance(left, str) and isinstance(right, str), f"merge rank {rank} contains a non-string token")
        _require(left in vocab, f"merge rank {rank} references missing left token {left!r}")
        _require(right in vocab, f"merge rank {rank} references missing right token {right!r}")
        merged = left + right
        _require(merged in vocab, f"merge rank {rank} produces missing token {merged!r}")
        key = (vocab[left], vocab[right])
        _require(key not in seen, f"duplicate merge pair at rank {rank}: {left!r} + {right!r}")
        seen.add(key)
        result.append((key[0], key[1], vocab[merged]))
    return result


def _validate_contract(tokenizer: dict) -> None:
    model = tokenizer.get("model")
    _require(isinstance(model, dict), "tokenizer.json is missing model")
    _require(model.get("type") == "BPE", "only a BPE DiffusionGemma tokenizer is supported")
    _require(model.get("byte_fallback") is True, "DiffusionGemma tokenizer must enable byte fallback")
    _require(model.get("dropout") is None, "BPE dropout is not supported")
    _require(model.get("continuing_subword_prefix") is None, "continuing-subword prefixes are not supported")
    _require(model.get("end_of_word_suffix") is None, "end-of-word suffixes are not supported")
    _require(model.get("ignore_merges") is False, "DiffusionGemma tokenizer must apply ordered merges")

    expected_normalizer = {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
    _require(tokenizer.get("normalizer") == expected_normalizer, "unsupported normalizer; expected space-to-U+2581 replacement")
    expected_pre_tokenizer = {
        "type": "Split",
        "pattern": {"String": " "},
        "behavior": "MergedWithPrevious",
        "invert": False,
    }
    _require(tokenizer.get("pre_tokenizer") == expected_pre_tokenizer, "unsupported DiffusionGemma pre-tokenizer")
    expected_decoder = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    _require(tokenizer.get("decoder") == expected_decoder, "unsupported DiffusionGemma decoder")


def _token_id(value: object, vocab: Dict[str, int], name: str) -> int:
    _require(isinstance(value, str), f"{name} must be a token string")
    _require(value in vocab, f"{name} token {value!r} is missing from the vocabulary")
    return vocab[value]


def _metadata(config: dict, vocab: Dict[str, int]) -> Dict[str, int]:
    names = ["bos_token", "eos_token", "pad_token", "unk_token", "mask_token"]
    model_specific = config.get("model_specific_special_tokens", {})
    _require(isinstance(model_specific, dict), "model_specific_special_tokens must be an object")
    values: Dict[str, object] = {name: config.get(name) for name in names}
    for name, token in model_specific.items():
        _require(isinstance(name, str), "model-specific token name must be a string")
        values[name] = token
    result = {}
    for name, token in values.items():
        if token is not None:
            result[name + "_id"] = _token_id(token, vocab, name)
    return result


def _generation_eos(generation: dict, vocab_size: int, fallback: int) -> List[int]:
    raw = generation.get("eos_token_id", fallback)
    values = raw if isinstance(raw, list) else [raw]
    _require(values, "generation EOS token list cannot be empty")
    result = []
    for value in values:
        _require(isinstance(value, int), "generation EOS token ids must be integers")
        _require(0 <= value < vocab_size, f"generation EOS token id {value} is outside the vocabulary")
        if value not in result:
            result.append(value)
    return result


def _write_u32(file: BinaryIO, value: int) -> None:
    file.write(U32.pack(value))


def _write_blob(file: BinaryIO, value: bytes) -> None:
    _write_u32(file, len(value))
    file.write(value)


def _write_file(
    output: Path,
    tokens: Sequence[bytes],
    merges: Sequence[Tuple[int, int, int]],
    special_ids: Sequence[int],
    eos_ids: Sequence[int],
    metadata: Dict[str, int],
    flags: int,
    template_name: bytes,
    chat_template: bytes,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=str(output.parent))
    try:
        with os.fdopen(temporary_fd, "w+b") as file:
            max_token_bytes = max(map(len, tokens))
            file.write(b"\0" * HEADER.size)
            for token_id in eos_ids:
                file.write(I32.pack(token_id))
            for name in sorted(metadata):
                _write_blob(file, name.encode("utf-8"))
                file.write(I32.pack(metadata[name]))
            for token_id in special_ids:
                _write_u32(file, token_id)
            for token in tokens:
                _write_blob(file, token)
            for left, right, merged in merges:
                file.write(MERGE.pack(left, right, merged))
            file.write(template_name)
            file.write(chat_template)
            file_size = file.tell()
            file.seek(0)
            file.write(
                HEADER.pack(
                    MAGIC,
                    VERSION,
                    ENDIAN_MARKER,
                    file_size,
                    HEADER.size,
                    len(tokens),
                    len(merges),
                    len(special_ids),
                    len(eos_ids),
                    len(metadata),
                    flags,
                    max_token_bytes,
                    len(template_name),
                    len(chat_template),
                )
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def convert(source: Path, output: Path) -> dict:
    tokenizer = _read_json(source / "tokenizer.json")
    config = _read_json(source / "tokenizer_config.json")
    generation = _read_json(source / "generation_config.json")
    _validate_contract(tokenizer)

    model = tokenizer["model"]
    tokens, vocab = _ordered_vocab(model.get("vocab"))
    merges = _resolve_merges(model.get("merges"), vocab)

    raw_added = tokenizer.get("added_tokens")
    _require(isinstance(raw_added, list), "tokenizer.json added_tokens must be an array")
    special_ids = []
    for entry in raw_added:
        _require(isinstance(entry, dict), "added token entry must be an object")
        if not entry.get("special", False):
            continue
        token_id = entry.get("id")
        content = entry.get("content")
        _require(isinstance(token_id, int) and 0 <= token_id < len(tokens), "invalid added special token id")
        _require(isinstance(content, str) and vocab.get(content) == token_id, f"added token {content!r} does not match vocab id {token_id}")
        _require(entry.get("normalized") is False, f"special token {content!r} must bypass normalization")
        special_ids.append(token_id)
    _require(len(special_ids) == len(set(special_ids)), "duplicate added special token id")

    metadata = _metadata(config, vocab)
    _require("eos_token_id" in metadata, "tokenizer config is missing eos_token")
    eos_ids = _generation_eos(generation, len(tokens), metadata["eos_token_id"])

    template_path = source / "chat_template.jinja"
    try:
        chat_template = template_path.read_bytes()
    except FileNotFoundError as exc:
        raise TokenizerConversionError(f"missing required file: {template_path}") from exc
    _require(chat_template, "chat_template.jinja cannot be empty")

    flags = FLAG_BYTE_FALLBACK | FLAG_REPLACE_SPACE
    _write_file(
        output,
        tokens,
        merges,
        special_ids,
        eos_ids,
        metadata,
        flags,
        b"gemma4-canonical-text-v1",
        chat_template,
    )
    return {
        "output": str(output),
        "bytes": output.stat().st_size,
        "vocab": len(tokens),
        "merges": len(merges),
        "special_tokens": len(special_ids),
        "generation_eos_ids": eos_ids,
        "metadata": metadata,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="DiffusionGemma checkpoint directory")
    parser.add_argument("output", nargs="?", type=Path, help="output .ddlt path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output or Path("diffusion-gemma-tokenizer.ddlt")
    try:
        summary = convert(args.source, output)
    except TokenizerConversionError as exc:
        raise SystemExit(f"error: {exc}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
