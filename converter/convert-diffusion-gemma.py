#!/usr/bin/env python3
"""Convert a Hugging Face DiffusionGemma checkpoint into a DDLM model."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from typing import BinaryIO, Iterable, Sequence

from ddllm_writer import TensorSpec, TensorType, inspect_model, stored_size, write_model


SUPPORTED_ARCHITECTURE = "DiffusionGemmaForBlockDiffusion"
INDEX_FILE = "model.safetensors.index.json"
MATRIX_SUFFIXES = {
    "experts.down_proj",
    "experts.gate_up_proj",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "router.proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
}
LAYER_VECTOR_SUFFIXES = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "post_feedforward_layernorm_1.weight",
    "post_feedforward_layernorm_2.weight",
    "pre_feedforward_layernorm.weight",
    "pre_feedforward_layernorm_2.weight",
    "router.per_expert_scale",
    "router.scale",
    "self_attn.k_norm.weight",
    "self_attn.q_norm.weight",
}


@dataclass(frozen=True)
class SourceTensor:
    output_name: str
    source_key: str
    shard_name: str
    shape: tuple[int, ...]
    tensor_type: int
    transpose: bool = False


class SafetensorsFile:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as source:
            raw_length = source.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"{path}: truncated safetensors header")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length == 0 or header_length > os.path.getsize(path) - 8:
                raise ValueError(f"{path}: invalid safetensors header size")
            self.header = json.loads(source.read(header_length))
        self.data_offset = 8 + header_length

    def info(self, key: str) -> dict:
        try:
            return self.header[key]
        except KeyError as error:
            raise ValueError(f"{self.path}: tensor {key} is missing from its indexed shard") from error

    def _bounds(self, key: str) -> tuple[int, int, str, tuple[int, ...]]:
        info = self.info(key)
        dtype = info["dtype"]
        shape = tuple(int(value) for value in info["shape"])
        start, end = (int(value) for value in info["data_offsets"])
        if start < 0 or end < start:
            raise ValueError(f"{self.path}: invalid offsets for {key}")
        absolute_start = self.data_offset + start
        absolute_end = self.data_offset + end
        if absolute_end > os.path.getsize(self.path):
            raise ValueError(f"{self.path}: tensor {key} extends beyond the shard")
        widths = {"F32": 4, "F16": 2, "BF16": 2}
        if dtype not in widths:
            raise ValueError(f"{self.path}: unsupported source dtype {dtype} for {key}")
        count = math.prod(shape)
        if end - start != count * widths[dtype]:
            raise ValueError(f"{self.path}: byte length does not match shape for {key}")
        return absolute_start, absolute_end, dtype, shape

    def write_f32(self, key: str, output: BinaryIO, chunk_values: int = 1 << 20) -> int:
        start, end, dtype, shape = self._bounds(key)
        if dtype == "F32":
            written = 0
            with open(self.path, "rb") as source:
                source.seek(start)
                remaining = end - start
                while remaining:
                    data = source.read(min(remaining, chunk_values * 4))
                    if not data:
                        raise ValueError(f"{self.path}: unexpected EOF while reading {key}")
                    output.write(data)
                    written += len(data)
                    remaining -= len(data)
            return written

        np = require_numpy()
        source_dtype = np.dtype("<u2") if dtype == "BF16" else np.dtype("<f2")
        values = np.memmap(self.path, dtype=source_dtype, mode="r", offset=start, shape=(math.prod(shape),))
        written = 0
        for offset in range(0, values.size, chunk_values):
            chunk = values[offset : offset + chunk_values]
            if dtype == "BF16":
                chunk = (chunk.astype(np.uint32) << 16).view(np.float32)
            else:
                chunk = chunk.astype(np.float32)
            data = chunk.astype("<f4", copy=False).tobytes()
            output.write(data)
            written += len(data)
        return written

    def iter_f32(self, key: str, chunk_values: int = 1 << 20):
        np = require_numpy()
        start, _, dtype, shape = self._bounds(key)
        source_dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[dtype]
        values = np.memmap(self.path, dtype=np.dtype(source_dtype), mode="r", offset=start, shape=(math.prod(shape),))
        for offset in range(0, values.size, chunk_values):
            chunk = values[offset : offset + chunk_values]
            if dtype == "BF16":
                yield (chunk.astype(np.uint32) << 16).view(np.float32)
            else:
                yield chunk.astype(np.float32, copy=False)


def require_numpy():
    try:
        import numpy
    except ImportError as error:
        raise RuntimeError(
            "conversion of BF16/F16 or Q40 tensors requires NumPy; install converter/requirements.txt"
        ) from error
    return numpy


def parse_layers(value: str | None, layer_count: int) -> list[int]:
    if value is None:
        return list(range(layer_count))
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("--layers contains an empty selection")
        if "-" in item:
            parts = item.split("-")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ValueError(f"invalid layer range {item!r}")
            first, last = (int(part) for part in parts)
            if last < first:
                raise ValueError(f"descending layer range {item!r}")
            candidates = range(first, last + 1)
        else:
            if not item.isdigit():
                raise ValueError(f"invalid layer index {item!r}")
            candidates = [int(item)]
        for layer in candidates:
            if layer < 0 or layer >= layer_count:
                raise ValueError(f"layer {layer} is outside 0..{layer_count - 1}")
            if layer in result:
                raise ValueError(f"layer {layer} was selected more than once")
            result.append(layer)
    if not result:
        raise ValueError("at least one layer must be selected")
    return result


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def validate_config(config: dict) -> dict:
    if config.get("architectures") != [SUPPORTED_ARCHITECTURE]:
        raise ValueError(f"expected architectures=[{SUPPORTED_ARCHITECTURE!r}]")
    text = config.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != "diffusion_gemma_text":
        raise ValueError("checkpoint does not contain a DiffusionGemma text configuration")
    if not config.get("tie_word_embeddings") or not text.get("tie_word_embeddings"):
        raise ValueError("DiffusionGemma conversion requires tied token embeddings")
    layer_count = int(text["num_hidden_layers"])
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != layer_count:
        raise ValueError("layer_types must contain one entry per transformer layer")
    if set(layer_types) - {"sliding_attention", "full_attention"}:
        raise ValueError("unsupported DiffusionGemma attention layer type")
    required_positive = (
        "hidden_size", "intermediate_size", "moe_intermediate_size", "num_attention_heads",
        "num_key_value_heads", "num_global_key_value_heads", "num_experts", "top_k_experts", "vocab_size",
    )
    for key in required_positive:
        if int(text[key]) <= 0:
            raise ValueError(f"text_config.{key} must be positive")
    return text


def expected_layer_shapes(text: dict, layer_type: str) -> dict[str, tuple[int, ...]]:
    hidden = int(text["hidden_size"])
    dense = int(text["intermediate_size"])
    moe = int(text["moe_intermediate_size"])
    experts = int(text["num_experts"])
    q_heads = int(text["num_attention_heads"])
    if layer_type == "sliding_attention":
        head_dim = int(text["head_dim"])
        kv_heads = int(text["num_key_value_heads"])
    else:
        head_dim = int(text["global_head_dim"])
        kv_heads = int(text["num_global_key_value_heads"])
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    shapes = {
        "experts.down_proj": (experts, hidden, moe),
        "experts.gate_up_proj": (experts, moe * 2, hidden),
        "input_layernorm.weight": (hidden,),
        "layer_scalar": (1,),
        "mlp.down_proj.weight": (hidden, dense),
        "mlp.gate_proj.weight": (dense, hidden),
        "mlp.up_proj.weight": (dense, hidden),
        "post_attention_layernorm.weight": (hidden,),
        "post_feedforward_layernorm.weight": (hidden,),
        "post_feedforward_layernorm_1.weight": (hidden,),
        "post_feedforward_layernorm_2.weight": (hidden,),
        "pre_feedforward_layernorm.weight": (hidden,),
        "pre_feedforward_layernorm_2.weight": (hidden,),
        "router.per_expert_scale": (experts,),
        "router.proj.weight": (experts, hidden),
        "router.scale": (hidden,),
        "self_attn.k_norm.weight": (head_dim,),
        "self_attn.k_proj.weight": (kv_dim, hidden),
        "self_attn.o_proj.weight": (hidden, q_dim),
        "self_attn.q_norm.weight": (head_dim,),
        "self_attn.q_proj.weight": (q_dim, hidden),
    }
    if layer_type == "sliding_attention":
        shapes["self_attn.v_proj.weight"] = (kv_dim, hidden)
    return shapes


def choose_key(weight_map: dict[str, str], description: str, candidates: Sequence[str]) -> str:
    present = [candidate for candidate in candidates if candidate in weight_map]
    if not present:
        raise ValueError(f"missing required tensor {description}; tried: {', '.join(candidates)}")
    return present[0]


def tensor_type_for(weight_type: str, is_matrix: bool) -> int:
    return TensorType.Q40 if weight_type == "q40" and is_matrix else TensorType.F32


def build_manifest(source_dir: str, weight_type: str, layers_value: str | None):
    config = load_json(os.path.join(source_dir, "config.json"))
    generation = load_json(os.path.join(source_dir, "generation_config.json"))
    text = validate_config(config)
    selected_layers = parse_layers(layers_value, int(text["num_hidden_layers"]))
    index = load_json(os.path.join(source_dir, INDEX_FILE))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{INDEX_FILE} does not contain a weight_map")

    requested: list[tuple[str, str, tuple[int, ...], bool]] = []
    hidden = int(text["hidden_size"])
    vocab = int(text["vocab_size"])
    dense = int(text["intermediate_size"])
    common = [
        ("token_embedding.weight", "token embedding", (vocab, hidden), True,
         ["model.decoder.embed_tokens.weight", "model.encoder.language_model.embed_tokens.weight", "lm_head.weight"]),
        ("self_conditioning.pre_norm.weight", "self-conditioning pre-norm", (hidden,), False,
         ["model.decoder.self_conditioning.pre_norm.weight"]),
        ("self_conditioning.gate_proj.weight", "self-conditioning gate projection", (dense, hidden), True,
         ["model.decoder.self_conditioning.gate_proj.weight"]),
        ("self_conditioning.up_proj.weight", "self-conditioning up projection", (dense, hidden), True,
         ["model.decoder.self_conditioning.up_proj.weight"]),
        ("self_conditioning.down_proj.weight", "self-conditioning down projection", (hidden, dense), True,
         ["model.decoder.self_conditioning.down_proj.weight"]),
        ("final_norm.weight", "final norm", (hidden,), False,
         ["model.decoder.norm.weight", "model.encoder.language_model.norm.weight"]),
    ]
    for output_name, description, shape, is_matrix, aliases in common:
        requested.append((output_name, choose_key(weight_map, description, aliases), shape, is_matrix))

    layer_mapping: list[dict] = []
    for output_layer, source_layer in enumerate(selected_layers):
        layer_type = text["layer_types"][source_layer]
        shapes = expected_layer_shapes(text, layer_type)
        expected_source_keys = {f"model.decoder.layers.{source_layer}.{suffix}" for suffix in shapes}
        actual_source_keys = {
            key for key in weight_map if key.startswith(f"model.decoder.layers.{source_layer}.")
        }
        unexpected = sorted(actual_source_keys - expected_source_keys)
        missing = sorted(expected_source_keys - set(weight_map))
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if unexpected:
                details.append("unexpected=" + ",".join(unexpected))
            raise ValueError(f"layer {source_layer} tensor manifest mismatch: {'; '.join(details)}")

        for suffix, shape in shapes.items():
            if suffix == "layer_scalar":
                output_name = f"layers.{output_layer}.decoder_layer_scalar"
            else:
                output_name = f"layers.{output_layer}.{suffix}"
            requested.append(
                (output_name, f"model.decoder.layers.{source_layer}.{suffix}", shape, suffix in MATRIX_SUFFIXES)
            )
        encoder_scalar = f"model.encoder.language_model.layers.{source_layer}.layer_scalar"
        if encoder_scalar not in weight_map:
            raise ValueError(f"missing required tensor {encoder_scalar}")
        requested.append((f"layers.{output_layer}.encoder_layer_scalar", encoder_scalar, (1,), False))
        layer_mapping.append(
            {"output_layer": output_layer, "source_layer": source_layer, "attention_type": layer_type}
        )

    required_shards = sorted({weight_map[source_key] for _, source_key, _, _ in requested})
    missing_shards = [name for name in required_shards if not os.path.isfile(os.path.join(source_dir, name))]
    if missing_shards:
        raise FileNotFoundError("missing required checkpoint shard(s): " + ", ".join(missing_shards))

    shard_readers = {
        name: SafetensorsFile(os.path.join(source_dir, name)) for name in required_shards
    }
    manifest: list[SourceTensor] = []
    for output_name, source_key, expected_shape, is_matrix in requested:
        shard_name = weight_map[source_key]
        # Validate the source payload during manifest construction as well as
        # during conversion.  This keeps --dry-run useful for qualifying a
        # downloaded checkpoint without allocating or writing a full model.
        _, _, _, actual_shape = shard_readers[shard_name]._bounds(source_key)
        if actual_shape != expected_shape:
            raise ValueError(
                f"tensor {source_key} has shape {actual_shape}, expected {expected_shape}"
            )
        manifest.append(
            SourceTensor(
                output_name,
                source_key,
                shard_name,
                actual_shape,
                tensor_type_for(weight_type, is_matrix),
            )
        )

    embedding = manifest[0]
    if embedding.output_name != "token_embedding.weight":
        raise AssertionError("token embedding must be the first converted tensor")
    manifest.insert(
        1,
        SourceTensor(
            "token_embedding.mix_weight",
            embedding.source_key,
            embedding.shard_name,
            (hidden, vocab),
            embedding.tensor_type,
            True,
        ),
    )

    converted_text = dict(text)
    converted_text["num_hidden_layers"] = len(selected_layers)
    converted_text["layer_types"] = [text["layer_types"][layer] for layer in selected_layers]
    metadata = {
        "architecture": SUPPORTED_ARCHITECTURE,
        "format_version": 1,
        "weight_type": weight_type,
        "text_config": converted_text,
        "generation_config": generation,
        "special_tokens": {
            key: config[key] for key in ("boi_token_id", "eoi_token_id", "eos_token_id", "image_token_id") if key in config
        },
        "canvas_length": int(config["canvas_length"]),
        "embedding_scale": math.sqrt(hidden),
        "source_layers": layer_mapping,
        "source_tensor_count": len(weight_map),
        "ignored_vision_tensor_count": sum(
            key.startswith("model.encoder.vision_tower.") or key.startswith("model.encoder.embed_vision.")
            for key in weight_map
        ),
    }
    return metadata, manifest, shard_readers, required_shards


def quantize_q40(values):
    np = require_numpy()
    groups = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, 32)
    maximum = groups.max(axis=1)
    minimum = groups.min(axis=1)
    delta = np.where(-minimum > maximum, minimum, maximum) / -8.0
    inverse = np.zeros_like(delta, dtype=np.float32)
    np.divide(1.0, delta, out=inverse, where=delta != 0)
    quantized = np.clip((groups * inverse[:, None] + 8.5).astype(np.int32), 0, 15).astype(np.uint8)
    packed = quantized[:, :16] | (quantized[:, 16:] << 4)
    records = np.empty((groups.shape[0], 18), dtype=np.uint8)
    records[:, :2] = delta.astype("<f2").view(np.uint8).reshape(-1, 2)
    records[:, 2:] = packed
    return records


def write_q40(reader: SafetensorsFile, key: str, output: BinaryIO) -> int:
    written = 0
    for chunk in reader.iter_f32(key):
        if chunk.size % 32:
            raise ValueError(f"tensor {key} Q40 chunk is not block aligned")
        data = quantize_q40(chunk).tobytes()
        output.write(data)
        written += len(data)
    return written


def write_transposed(
    reader: SafetensorsFile,
    key: str,
    output_shape: tuple[int, ...],
    tensor_type: int,
    output: BinaryIO,
    tile_rows: int = 4096,
) -> int:
    np = require_numpy()
    start, _, dtype, source_shape = reader._bounds(key)
    if len(source_shape) != 2 or output_shape != (source_shape[1], source_shape[0]):
        raise ValueError(f"tensor {key} cannot be transposed from {source_shape} to {output_shape}")
    output_rows, output_columns = output_shape
    if tensor_type == TensorType.Q40 and output_columns % 32:
        raise ValueError(f"transposed Q40 tensor {key} is not block aligned by row")

    source_dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[dtype]
    source = np.memmap(
        reader.path,
        dtype=np.dtype(source_dtype),
        mode="r",
        offset=start,
        shape=source_shape,
    )
    tensor_start = output.tell()
    total_bytes = stored_size(output_shape, tensor_type)
    output.seek(tensor_start + total_bytes - 1)
    output.write(b"\x00")
    if tensor_type == TensorType.F32:
        row_bytes = output_columns * 4
    else:
        row_bytes = output_columns // 32 * 18

    for row_start in range(0, source_shape[0], tile_rows):
        row_end = min(row_start + tile_rows, source_shape[0])
        if tensor_type == TensorType.Q40 and (row_start % 32 or (row_end - row_start) % 32):
            raise ValueError(f"transposed Q40 tile for {key} is not block aligned")
        source_tile = source[row_start:row_end]
        if dtype == "BF16":
            source_tile = (source_tile.astype(np.uint32) << 16).view(np.float32)
        else:
            source_tile = source_tile.astype(np.float32, copy=False)
        transposed = np.ascontiguousarray(source_tile.T)
        if tensor_type == TensorType.F32:
            encoded = transposed.astype("<f4", copy=False).view(np.uint8).reshape(output_rows, -1)
            tile_offset = row_start * 4
        else:
            encoded = quantize_q40(transposed).reshape(output_rows, -1)
            tile_offset = row_start // 32 * 18
        for row in range(output_rows):
            output.seek(tensor_start + row * row_bytes + tile_offset)
            output.write(encoded[row].tobytes())
    output.seek(tensor_start + total_bytes)
    return total_bytes


def make_tensor_spec(tensor: SourceTensor, readers: dict[str, SafetensorsFile]) -> TensorSpec:
    reader = readers[tensor.shard_name]
    if tensor.transpose:
        def write(output: BinaryIO, reader=reader, tensor=tensor) -> int:
            return write_transposed(
                reader, tensor.source_key, tensor.shape, tensor.tensor_type, output
            )
    elif tensor.tensor_type == TensorType.F32:
        def write(output: BinaryIO, reader=reader, key=tensor.source_key) -> int:
            return reader.write_f32(key, output)
    else:
        def write(output: BinaryIO, reader=reader, key=tensor.source_key) -> int:
            return write_q40(reader, key, output)
    return TensorSpec(tensor.output_name, tensor.shape, tensor.tensor_type, write)


def manifest_json(metadata: dict, tensors: Sequence[SourceTensor], shards: Sequence[str]) -> dict:
    return {
        "metadata": metadata,
        "required_shards": list(shards),
        "tensors": [
            {
                "name": tensor.output_name,
                "source": tensor.source_key,
                "shard": tensor.shard_name,
                "shape": list(tensor.shape),
                "type": "q40" if tensor.tensor_type == TensorType.Q40 else "f32",
                "bytes": stored_size(tensor.shape, tensor.tensor_type),
                "transform": "transpose" if tensor.transpose else "identity",
            }
            for tensor in tensors
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="DiffusionGemma Hugging Face checkpoint directory")
    parser.add_argument("output", nargs="?", help="output DDLM model")
    parser.add_argument("--type", choices=("f32", "q40"), default="f32", dest="weight_type")
    parser.add_argument(
        "--layers",
        help="source layers to convert, e.g. 0,5-7; omitted means every layer",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print the manifest without writing")
    args = parser.parse_args(argv)

    try:
        metadata, manifest, readers, shards = build_manifest(
            os.path.abspath(args.source), args.weight_type, args.layers
        )
        if args.dry_run:
            print(json.dumps(manifest_json(metadata, manifest, shards), indent=2, sort_keys=True))
            return 0
        if not args.output:
            parser.error("output is required unless --dry-run is used")
        specs = [make_tensor_spec(tensor, readers) for tensor in manifest]
        entries = write_model(args.output, metadata, specs)
        inspected_metadata, inspected_entries = inspect_model(args.output)
        if inspected_metadata != metadata or inspected_entries != entries:
            raise ValueError("written DDLM model failed its read-back validation")
        print(
            f"Wrote {args.output}: {len(entries)} tensors, {os.path.getsize(args.output)} bytes, "
            f"layers={metadata['source_layers']}"
        )
        return 0
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
