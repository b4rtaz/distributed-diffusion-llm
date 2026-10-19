"""Writer and inspector for the named-tensor DDLM model container."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import struct
from typing import BinaryIO, Callable, Iterable, Sequence


MAGIC = b"DDLM\x00\x01\x00\x00"
FORMAT_VERSION = 1
ENDIAN_MARKER = 0x01020304
ALIGNMENT = 64
HEADER = struct.Struct("<8sIIIIQQQQ")
DIRECTORY_PREFIX = struct.Struct("<IIIIQQ")


class TensorType:
    F32 = 0
    Q40 = 2


TYPE_NAMES = {TensorType.F32: "f32", TensorType.Q40: "q40"}


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def element_count(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        if dimension <= 0:
            raise ValueError(f"invalid tensor dimension {dimension}")
        result *= dimension
    return result


def stored_size(shape: Sequence[int], tensor_type: int) -> int:
    count = element_count(shape)
    if tensor_type == TensorType.F32:
        return count * 4
    if tensor_type == TensorType.Q40:
        if count % 32:
            raise ValueError(f"Q40 tensor with {count} elements is not block aligned")
        return count // 32 * 18
    raise ValueError(f"unsupported DDLM tensor type {tensor_type}")


TensorWriter = Callable[[BinaryIO], int]


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    tensor_type: int
    write: TensorWriter


@dataclasses.dataclass(frozen=True)
class TensorEntry:
    name: str
    shape: tuple[int, ...]
    tensor_type: int
    offset: int
    length: int


def _encode_directory(entries: Iterable[TensorEntry]) -> bytes:
    output = bytearray()
    for entry in entries:
        name = entry.name.encode("utf-8")
        output.extend(
            DIRECTORY_PREFIX.pack(
                len(name),
                entry.tensor_type,
                len(entry.shape),
                0,
                entry.offset,
                entry.length,
            )
        )
        output.extend(struct.pack(f"<{len(entry.shape)}Q", *entry.shape))
        output.extend(name)
    return bytes(output)


def write_model(path: str, metadata: dict, tensors: Sequence[TensorSpec]) -> list[TensorEntry]:
    names = [tensor.name for tensor in tensors]
    if len(names) != len(set(names)):
        raise ValueError("DDLM tensor names must be unique")

    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    directory_offset = HEADER.size + len(metadata_bytes)

    placeholder_entries = [
        TensorEntry(t.name, t.shape, t.tensor_type, 0, stored_size(t.shape, t.tensor_type))
        for t in tensors
    ]
    directory_size = len(_encode_directory(placeholder_entries))
    data_offset = align(directory_offset + directory_size)

    entries: list[TensorEntry] = []
    cursor = data_offset
    for tensor, placeholder in zip(tensors, placeholder_entries):
        cursor = align(cursor)
        entries.append(
            TensorEntry(
                tensor.name,
                tensor.shape,
                tensor.tensor_type,
                cursor,
                placeholder.length,
            )
        )
        cursor += placeholder.length
    file_size = cursor
    directory_bytes = _encode_directory(entries)
    if len(directory_bytes) != directory_size:
        raise AssertionError("DDLM directory size changed while assigning offsets")

    temporary_path = path + ".partial"
    try:
        with open(temporary_path, "wb") as output:
            output.write(
                HEADER.pack(
                    MAGIC,
                    ENDIAN_MARKER,
                    FORMAT_VERSION,
                    HEADER.size,
                    len(entries),
                    len(metadata_bytes),
                    directory_offset,
                    data_offset,
                    file_size,
                )
            )
            output.write(metadata_bytes)
            output.write(directory_bytes)
            output.write(b"\x00" * (data_offset - output.tell()))

            for spec, entry in zip(tensors, entries):
                if output.tell() > entry.offset:
                    raise AssertionError(f"writer passed the offset for {entry.name}")
                output.write(b"\x00" * (entry.offset - output.tell()))
                start = output.tell()
                written = spec.write(output)
                actual = output.tell() - start
                if written != actual or actual != entry.length:
                    raise ValueError(
                        f"tensor {entry.name} wrote {actual} bytes (reported {written}), expected {entry.length}"
                    )
            if output.tell() != file_size:
                raise AssertionError(f"DDLM byte accounting mismatch: {output.tell()} != {file_size}")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return entries


def inspect_model(path: str) -> tuple[dict, list[TensorEntry]]:
    file_size = os.path.getsize(path)
    with open(path, "rb") as source:
        raw_header = source.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise ValueError("truncated DDLM header")
        magic, endian, version, header_size, tensor_count, metadata_size, directory_offset, data_offset, declared_size = HEADER.unpack(raw_header)
        if magic != MAGIC:
            raise ValueError("invalid DDLM magic")
        if endian != ENDIAN_MARKER:
            raise ValueError("unsupported DDLM byte order")
        if version != FORMAT_VERSION or header_size != HEADER.size:
            raise ValueError(f"unsupported DDLM format version {version}")
        if declared_size != file_size:
            raise ValueError(f"DDLM size mismatch: header={declared_size}, file={file_size}")
        if directory_offset != HEADER.size + metadata_size:
            raise ValueError("invalid DDLM metadata/directory offsets")
        source.seek(HEADER.size)
        metadata = json.loads(source.read(metadata_size))
        source.seek(directory_offset)
        entries: list[TensorEntry] = []
        names: set[str] = set()
        for _ in range(tensor_count):
            prefix = source.read(DIRECTORY_PREFIX.size)
            if len(prefix) != DIRECTORY_PREFIX.size:
                raise ValueError("truncated DDLM tensor directory")
            name_size, tensor_type, rank, flags, offset, length = DIRECTORY_PREFIX.unpack(prefix)
            if flags != 0 or rank == 0 or rank > 8 or name_size == 0:
                raise ValueError("invalid DDLM tensor directory entry")
            dimensions = struct.unpack(f"<{rank}Q", source.read(rank * 8))
            name = source.read(name_size).decode("utf-8")
            if name in names:
                raise ValueError(f"duplicate DDLM tensor {name}")
            names.add(name)
            if stored_size(dimensions, tensor_type) != length:
                raise ValueError(f"invalid stored size for DDLM tensor {name}")
            if offset < data_offset or offset + length > file_size:
                raise ValueError(f"DDLM tensor {name} lies outside the data section")
            entries.append(TensorEntry(name, dimensions, tensor_type, offset, length))
        ordered = sorted(entries, key=lambda entry: entry.offset)
        previous_end = data_offset
        for entry in ordered:
            if entry.offset < previous_end:
                raise ValueError(f"overlapping DDLM tensor {entry.name}")
            previous_end = entry.offset + entry.length
        if ordered and previous_end != file_size:
            raise ValueError("DDLM file contains unaccounted trailing bytes")
        return metadata, entries
