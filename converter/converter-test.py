#!/usr/bin/env python3

import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest


CONVERTER_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(CONVERTER_DIR, "convert-diffusion-gemma.py")
sys.path.insert(0, CONVERTER_DIR)
from ddllm_writer import inspect_model


def load_converter_module():
    spec = importlib.util.spec_from_file_location("convert_diffusion_gemma", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONVERTER_MODULE = load_converter_module()


def write_safetensors(path, tensors):
    offset = 0
    header = {}
    for name, shape in tensors.items():
        size = 4
        for dimension in shape:
            size *= dimension
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        remaining = offset
        zeros = b"\x00" * 65536
        while remaining:
            chunk = zeros[: min(remaining, len(zeros))]
            output.write(chunk)
            remaining -= len(chunk)


def create_checkpoint(path):
    text = {
        "model_type": "diffusion_gemma_text",
        "hidden_size": 32,
        "intermediate_size": 32,
        "moe_intermediate_size": 32,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 1,
        "head_dim": 32,
        "global_head_dim": 32,
        "num_experts": 2,
        "top_k_experts": 1,
        "vocab_size": 512,
        "num_hidden_layers": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "tie_word_embeddings": True,
        "max_position_embeddings": 128,
        "sliding_window": 4,
        "rms_norm_eps": 1e-6,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
        },
    }
    config = {
        "architectures": ["DiffusionGemmaForBlockDiffusion"],
        "model_type": "diffusion_gemma",
        "tie_word_embeddings": True,
        "canvas_length": 4,
        "eos_token_id": [1, 2],
        "text_config": text,
    }
    generation = {"max_denoising_steps": 2, "t_min": 0.4, "t_max": 0.8}
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as output:
        json.dump(config, output)
    with open(os.path.join(path, "generation_config.json"), "w", encoding="utf-8") as output:
        json.dump(generation, output)

    shard1 = "model-00001-of-00002.safetensors"
    shard2 = "model-00002-of-00002.safetensors"
    tensors = {
        "model.decoder.embed_tokens.weight": (text["vocab_size"], text["hidden_size"]),
        "model.decoder.self_conditioning.pre_norm.weight": (text["hidden_size"],),
        "model.decoder.self_conditioning.gate_proj.weight": (text["intermediate_size"], text["hidden_size"]),
        "model.decoder.self_conditioning.up_proj.weight": (text["intermediate_size"], text["hidden_size"]),
        "model.decoder.self_conditioning.down_proj.weight": (text["hidden_size"], text["intermediate_size"]),
        "model.decoder.norm.weight": (text["hidden_size"],),
    }
    weight_map = {name: shard1 for name in tensors}
    for layer, layer_type in enumerate(text["layer_types"]):
        shapes = CONVERTER_MODULE.expected_layer_shapes(text, layer_type)
        shard = shard1 if layer == 0 else shard2
        for suffix, shape in shapes.items():
            key = f"model.decoder.layers.{layer}.{suffix}"
            weight_map[key] = shard
            if layer == 0:
                tensors[key] = shape
        encoder_scalar = f"model.encoder.language_model.layers.{layer}.layer_scalar"
        weight_map[encoder_scalar] = shard
        if layer == 0:
            tensors[encoder_scalar] = (1,)
    weight_map["model.encoder.vision_tower.fake.weight"] = shard2
    with open(os.path.join(path, "model.safetensors.index.json"), "w", encoding="utf-8") as output:
        json.dump({"metadata": {}, "weight_map": weight_map}, output)
    write_safetensors(os.path.join(path, shard1), tensors)


class ConverterTest(unittest.TestCase):
    def test_layer_parser(self):
        self.assertEqual(CONVERTER_MODULE.parse_layers("0,2-3,1", 4), [0, 2, 3, 1])
        with self.assertRaises(ValueError):
            CONVERTER_MODULE.parse_layers("0,0", 4)

    def test_selective_conversion_is_an_ordinary_model(self):
        with tempfile.TemporaryDirectory() as directory:
            create_checkpoint(directory)
            output = os.path.join(directory, "one-layer.ddlm")
            result = subprocess.run(
                [sys.executable, CONVERTER, directory, output, "--layers", "0", "--type", "f32"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata, tensors = inspect_model(output)
            self.assertEqual(metadata["text_config"]["num_hidden_layers"], 1)
            self.assertEqual(metadata["text_config"]["layer_types"], ["sliding_attention"])
            self.assertEqual(metadata["source_layers"][0]["source_layer"], 0)
            self.assertEqual(len(tensors), 29)
            self.assertEqual(tensors[0].name, "token_embedding.weight")
            self.assertEqual(tensors[-1].name, "layers.0.encoder_layer_scalar")

    def test_unselected_missing_shard_is_not_required(self):
        with tempfile.TemporaryDirectory() as directory:
            create_checkpoint(directory)
            selected = subprocess.run(
                [sys.executable, CONVERTER, directory, "--layers", "0", "--dry-run"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(selected.returncode, 0, selected.stderr)
            manifest = json.loads(selected.stdout)
            self.assertEqual(manifest["required_shards"], ["model-00001-of-00002.safetensors"])

            missing = subprocess.run(
                [sys.executable, CONVERTER, directory, "--layers", "1", "--dry-run"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(missing.returncode, 2)
            self.assertIn("model-00002-of-00002.safetensors", missing.stderr)

    def test_dry_run_rejects_truncated_selected_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            create_checkpoint(directory)
            shard = os.path.join(directory, "model-00001-of-00002.safetensors")
            os.truncate(shard, os.path.getsize(shard) - 1)

            result = subprocess.run(
                [sys.executable, CONVERTER, directory, "--layers", "0", "--dry-run"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("extends beyond the shard", result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "NumPy is not installed")
    def test_q40_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            create_checkpoint(directory)
            output = os.path.join(directory, "one-layer-q40.ddlm")
            result = subprocess.run(
                [sys.executable, CONVERTER, directory, output, "--layers", "0", "--type", "q40"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata, tensors = inspect_model(output)
            self.assertEqual(metadata["weight_type"], "q40")
            self.assertEqual(tensors[0].tensor_type, 2)
            self.assertEqual(tensors[1].tensor_type, 0)


if __name__ == "__main__":
    unittest.main()
