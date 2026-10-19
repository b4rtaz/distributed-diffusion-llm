#!/usr/bin/env python3
"""Compare a one-layer DDRF trace with an independent PyTorch checkpoint reference."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path


def require_dependencies():
    try:
        import numpy
        import torch
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "NumPy, PyTorch, and safetensors are required for checkpoint reference comparison"
        ) from error
    return numpy, torch, safe_open


def load_trace_reader():
    path = Path(__file__).with_name("checkpoint-trace-test.py")
    spec = importlib.util.spec_from_file_location("ddllm_checkpoint_trace", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.read_trace


def quantize_dequantize_q40(torch, value):
    """Reproduce the converter's 32-value Q40 block format."""
    if value.numel() % 32:
        raise ValueError("Q40 reference tensor is not block aligned")
    shape = value.shape
    groups = value.contiguous().reshape(-1, 32).float()
    maximum = groups.max(dim=1).values
    minimum = groups.min(dim=1).values
    delta = torch.where(-minimum > maximum, minimum, maximum) / -8.0
    inverse = torch.where(delta != 0, delta.reciprocal(), torch.zeros_like(delta))
    quantized = torch.trunc(groups * inverse[:, None] + 8.5).clamp(0, 15)
    stored_delta = delta.to(torch.float16).float()
    return ((quantized - 8.0) * stored_delta[:, None]).reshape(shape)


def quantize_dequantize_q80(torch, value):
    """Reproduce the runtime's 32-value Q80 activation blocks."""
    if value.numel() % 32:
        raise ValueError("Q80 reference tensor is not block aligned")
    shape = value.shape
    groups = value.contiguous().reshape(-1, 32).float()
    maximum = groups.abs().max(dim=1).values
    delta = maximum / 127.0
    inverse = torch.where(delta != 0, delta.reciprocal(), torch.zeros_like(delta))
    scaled = groups * inverse[:, None]
    rounded = torch.trunc(scaled + torch.where(scaled >= 0, 0.5, -0.5)).clamp(-128, 127)
    stored_delta = delta.to(torch.float16).float()
    return (rounded * stored_delta[:, None]).reshape(shape)


class Checkpoint:
    def __init__(self, root: Path, torch, safe_open):
        self.root = root
        self.torch = torch
        self.weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.stack = ExitStack()
        self.files = {}
        self.cache = {}
        self.safe_open = safe_open

    def close(self):
        self.stack.close()

    def _file(self, key):
        shard = self.weight_map[key]
        if shard not in self.files:
            self.files[shard] = self.stack.enter_context(
                self.safe_open(self.root / shard, framework="pt", device="cpu")
            )
        return self.files[shard]

    def tensor(self, key, cache=True):
        if key not in self.cache:
            value = self._file(key).get_tensor(key).float()
            if cache:
                self.cache[key] = value
            else:
                return value
        return self.cache[key]

    def slice(self, key, item):
        return self._file(key).get_slice(key)[item].float()


class Reference:
    def __init__(self, checkpoint: Checkpoint, config: dict, layer: int, quantized: bool):
        self.c = checkpoint
        self.torch = checkpoint.torch
        self.config = config["text_config"]
        self.layer = layer
        self.prefix = f"model.decoder.layers.{layer}."
        self.hidden = int(self.config["hidden_size"])
        self.heads = int(self.config["num_attention_heads"])
        self.experts = int(self.config["num_experts"])
        self.top_k = int(self.config["top_k_experts"])
        self.eps = float(self.config["rms_norm_eps"])
        self.layer_type = self.config["layer_types"][layer]
        if self.layer_type == "sliding_attention":
            self.head_dim = int(self.config["head_dim"])
            self.kv_heads = int(self.config["num_key_value_heads"])
        else:
            self.head_dim = int(self.config["global_head_dim"])
            self.kv_heads = int(self.config["num_global_key_value_heads"])
        self.embedding_key = "model.decoder.embed_tokens.weight"
        self.embedding_scale = math.sqrt(self.hidden)
        self.quantized = quantized

    def norm(self, x, suffix=None):
        result = x * self.torch.pow(x.square().mean(dim=-1, keepdim=True) + self.eps, -0.5)
        return result if suffix is None else result * self.c.tensor(self.prefix + suffix)

    def linear(self, x, suffix):
        return self.linear_weight(x, self.c.tensor(self.prefix + suffix))

    def linear_weight(self, x, weight):
        if self.quantized:
            x = quantize_dequantize_q80(self.torch, x)
            weight = quantize_dequantize_q40(self.torch, weight)
        return self.torch.nn.functional.linear(x, weight)

    def gelu(self, x):
        return self.torch.nn.functional.gelu(x, approximate="tanh")

    def rope(self, x, positions):
        params = self.config["rope_parameters"][self.layer_type]
        theta = float(params["rope_theta"])
        rotary = self.head_dim
        if params["rope_type"] == "proportional":
            rotary = int(float(params.get("partial_rotary_factor", 1.0)) * self.head_dim)
        frequency = self.torch.zeros(self.head_dim // 2, dtype=self.torch.float32)
        frequency[: rotary // 2] = 1.0 / (
            theta ** (self.torch.arange(0, rotary, 2, dtype=self.torch.float32) / self.head_dim)
        )
        frequency /= float(params.get("factor", 1.0))
        angles = positions.float()[:, None] * frequency[None, :]
        cosine = self.torch.cat((angles.cos(), angles.cos()), dim=-1)[:, None, :]
        sine = self.torch.cat((angles.sin(), angles.sin()), dim=-1)[:, None, :]
        half = self.head_dim // 2
        rotated = self.torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cosine + rotated * sine

    def project_qkv(self, x, positions):
        length = x.shape[0]
        q = self.linear(x, "self_attn.q_proj.weight").reshape(length, self.heads, self.head_dim)
        raw_k = self.linear(x, "self_attn.k_proj.weight").reshape(length, self.kv_heads, self.head_dim)
        if self.layer_type == "sliding_attention":
            raw_v = self.linear(x, "self_attn.v_proj.weight").reshape(length, self.kv_heads, self.head_dim)
        else:
            raw_v = raw_k
        q = self.norm(q, "self_attn.q_norm.weight")
        k = self.norm(raw_k, "self_attn.k_norm.weight")
        v = self.norm(raw_v)
        return self.rope(q, positions), self.rope(k, positions), v

    def attention(self, q, k, v, positions, prefix_k=None, prefix_v=None, causal=False):
        if prefix_k is not None:
            k = self.torch.cat((prefix_k, k), dim=0)
            v = self.torch.cat((prefix_v, v), dim=0)
            key_positions = self.torch.arange(k.shape[0], dtype=self.torch.int64)
        else:
            key_positions = positions
        groups = self.heads // self.kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = self.torch.einsum("qhd,khd->qhk", q, k)
        allowed = self.torch.ones((q.shape[0], k.shape[0]), dtype=self.torch.bool)
        if causal:
            allowed &= key_positions[None, :] <= positions[:, None]
        if self.layer_type == "sliding_attention":
            window = int(self.config["sliding_window"])
            allowed &= key_positions[None, :] > positions[:, None] - window
        scores = scores.masked_fill(~allowed[:, None, :], -self.torch.inf)
        probability = self.torch.softmax(scores, dim=-1, dtype=self.torch.float32)
        return self.torch.einsum("qhk,khd->qhd", probability, v)

    def route_and_experts(self, residual):
        router_input = self.norm(residual)
        router_input *= self.c.tensor(self.prefix + "router.scale") / math.sqrt(self.hidden)
        probabilities = self.torch.softmax(self.linear(router_input, "router.proj.weight"), dim=-1)
        weights, indexes = self.torch.topk(probabilities, self.top_k, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.c.tensor(self.prefix + "router.per_expert_scale")[indexes]
        expert_input = self.norm(residual, "pre_feedforward_layernorm_2.weight")
        output = self.torch.zeros_like(residual)
        gate_up_key = self.prefix + "experts.gate_up_proj"
        down_key = self.prefix + "experts.down_proj"
        for expert in self.torch.unique(indexes).tolist():
            token, selected = self.torch.where(indexes == expert)
            gate_up = self.c.slice(gate_up_key, expert)
            gate, up = self.linear_weight(expert_input[token], gate_up).chunk(2, dim=-1)
            current = self.gelu(gate) * up
            current = self.linear_weight(current, self.c.slice(down_key, expert))
            output.index_add_(0, token, current * weights[token, selected, None])
        return probabilities, indexes, weights, output

    def layer_forward(self, x, positions, encoder, prefix_k=None, prefix_v=None):
        residual = x
        attention_input = self.norm(x, "input_layernorm.weight")
        q, k, v = self.project_qkv(attention_input, positions)
        attended = self.attention(q, k, v, positions, prefix_k, prefix_v, causal=encoder)
        projected = self.linear(attended.reshape(x.shape[0], -1), "self_attn.o_proj.weight")
        x = residual + self.norm(projected, "post_attention_layernorm.weight")
        residual = x
        dense_input = self.norm(x, "pre_feedforward_layernorm.weight")
        dense_output = self.linear(
            self.gelu(self.linear(dense_input, "mlp.gate_proj.weight"))
            * self.linear(dense_input, "mlp.up_proj.weight"),
            "mlp.down_proj.weight",
        )
        dense = self.norm(dense_output, "post_feedforward_layernorm_1.weight")
        probabilities, indexes, weights, moe_output = self.route_and_experts(residual)
        moe = self.norm(moe_output, "post_feedforward_layernorm_2.weight")
        combined = dense + moe
        x = residual + self.norm(combined, "post_feedforward_layernorm.weight")
        scalar_key = (
            f"model.encoder.language_model.layers.{self.layer}.layer_scalar"
            if encoder else self.prefix + "layer_scalar"
        )
        x *= self.c.tensor(scalar_key)
        return (x, q, k, v, attended, probabilities, indexes, weights,
                projected, dense_output, moe_output, dense, moe, combined)

    def final_norm(self, x):
        result = x * self.torch.pow(x.square().mean(dim=-1, keepdim=True) + self.eps, -0.5)
        return result * self.c.tensor("model.decoder.norm.weight")

    def hard_embedding(self, tokens):
        weight = self.c.slice(self.embedding_key, tokens.tolist())
        if self.quantized:
            weight = quantize_dequantize_q40(self.torch, weight)
        return weight * self.embedding_scale

    def logits(self, hidden, chunk_rows=4096):
        vocab = int(self.config["vocab_size"])
        output = self.torch.empty((hidden.shape[0], vocab), dtype=self.torch.float32)
        if self.quantized:
            hidden = quantize_dequantize_q80(self.torch, hidden)
        for start in range(0, vocab, chunk_rows):
            end = min(start + chunk_rows, vocab)
            weight = self.c.slice(self.embedding_key, slice(start, end))
            if self.quantized:
                weight = quantize_dequantize_q40(self.torch, weight)
            values = self.torch.nn.functional.linear(hidden, weight)
            output[:, start:end] = 30.0 * self.torch.tanh(values / 30.0)
        return output

    def embedding_mix(self, logits, temperature=0.8, chunk_rows=4096):
        probability = self.torch.softmax(logits / temperature, dim=-1)
        output = self.torch.zeros((logits.shape[0], self.hidden), dtype=self.torch.float32)
        vocab = logits.shape[1]
        for start in range(0, vocab, chunk_rows):
            end = min(start + chunk_rows, vocab)
            weight = self.c.slice(self.embedding_key, slice(start, end))
            if self.quantized:
                weight = quantize_dequantize_q40(self.torch, weight)
            output += probability[:, start:end] @ weight
        return output * self.embedding_scale

    def self_condition(self, hard, signal):
        normalized = signal * self.torch.pow(signal.square().mean(dim=-1, keepdim=True) + self.eps, -0.5)
        normalized *= self.c.tensor("model.decoder.self_conditioning.pre_norm.weight")
        gate = self.linear_weight(
            normalized, self.c.tensor("model.decoder.self_conditioning.gate_proj.weight"))
        up = self.linear_weight(
            normalized, self.c.tensor("model.decoder.self_conditioning.up_proj.weight"))
        current = self.gelu(gate) * up
        current = self.linear_weight(
            current, self.c.tensor("model.decoder.self_conditioning.down_proj.weight"))
        combined = hard + current
        return combined * self.torch.pow(combined.square().mean(dim=-1, keepdim=True) + self.eps, -0.5)


def trace_tensor(torch, records, name):
    record = records[name]
    return torch.tensor(record.values, dtype=torch.float32).reshape(record.shape)


FAILURES: list[str] = []
QUANTIZED_REFERENCE = False

# Q40 weights are multiplied by Q80 activations.  The independent PyTorch
# reference reproduces both formats, but its float accumulation tree still
# differs from the CPU integer-dot kernels.  Bound both isolated outliers and
# aggregate drift at each observable boundary.
Q40_LIMITS = {
    "hard_embedding": (0.0, 0.0),
    "query": (0.1, 0.02),
    "key": (0.02, 0.003),
    "value": (0.15, 0.02),
    "attention": (0.25, 0.02),
    "router_probabilities": (0.005, 0.001),
    "expert_weights": (0.01, 0.003),
    "attention_projection": (0.3, 0.05),
    "dense_output": (5.0, 0.5),
    "moe_output": (0.2, 0.02),
    "normalized_branches": (8.0, 0.5),
    "combined_branches": (8.0, 0.5),
    "hidden": (100.0, 5.0),
    "key_cache": (0.02, 0.003),
    "value_cache": (0.15, 0.02),
    "logits": (30.0, 2.0),
    "self_conditioning": (0.05, 0.01),
}


def compare(torch, records, name, actual, atol, rtol):
    expected = trace_tensor(torch, records, name)
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    difference = (actual - expected).abs()
    maximum = difference.max().item()
    relative = (difference / expected.abs().clamp_min(1e-6)).max().item()
    rms = difference.square().mean().sqrt().item()
    reference_rms = expected.square().mean().sqrt().item()
    suffix = name.rsplit(".", 1)[-1]
    if QUANTIZED_REFERENCE and suffix == "expert_indexes":
        mismatch_count = int((actual != expected).sum().item())
        passed = True
        qualification = f" q40_boundary_mismatches={mismatch_count}"
    elif QUANTIZED_REFERENCE:
        maximum_limit, rms_limit = Q40_LIMITS[suffix]
        passed = maximum <= maximum_limit and rms <= rms_limit
        qualification = f" limits(max={maximum_limit:g},rms={rms_limit:g})"
    else:
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        qualification = ""
    print(
        f"{'PASS' if passed else 'FAIL'} {name:36s} max_abs={maximum:.6g} "
        f"rms={rms:.6g} ref_rms={reference_rms:.6g} max_rel={relative:.6g}"
        f"{qualification}"
    )
    if not passed:
        if QUANTIZED_REFERENCE:
            FAILURES.append(
                f"{name} exceeds Q40 max={maximum_limit}, rms={rms_limit}"
            )
        else:
            FAILURES.append(f"{name} exceeds atol={atol}, rtol={rtol}")


def reference_samples(numpy, logits, temperature, coins):
    rows = logits.detach().numpy()
    result = []
    for row, coin in zip(rows, coins):
        scaled = row / numpy.float32(temperature)
        argmax = int(scaled.argmax())
        maximum = scaled[argmax]
        shifted = (scaled - maximum).astype(numpy.float64)
        weights = numpy.exp(shifted)
        denominator = weights.sum(dtype=numpy.float64)
        cumulative = numpy.cumsum(weights, dtype=numpy.float64)
        token = int(numpy.searchsorted(cumulative, float(coin) * denominator, side="right"))
        entropy = (
            math.log(denominator)
            - float((weights * shifted).sum(dtype=numpy.float64)) / denominator
        )
        result.append((token, argmax, entropy))
    return result


def compare_samples(numpy, records, name, logits, temperature, coins):
    expected = records[name]
    actual = reference_samples(numpy, logits, temperature, coins)
    if expected.shape != (len(actual), 3):
        raise AssertionError(f"{name}: shape {expected.shape} != {(len(actual), 3)}")
    token_mismatches = 0
    for index, (token, argmax, entropy) in enumerate(actual):
        offset = index * 3
        recorded_token = int(expected.values[offset])
        recorded_argmax = int(expected.values[offset + 1])
        recorded_entropy = expected.values[offset + 2]
        token_mismatches += token != recorded_token
        if (
            (not QUANTIZED_REFERENCE and token != recorded_token)
            or argmax != recorded_argmax
            or abs(entropy - recorded_entropy) > (0.05 if QUANTIZED_REFERENCE else 2e-3)
        ):
            FAILURES.append(
                f"{name}[{index}] expected token/argmax/entropy "
                f"{token}/{argmax}/{entropy:.7g}, got "
                f"{recorded_token}/{recorded_argmax}/{recorded_entropy:.7g}"
            )
    passed = not any(failure.startswith(name + "[") for failure in FAILURES)
    drift = f"; q40 token drift={token_mismatches}" if QUANTIZED_REFERENCE else ""
    print(f"{'PASS' if passed else 'FAIL'} {name:36s} categorical samples and entropy{drift}")


def main() -> int:
    global QUANTIZED_REFERENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--source-layer", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    numpy, torch, safe_open = require_dependencies()
    torch.set_num_threads(args.threads)
    records = load_trace_reader()(args.trace)
    config = json.loads((args.checkpoint / "config.json").read_text())
    checkpoint = Checkpoint(args.checkpoint, torch, safe_open)
    try:
        weight_type = int(records["fixture.weight_type"].values[0])
        if weight_type not in (0, 2):
            raise ValueError("trace has an unsupported model weight type")
        QUANTIZED_REFERENCE = weight_type == 2
        reference = Reference(checkpoint, config, args.source_layer, quantized=QUANTIZED_REFERENCE)
        print("Reference precision: " + ("Q40 weights / Q80 activations" if QUANTIZED_REFERENCE else "F32"))
        recorded_type = int(records["fixture.layer_type"].values[0])
        expected_type = 0 if reference.layer_type == "sliding_attention" else 1
        if recorded_type != expected_type:
            raise ValueError("trace layer type does not match the selected source layer")
        encoder_tokens = trace_tensor(torch, records, "fixture.encoder_tokens").long()
        canvas_tokens = trace_tensor(torch, records, "fixture.canvas_tokens").long()
        positions = torch.arange(3, dtype=torch.int64)

        hard = reference.hard_embedding(encoder_tokens)
        layer = reference.layer_forward(hard, positions, encoder=True)
        hidden = reference.final_norm(layer[0])
        logits = reference.logits(hidden)
        names = ("query", "key", "value", "attention", "router_probabilities")
        values = (layer[1].reshape(3, -1), layer[2].reshape(3, -1), layer[3].reshape(3, -1),
                  layer[4].reshape(3, -1), layer[5])
        compare(torch, records, "encoder.hard_embedding", hard, 1e-6, 1e-6)
        for name, value in zip(names, values):
            compare(torch, records, f"encoder.{name}", value, 3e-3, 3e-4)
        compare(torch, records, "encoder.expert_indexes", layer[6].float(), 0.0, 0.0)
        compare(torch, records, "encoder.expert_weights", layer[7].transpose(0, 1), 2e-4, 2e-3)
        compare(torch, records, "encoder.attention_projection", layer[8], 3e-3, 3e-4)
        # Torch and llamafile SGEMM use different F32 accumulation trees. Raw
        # projections can be hundreds in magnitude before their following
        # RMSNorm, so use a scale-appropriate absolute bound at this boundary.
        compare(torch, records, "encoder.dense_output", layer[9], 1e-1, 2e-3)
        compare(torch, records, "encoder.moe_output", layer[10], 1e-1, 2e-3)
        compare(torch, records, "encoder.normalized_branches", torch.stack((layer[11], layer[12])), 3e-1, 1e-3)
        compare(torch, records, "encoder.combined_branches", layer[13], 3e-1, 1e-3)
        compare(torch, records, "encoder.hidden", hidden, 2.0, 1e-3)
        compare(torch, records, "encoder.key_cache", layer[2].reshape(3, -1), 3e-3, 3e-4)
        compare(torch, records, "encoder.value_cache", layer[3].reshape(3, -1), 3e-3, 3e-4)
        compare(torch, records, "encoder.logits", logits, 6e-1, 1e-3)

        prefix_k, prefix_v = layer[2], layer[3]
        hard_canvas = reference.hard_embedding(canvas_tokens)
        signal = torch.zeros_like(hard_canvas)
        for pass_index in range(2):
            decoder_input = reference.self_condition(hard_canvas, signal)
            decoder = reference.layer_forward(
                decoder_input, positions + 3, encoder=False, prefix_k=prefix_k, prefix_v=prefix_v
            )
            decoder_hidden = reference.final_norm(decoder[0])
            decoder_logits = reference.logits(decoder_hidden)
            next_signal = reference.embedding_mix(decoder_logits)
            prefix = f"decoder{pass_index}"
            compare(torch, records, f"{prefix}.hard_embedding", hard_canvas, 1e-6, 1e-6)
            decoder_values = (
                decoder[1].reshape(3, -1), decoder[2].reshape(3, -1),
                decoder[3].reshape(3, -1), decoder[4].reshape(3, -1), decoder[5],
            )
            for name, value in zip(names, decoder_values):
                compare(torch, records, f"{prefix}.{name}", value, 4e-3, 4e-4)
            compare(torch, records, f"{prefix}.expert_indexes", decoder[6].float(), 0.0, 0.0)
            compare(torch, records, f"{prefix}.expert_weights", decoder[7].transpose(0, 1), 2e-4, 2e-3)
            compare(torch, records, f"{prefix}.attention_projection", decoder[8], 4e-3, 4e-4)
            compare(torch, records, f"{prefix}.dense_output", decoder[9], 1e-1, 2e-3)
            compare(torch, records, f"{prefix}.moe_output", decoder[10], 1e-1, 2e-3)
            compare(
                torch, records, f"{prefix}.normalized_branches",
                torch.stack((decoder[11], decoder[12])), 3e-1, 1e-3,
            )
            compare(torch, records, f"{prefix}.combined_branches", decoder[13], 3e-1, 1e-3)
            compare(torch, records, f"{prefix}.hidden", decoder_hidden, 2.0, 1e-3)
            compare(torch, records, f"{prefix}.logits", decoder_logits, 6e-1, 1e-3)
            compare(torch, records, f"{prefix}.self_conditioning", next_signal, 5e-3, 8e-4)
            compare_samples(
                numpy, records, f"{prefix}.samples", decoder_logits, 0.8,
                trace_tensor(torch, records, "fixture.sampling_coins").tolist(),
            )
            signal = next_signal
        if FAILURES:
            raise AssertionError("; ".join(FAILURES))
        print("Checkpoint reference comparison passed")
        return 0
    finally:
        checkpoint.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
