# Distributed Diffusion LLM

This project is a fork of [Distributed Llama](https://github.com/b4rtaz/distributed-llama), adapted to run Google's Diffusion Gemma 26B A4B model in a distributed environment. DDLLM supports inference on CPUs with ARM or AVX2 optimizations, as well as on Vulkan-compatible GPUs. It can run inference across 1, 2, 4, or 8 nodes using weights in float32 or Q40 format.

### 🔥 Setup

Build the project on the root and worker nodes:

```bash
make -j
```

Start the worker first. Its port must be reachable from the root:

```bash
./ddllm worker --host 0.0.0.0 --port 9990 --nthreads 8
```

On the root, download the model and start inference using the worker's IP address:

```bash
python3 launch.py

./ddllm inference \
  --model models/diffusion-gemma-q40.ddlm \
  --tokenizer models/diffusion-gemma.ddlt \
  --workers 192.168.1.20:9990 \
  --prompt "Hello" \
  --steps 64 \
  --nthreads 8
```

The root sends the model weights to workers, so the model files are needed only on the root.

## 💡 License

This project is released under the MIT license.
