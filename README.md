# voxcpm-test

FastAPI wrapper around NanoVLLM-VoxCPM for text-to-speech and voice cloning.

## Run locally

1. Create and activate a virtual environment.
2. Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Start the API:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4. Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Notes:

- NanoVLLM-VoxCPM is GPU-centric and expects Linux + CUDA + FlashAttention.
- `requirements.txt` now installs `nano-vllm-voxcpm` instead of the old PyTorch VoxCPM wrapper.
- If you only have CPU-only PyTorch available, the model will not run.
- The main model path is read from `NANOVLLM_MODEL_PATH` and defaults to `openbmb/VoxCPM2`.
- `NANOVLLM_SERVERPOOL_DEVICES` controls which GPUs the server pool uses, for example `0` or `0,1`.
- The API now exposes `POST /generate` in addition to the compatibility aliases `/tts`, `/tts/clone`, and `POST /encode_latents`.

## Build and run with Docker

Build the image:

```powershell
set DOCKER_BUILDKIT=1
docker build -t voxcpm-test .
```

BuildKit lets Docker reuse the pip download cache between builds, so unchanged dependencies do not have to be fetched again.

Run the container with GPU access:

```powershell
docker run --rm -p 8000:8000 --gpus all voxcpm-test
```

To avoid redownloading the model on every run, mount a persistent Hugging Face cache volume:

```powershell
docker volume create voxcpm-hf-cache
docker run --rm -p 8000:8000 --gpus all `
  -v voxcpm-hf-cache:/cache/huggingface `
  voxcpm-test
```

If you want to override the model path or server-pool settings, pass environment variables:

```powershell
docker run --rm -p 8000:8000 --gpus all `
  -v voxcpm-hf-cache:/cache/huggingface `
  -e NANOVLLM_MODEL_PATH=openbmb/VoxCPM2 `
  -e NANOVLLM_SERVERPOOL_DEVICES=0 `
  -e NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS=8192 `
  -e NANOVLLM_SERVERPOOL_MAX_NUM_SEQS=16 `
  -e NANOVLLM_SERVERPOOL_MAX_MODEL_LEN=4096 `
  -e NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION=0.95 `
  -e DEFAULT_AUDIO_FORMAT=mp3 `
  voxcpm-test
```

## Endpoints

- `POST /generate` for NanoVLLM-style text-to-speech generation
- `POST /tts` for compatibility with the previous API
- `POST /tts/clone` for compatibility with reference-audio voice cloning
- `POST /encode_latents` to encode prompt audio into latents
- `GET /info` for model metadata
- `GET /ready` for readiness
- `GET /health` for a simple health check

For voice design, pass `control_instruction` and it will be prepended in parentheses to the text prompt, for example `(young woman, gentle and sweet)Hello, welcome to VoxCPM!`.
