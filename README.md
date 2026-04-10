# voxcpm-test

FastAPI wrapper around VoxCPM2 for text-to-speech and voice cloning.

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

- The project expects a CUDA-enabled PyTorch install for GPU inference.
- `requirements.txt` uses the PyTorch CUDA wheel index for `cu130`.
- If you only have CPU PyTorch installed, VoxCPM may fail during model warmup.
- If `torch.compile` causes startup issues, set `VOXCPM_OPTIMIZE=false` before launching the app.
- The denoiser is enabled by default; set `ENABLE_DENOISER=false` if you want to disable it.
- VoxCPM2 reports a native output sample rate of `48000`, so the app now uses that automatically unless you override `OUTPUT_SAMPLE_RATE`.
- The Docker image sets `TORCH_MATMUL_PRECISION=high`, which enables TF32-accelerated float32 matmuls on supported NVIDIA GPUs.

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

If you want to override the model path or audio settings, pass environment variables:

```powershell
docker run --rm -p 8000:8000 --gpus all `
  -v voxcpm-hf-cache:/cache/huggingface `
  -e MODEL_PATH=openbmb/VoxCPM2 `
  -e VOXCPM_OPTIMIZE=true `
  -e TORCH_MATMUL_PRECISION=high `
  -e ENABLE_DENOISER=false `
  -e DEFAULT_AUDIO_FORMAT=mp3 `
  voxcpm-test
```

## Endpoints

- `POST /tts` for text-to-speech
- `POST /tts/clone` for voice cloning with a reference audio file
- `GET /health` for a simple health check

For voice design, pass `control_instruction` and it will be prepended in parentheses to the text prompt, for example `(young woman, gentle and sweet)Hello, welcome to VoxCPM!`.
