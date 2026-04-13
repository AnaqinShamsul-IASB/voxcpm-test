# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    NANOVLLM_MODEL_PATH=openbmb/VoxCPM2 \
    NANOVLLM_SERVERPOOL_DEVICES=0 \
    TORCH_MATMUL_PRECISION=high \
    DEFAULT_AUDIO_FORMAT=mp3

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 python3 python3-pip python3-dev python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install numpy packaging psutil setuptools wheel \
    && python -m pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install --no-build-isolation -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
