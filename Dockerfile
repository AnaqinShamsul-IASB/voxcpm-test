# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    MODEL_PATH=openbmb/VoxCPM2 \
    REFERENCE_SAMPLE_RATE=48000 \
    TORCH_MATMUL_PRECISION=high \
    DEFAULT_AUDIO_FORMAT=mp3

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
