import asyncio
import base64
import inspect
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _parse_int_list_env(name: str, default: str) -> list[int]:
    value = os.getenv(name, default).strip()
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_env(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _parse_float_env(name: str, default: str) -> float:
    return float(os.getenv(name, default))


app = FastAPI(
    title="NanoVLLM VoxCPM API",
    description="Text-to-speech API powered by NanoVLLM-VoxCPM with voice design and voice cloning support.",
    version="3.0.0",
)

MODEL_ID = os.getenv("NANOVLLM_MODEL_PATH", os.getenv("MODEL_PATH", "openbmb/VoxCPM2"))
OUTPUT_SAMPLE_RATE = os.getenv("OUTPUT_SAMPLE_RATE")
DEFAULT_OUTPUT_FORMAT = os.getenv("DEFAULT_AUDIO_FORMAT", "mp3").lower()
TORCH_MATMUL_PRECISION = os.getenv("TORCH_MATMUL_PRECISION", "high").lower()

NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS = _parse_int_env("NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS", "8192")
NANOVLLM_SERVERPOOL_MAX_NUM_SEQS = _parse_int_env("NANOVLLM_SERVERPOOL_MAX_NUM_SEQS", "16")
NANOVLLM_SERVERPOOL_MAX_MODEL_LEN = _parse_int_env("NANOVLLM_SERVERPOOL_MAX_MODEL_LEN", "4096")
NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION = _parse_float_env("NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION", "0.95")
NANOVLLM_SERVERPOOL_ENFORCE_EAGER = _parse_bool_env("NANOVLLM_SERVERPOOL_ENFORCE_EAGER", "false")
NANOVLLM_SERVERPOOL_DEVICES = _parse_int_list_env("NANOVLLM_SERVERPOOL_DEVICES", "0")

tts_model: Any = None
tts_model_lock = asyncio.Lock()

if TORCH_MATMUL_PRECISION in {"highest", "high", "medium"}:
    torch.set_float32_matmul_precision(TORCH_MATMUL_PRECISION)
    logger.info("Set torch float32 matmul precision to %s", TORCH_MATMUL_PRECISION)
else:
    logger.warning("Ignoring invalid TORCH_MATMUL_PRECISION=%s", TORCH_MATMUL_PRECISION)


@dataclass
class LatentEncoding:
    raw: Any
    base64: Optional[str]
    feat_dim: Optional[int]
    sample_rate: Optional[int]
    channels: Optional[int]


def _normalize_output_format(value: Optional[str]) -> Literal["mp3", "wav"]:
    if value and value.lower() == "wav":
        return "wav"
    return "mp3"


def _guess_upload_suffix(upload: UploadFile) -> str:
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix in {".wav", ".wave", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}:
        return suffix

    content_type = (upload.content_type or "").lower()
    if "mpeg" in content_type or "mp3" in content_type:
        return ".mp3"
    return ".wav"


def _save_audio_upload_as_wav(upload: UploadFile, sample_rate: int = 16000) -> str:
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Reference audio is empty.")

    input_suffix = _guess_upload_suffix(upload)
    with tempfile.NamedTemporaryFile(suffix=input_suffix, delete=False) as tmp_input:
        tmp_input.write(raw)
        input_path = tmp_input.name

    output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
        audio.export(output_path, format="wav")
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass

    return output_path


def _save_bytes_as_wav(raw: bytes, suffix: str = ".wav", sample_rate: int = 16000) -> str:
    if not raw:
        raise HTTPException(status_code=400, detail="Audio payload is empty.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_input:
        tmp_input.write(raw)
        input_path = tmp_input.name

    output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
        audio.export(output_path, format="wav")
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass

    return output_path


def _decode_base64_bytes(value: str) -> bytes:
    return base64.b64decode(value.encode("utf-8"))


def _apply_control_instruction(text: str, control_instruction: Optional[str]) -> str:
    instruction = (control_instruction or "").strip()
    if not instruction:
        return text

    if not (instruction.startswith("(") and instruction.endswith(")")):
        instruction = f"({instruction})"

    return f"{instruction}{text}"


def _get_output_sample_rate(model: Any = None) -> int:
    if OUTPUT_SAMPLE_RATE:
        return int(OUTPUT_SAMPLE_RATE)

    for candidate in (model, getattr(model, "tts_model", None), getattr(model, "core", None)):
        sample_rate = getattr(candidate, "sample_rate", None)
        if sample_rate:
            return int(sample_rate)

    return 16000


def _get_output_channels(model: Any = None) -> int:
    for candidate in (model, getattr(model, "tts_model", None), getattr(model, "core", None)):
        channels = getattr(candidate, "channels", None)
        if channels:
            return int(channels)
    return 1


def _get_model_feat_dim(model: Any = None) -> Optional[int]:
    for candidate in (model, getattr(model, "tts_model", None), getattr(model, "core", None)):
        feat_dim = getattr(candidate, "feat_dim", None)
        if feat_dim:
            return int(feat_dim)
    return None


def _filter_supported_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    supported = set(signature.parameters)
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_var_kwargs:
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if key in supported and value is not None}


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _normalize_latent_payload(value: Any, feat_dim: Optional[int] = None) -> LatentEncoding:
    payload = value
    base64_payload: Optional[str] = None
    inferred_feat_dim = feat_dim
    sample_rate: Optional[int] = None
    channels: Optional[int] = None

    if isinstance(value, dict):
        sample_rate = value.get("sample_rate")
        channels = value.get("channels")
        inferred_feat_dim = value.get("feat_dim", inferred_feat_dim)
        for key in ("prompt_latents_base64", "ref_audio_latents_base64", "latents_base64", "base64"):
            if key in value and value[key] is not None:
                base64_payload = value[key]
                payload = np.frombuffer(base64.b64decode(base64_payload), dtype=np.float32)
                if inferred_feat_dim is None and payload.ndim > 1:
                    inferred_feat_dim = int(payload.shape[-1])
                return LatentEncoding(payload, base64_payload, inferred_feat_dim, sample_rate, channels)
        for key in ("prompt_latents", "ref_audio_latents", "latents"):
            if key in value and value[key] is not None:
                payload = value[key]
                break

    if isinstance(payload, str):
        base64_payload = payload
        try:
            payload = np.frombuffer(base64.b64decode(payload), dtype=np.float32)
        except Exception:
            payload = value
    elif isinstance(payload, bytes):
        payload = np.frombuffer(payload, dtype=np.float32)
        base64_payload = base64.b64encode(payload.tobytes()).decode("utf-8")
    elif isinstance(payload, bytearray):
        payload = np.frombuffer(bytes(payload), dtype=np.float32)
        base64_payload = base64.b64encode(payload.tobytes()).decode("utf-8")
    elif isinstance(payload, np.ndarray):
        array = np.asarray(payload, dtype=np.float32)
        if inferred_feat_dim is None and array.ndim > 1:
            inferred_feat_dim = int(array.shape[-1])
        base64_payload = base64.b64encode(array.tobytes()).decode("utf-8")
    else:
        array = np.asarray(payload, dtype=np.float32) if payload is not None else None
        if array is not None and array.size:
            if inferred_feat_dim is None and array.ndim > 1:
                inferred_feat_dim = int(array.shape[-1])
            base64_payload = base64.b64encode(array.tobytes()).decode("utf-8")

    return LatentEncoding(payload, base64_payload, inferred_feat_dim, sample_rate, channels)


async def load_model() -> Any:
    global tts_model
    if tts_model is not None:
        return tts_model

    async with tts_model_lock:
        if tts_model is not None:
            return tts_model

        logger.info("Loading NanoVLLM VoxCPM model from %s", MODEL_ID)
        from nanovllm_voxcpm import VoxCPM

        from_pretrained_kwargs = _filter_supported_kwargs(
            VoxCPM.from_pretrained,
            {
                "model": MODEL_ID,
                "model_path": MODEL_ID,
                "devices": NANOVLLM_SERVERPOOL_DEVICES or [0],
                "max_num_batched_tokens": NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS,
                "max_num_seqs": NANOVLLM_SERVERPOOL_MAX_NUM_SEQS,
                "max_model_len": NANOVLLM_SERVERPOOL_MAX_MODEL_LEN,
                "gpu_memory_utilization": NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION,
                "enforce_eager": NANOVLLM_SERVERPOOL_ENFORCE_EAGER,
            },
        )

        tts_model = VoxCPM.from_pretrained(**from_pretrained_kwargs)
        wait_for_ready = getattr(tts_model, "wait_for_ready", None)
        if callable(wait_for_ready):
            await _maybe_await(wait_for_ready())
        logger.info("NanoVLLM VoxCPM model loaded successfully.")
        return tts_model


async def shutdown_model() -> None:
    global tts_model
    if tts_model is None:
        return

    stop = getattr(tts_model, "stop", None)
    if callable(stop):
        try:
            await _maybe_await(stop())
        except Exception:
            logger.exception("Failed to stop NanoVLLM model cleanly")
    tts_model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    await load_model()
    try:
        yield
    finally:
        await shutdown_model()


app.router.lifespan_context = lifespan


@app.on_event("startup")
async def startup_event():
    await load_model()


@app.on_event("shutdown")
async def shutdown_event():
    await shutdown_model()


def _iter_audio_chunks(wav_array: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav_array, dtype=np.float32).squeeze()
    if wav.ndim != 1:
        raise ValueError("NanoVLLM returned an invalid waveform shape.")
    return wav


def _waveform_to_response(
    wav_array: np.ndarray,
    sample_rate: int,
    output_format: Literal["mp3", "wav"],
    filename_stem: str,
) -> StreamingResponse:
    wav = _iter_audio_chunks(wav_array)
    headers = {
        "Content-Disposition": f"attachment; filename={filename_stem}.{output_format}",
        "X-Audio-Sample-Rate": str(sample_rate),
        "X-Audio-Channels": "1",
    }

    if output_format == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, wav, sample_rate, format="WAV")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/wav", headers=headers)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        sf.write(tmp_wav.name, wav, sample_rate)
        wav_path = tmp_wav.name

    try:
        audio = AudioSegment.from_file(wav_path, format="wav")
        mp3_buffer = io.BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate="192k")
        mp3_buffer.seek(0)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    return StreamingResponse(mp3_buffer, media_type="audio/mpeg", headers=headers)


async def _generate_waveform(
    *,
    target_text: str,
    control_instruction: Optional[str] = None,
    prompt_text: Optional[str] = None,
    prompt_latents: Optional[LatentEncoding] = None,
    prompt_wav_base64: Optional[str] = None,
    prompt_wav_format: Optional[str] = None,
    ref_audio_latents: Optional[LatentEncoding] = None,
    ref_audio_wav_base64: Optional[str] = None,
    ref_audio_wav_format: Optional[str] = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = False,
    denoise: bool = False,
) -> np.ndarray:
    model = await load_model()
    target_text = _apply_control_instruction(target_text, control_instruction)

    generate_fn = getattr(model, "generate", None)
    if not callable(generate_fn):
        raise HTTPException(status_code=500, detail="NanoVLLM model does not expose a generate() method.")

    signature = inspect.signature(generate_fn)
    supported = set(signature.parameters)

    kwargs: dict[str, Any] = {}
    if "target_text" in supported:
        kwargs["target_text"] = target_text
    elif "text" in supported:
        kwargs["text"] = target_text
    else:
        kwargs["target_text"] = target_text

    for key, value in {
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
        "normalize": bool(normalize),
        "denoise": bool(denoise),
        "prompt_text": prompt_text,
        "prompt_wav_base64": prompt_wav_base64,
        "prompt_wav_format": prompt_wav_format,
        "ref_audio_wav_base64": ref_audio_wav_base64,
        "ref_audio_wav_format": ref_audio_wav_format,
    }.items():
        if key in supported and value is not None:
            kwargs[key] = value

    if prompt_latents is not None:
        if "prompt_latents_base64" in supported and prompt_latents.base64 is not None:
            kwargs["prompt_latents_base64"] = prompt_latents.base64
        elif "prompt_latents" in supported:
            kwargs["prompt_latents"] = prompt_latents.raw
    if ref_audio_latents is not None:
        if "ref_audio_latents_base64" in supported and ref_audio_latents.base64 is not None:
            kwargs["ref_audio_latents_base64"] = ref_audio_latents.base64
        elif "ref_audio_latents" in supported:
            kwargs["ref_audio_latents"] = ref_audio_latents.raw

    if "prompt_text" in supported and prompt_text is not None:
        kwargs.setdefault("prompt_text", prompt_text)

    if "prompt_wav_base64" in supported and prompt_wav_base64 is not None:
        kwargs.setdefault("prompt_wav_base64", prompt_wav_base64)
    if "prompt_wav_format" in supported and prompt_wav_format is not None:
        kwargs.setdefault("prompt_wav_format", prompt_wav_format)
    if "ref_audio_wav_base64" in supported and ref_audio_wav_base64 is not None:
        kwargs.setdefault("ref_audio_wav_base64", ref_audio_wav_base64)
    if "ref_audio_wav_format" in supported and ref_audio_wav_format is not None:
        kwargs.setdefault("ref_audio_wav_format", ref_audio_wav_format)

    result = generate_fn(**kwargs)
    wav_chunks: list[np.ndarray] = []

    if hasattr(result, "__aiter__"):
        async for chunk in result:
            wav_chunks.append(np.asarray(chunk, dtype=np.float32))
    else:
        maybe_result = await _maybe_await(result)
        if isinstance(maybe_result, np.ndarray):
            wav_chunks.append(np.asarray(maybe_result, dtype=np.float32))
        elif isinstance(maybe_result, (list, tuple)):
            for chunk in maybe_result:
                wav_chunks.append(np.asarray(chunk, dtype=np.float32))
        elif hasattr(maybe_result, "__iter__") and not isinstance(maybe_result, (str, bytes, bytearray)):
            for chunk in maybe_result:
                wav_chunks.append(np.asarray(chunk, dtype=np.float32))
        else:
            wav_chunks.append(np.asarray(maybe_result, dtype=np.float32))

    if not wav_chunks:
        raise HTTPException(status_code=500, detail="NanoVLLM did not return any audio.")

    return np.concatenate(wav_chunks, axis=0)


async def _encode_prompt_latents_from_wav_path(wav_path: str) -> LatentEncoding:
    model = await load_model()
    for method_name in ("encode_latents", "encode_prompt_latents", "encode_prompt_audio", "encode_audio"):
        method = getattr(model, method_name, None)
        if not callable(method):
            continue

        kwargs = _filter_supported_kwargs(
            method,
            {
                "wav_path": wav_path,
                "audio_path": wav_path,
                "path": wav_path,
                "reference_audio_path": wav_path,
                "wav": wav_path,
            },
        )
        if not kwargs:
            try:
                result = method(wav_path)
            except TypeError:
                continue
        else:
            result = method(**kwargs)

        result = await _maybe_await(result)
        latent = _normalize_latent_payload(result, _get_model_feat_dim(model))
        if latent.base64 is None and latent.raw is None:
            continue
        return latent

    raise HTTPException(
        status_code=501,
        detail="This NanoVLLM model does not expose an audio-to-latents encoder.",
    )


class GenerateRequest(BaseModel):
    target_text: str
    control_instruction: Optional[str] = None
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    normalize: bool = False
    denoise: bool = False
    output_format: Literal["mp3", "wav"] = DEFAULT_OUTPUT_FORMAT if DEFAULT_OUTPUT_FORMAT in {"mp3", "wav"} else "mp3"
    prompt_text: Optional[str] = None
    prompt_latents_base64: Optional[str] = None
    prompt_wav_base64: Optional[str] = None
    prompt_wav_format: Optional[str] = None
    ref_audio_latents_base64: Optional[str] = None
    ref_audio_wav_base64: Optional[str] = None
    ref_audio_wav_format: Optional[str] = None


class EncodeLatentsRequest(BaseModel):
    wav_base64: str
    wav_format: str = "wav"


@app.post(
    "/generate",
    response_class=StreamingResponse,
    summary="Generate speech from text",
    description="Generate speech using NanoVLLM-VoxCPM. Use control_instruction for voice design.",
)
async def generate(request: GenerateRequest):
    if not request.target_text.strip():
        raise HTTPException(status_code=400, detail="Target text cannot be empty.")

    prompt_latents = None
    ref_audio_latents = None

    if request.prompt_latents_base64:
        prompt_latents = _normalize_latent_payload(request.prompt_latents_base64)
    elif request.prompt_wav_base64:
        prompt_wav_path = None
        try:
            prompt_wav_path = _save_bytes_as_wav(
                _decode_base64_bytes(request.prompt_wav_base64),
                suffix=f".{request.prompt_wav_format or 'wav'}",
            )
            prompt_latents = await _encode_prompt_latents_from_wav_path(prompt_wav_path)
        finally:
            if prompt_wav_path and os.path.exists(prompt_wav_path):
                try:
                    os.unlink(prompt_wav_path)
                except OSError:
                    pass

    if request.ref_audio_latents_base64:
        ref_audio_latents = _normalize_latent_payload(request.ref_audio_latents_base64)
    elif request.ref_audio_wav_base64:
        ref_wav_path = None
        try:
            ref_wav_path = _save_bytes_as_wav(
                _decode_base64_bytes(request.ref_audio_wav_base64),
                suffix=f".{request.ref_audio_wav_format or 'wav'}",
            )
            ref_audio_latents = await _encode_prompt_latents_from_wav_path(ref_wav_path)
        finally:
            if ref_wav_path and os.path.exists(ref_wav_path):
                try:
                    os.unlink(ref_wav_path)
                except OSError:
                    pass

    try:
        wav = await _generate_waveform(
            target_text=request.target_text,
            control_instruction=request.control_instruction,
            prompt_text=request.prompt_text,
            prompt_latents=prompt_latents,
            prompt_wav_base64=request.prompt_wav_base64,
            prompt_wav_format=request.prompt_wav_format,
            ref_audio_latents=ref_audio_latents,
            ref_audio_wav_base64=request.ref_audio_wav_base64,
            ref_audio_wav_format=request.ref_audio_wav_format,
            cfg_value=request.cfg_value,
            inference_timesteps=request.inference_timesteps,
            normalize=request.normalize,
            denoise=request.denoise,
        )
        return _waveform_to_response(
            wav,
            _get_output_sample_rate(await load_model()),
            request.output_format,
            "speech",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("NanoVLLM generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


class TTSRequest(BaseModel):
    text: str
    control_instruction: Optional[str] = None
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    normalize: bool = False
    denoise: bool = False
    output_format: Literal["mp3", "wav"] = DEFAULT_OUTPUT_FORMAT if DEFAULT_OUTPUT_FORMAT in {"mp3", "wav"} else "mp3"


@app.post(
    "/tts",
    response_class=StreamingResponse,
    summary="Generate speech from text",
    description="Compatibility alias for /generate.",
)
async def text_to_speech(request: TTSRequest):
    generate_request = GenerateRequest(
        target_text=request.text,
        control_instruction=request.control_instruction,
        cfg_value=request.cfg_value,
        inference_timesteps=request.inference_timesteps,
        normalize=request.normalize,
        denoise=request.denoise,
        output_format=request.output_format,
    )
    return await generate(generate_request)


@app.post(
    "/encode_latents",
    summary="Encode prompt audio to latents",
    description="Encode audio bytes into prompt latents compatible with NanoVLLM-VoxCPM.",
)
async def encode_latents(request: EncodeLatentsRequest):
    wav_path = None
    try:
        wav_path = _save_bytes_as_wav(_decode_base64_bytes(request.wav_base64), suffix=f".{request.wav_format}")
        latent = await _encode_prompt_latents_from_wav_path(wav_path)

        if latent.base64 is None:
            raise HTTPException(status_code=500, detail="Latent encoder did not return a base64 payload.")

        return {
            "prompt_latents_base64": latent.base64,
            "feat_dim": latent.feat_dim or _get_model_feat_dim(await load_model()) or 0,
            "latents_dtype": "float32",
            "sample_rate": _get_output_sample_rate(await load_model()),
            "channels": _get_output_channels(await load_model()),
        }
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


@app.post(
    "/tts/clone",
    response_class=StreamingResponse,
    summary="Clone a voice from reference audio",
    description="Compatibility alias for prompt-conditioned generation using reference audio.",
)
async def clone_voice(
    text: str = Form(..., description="Text to synthesize"),
    reference_audio: UploadFile = File(..., description="Reference audio file (WAV, MP3, or other supported formats)"),
    prompt_text: str = Form(..., description="Transcript for the reference audio"),
    control_instruction: Optional[str] = Form(None, description="Optional style control, e.g. 'slightly faster, cheerful tone'"),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
    normalize: bool = Form(False),
    denoise: bool = Form(False),
    output_format: Literal["mp3", "wav"] = Form(DEFAULT_OUTPUT_FORMAT if DEFAULT_OUTPUT_FORMAT in {"mp3", "wav"} else "mp3"),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if reference_audio.content_type and not reference_audio.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {reference_audio.content_type}",
        )

    ref_path = None
    try:
        ref_path = _save_audio_upload_as_wav(reference_audio)
        ref_latents = await _encode_prompt_latents_from_wav_path(ref_path)

        wav = await _generate_waveform(
            target_text=text,
            control_instruction=control_instruction,
            prompt_text=prompt_text.strip(),
            prompt_latents=ref_latents,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
        )

        return _waveform_to_response(
            wav,
            _get_output_sample_rate(await load_model()),
            output_format,
            "cloned_speech",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Voice cloning failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if ref_path and os.path.exists(ref_path):
            try:
                os.unlink(ref_path)
            except OSError:
                pass


@app.get("/info", summary="Model info")
async def info():
    model = await load_model()
    return {
        "model": MODEL_ID,
        "model_loaded": model is not None,
        "backend": "nanovllm_voxcpm",
        "sample_rate": _get_output_sample_rate(model),
        "channels": _get_output_channels(model),
        "feat_dim": _get_model_feat_dim(model),
    }


@app.get("/ready", summary="Readiness probe")
async def ready():
    model = await load_model()
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/health", summary="Health check")
async def health():
    model = tts_model
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_loaded": model is not None,
        "backend": "nanovllm_voxcpm",
        "output_sample_rate": _get_output_sample_rate(model),
    }
