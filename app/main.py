import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VoxCPM2 API",
    description="Text-to-speech API powered by VoxCPM2 with voice design and voice cloning support.",
    version="2.0.0",
)

MODEL_ID = os.getenv("MODEL_PATH", "openbmb/VoxCPM2")
OUTPUT_SAMPLE_RATE = os.getenv("OUTPUT_SAMPLE_RATE")
REFERENCE_SAMPLE_RATE = int(os.getenv("REFERENCE_SAMPLE_RATE", "16000"))
DEFAULT_OUTPUT_FORMAT = os.getenv("DEFAULT_AUDIO_FORMAT", "mp3").lower()
VOXCPM_OPTIMIZE = os.getenv("VOXCPM_OPTIMIZE", "true").lower() in {"1", "true", "yes", "on"}
ENABLE_DENOISER = os.getenv("ENABLE_DENOISER", "true").lower() in {"1", "true", "yes", "on"}
TORCH_MATMUL_PRECISION = os.getenv("TORCH_MATMUL_PRECISION", "high").lower()

tts_model = None

if TORCH_MATMUL_PRECISION in {"highest", "high", "medium"}:
    torch.set_float32_matmul_precision(TORCH_MATMUL_PRECISION)
    logger.info("Set torch float32 matmul precision to %s", TORCH_MATMUL_PRECISION)
else:
    logger.warning("Ignoring invalid TORCH_MATMUL_PRECISION=%s", TORCH_MATMUL_PRECISION)


def load_model():
    global tts_model
    if tts_model is None:
        logger.info("Loading VoxCPM model from %s", MODEL_ID)
        from voxcpm import VoxCPM

        tts_model = VoxCPM.from_pretrained(
            MODEL_ID,
            load_denoiser=ENABLE_DENOISER,
            optimize=VOXCPM_OPTIMIZE,
        )
        logger.info("VoxCPM model loaded successfully. Denoiser enabled: %s", ENABLE_DENOISER)
    return tts_model


def _guess_upload_suffix(upload: UploadFile) -> str:
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix in {".wav", ".wave", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}:
        return suffix

    content_type = (upload.content_type or "").lower()
    if "mpeg" in content_type or "mp3" in content_type:
        return ".mp3"
    return ".wav"


def _save_audio_upload_as_wav(upload: UploadFile, sample_rate: int = REFERENCE_SAMPLE_RATE) -> str:
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


def _get_output_sample_rate(model=None) -> int:
    if OUTPUT_SAMPLE_RATE:
        return int(OUTPUT_SAMPLE_RATE)
    if model is not None:
        sample_rate = getattr(getattr(model, "tts_model", None), "sample_rate", None)
        if sample_rate:
            return int(sample_rate)
    return 16000


def _apply_control_instruction(text: str, control_instruction: Optional[str]) -> str:
    instruction = (control_instruction or "").strip()
    if not instruction:
        return text

    if not (instruction.startswith("(") and instruction.endswith(")")):
        instruction = f"({instruction})"

    return f"{instruction}{text}"


def _waveform_to_response(
    wav_array: np.ndarray,
    sample_rate: int,
    output_format: Literal["mp3", "wav"],
    filename_stem: str,
) -> StreamingResponse:
    wav = np.asarray(wav_array, dtype=np.float32).squeeze()
    if wav.ndim != 1:
        raise ValueError("VoxCPM returned an invalid waveform shape.")

    if output_format == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, wav, sample_rate, format="WAV")
        buffer.seek(0)
        media_type = "audio/wav"
        filename = f"{filename_stem}.wav"
        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

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

    return StreamingResponse(
        mp3_buffer,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename={filename_stem}.mp3"},
    )


def _generate_audio(
    *,
    text: str,
    control_instruction: Optional[str] = None,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = False,
    denoise: bool = False,
) -> np.ndarray:
    model = load_model()
    text = _apply_control_instruction(text, control_instruction)

    generate_kwargs = {
        "text": text,
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
        "normalize": bool(normalize),
        "denoise": bool(denoise),
    }

    if prompt_wav_path and prompt_text:
        generate_kwargs["prompt_wav_path"] = prompt_wav_path
        generate_kwargs["prompt_text"] = prompt_text

    if denoise and getattr(model, "denoiser", None) is None:
        logger.warning("Denoise requested, but the model was loaded without a denoiser.")
        generate_kwargs["denoise"] = False

    wav = model.generate(**generate_kwargs)
    return np.asarray(wav, dtype=np.float32)


class TTSRequest(BaseModel):
    text: str
    control_instruction: Optional[str] = None
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    normalize: bool = False
    output_format: Literal["mp3", "wav"] = DEFAULT_OUTPUT_FORMAT if DEFAULT_OUTPUT_FORMAT in {"mp3", "wav"} else "mp3"


@app.on_event("startup")
async def startup_event():
    load_model()


@app.post(
    "/tts",
    response_class=StreamingResponse,
    summary="Generate speech from text",
    description="Generate speech directly from text. Use control_instruction for voice design.",
)
async def text_to_speech(request: TTSRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        wav = _generate_audio(
            text=request.text,
            control_instruction=request.control_instruction,
            cfg_value=request.cfg_value,
            inference_timesteps=request.inference_timesteps,
            normalize=request.normalize,
        )
        return _waveform_to_response(
            wav,
            _get_output_sample_rate(load_model()),
            request.output_format,
            "speech",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/tts/clone",
    response_class=StreamingResponse,
    summary="Clone a voice from reference audio",
    description="Upload reference audio and generate speech in that voice. Provide prompt_text for ultimate cloning.",
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

        wav = _generate_audio(
            text=text,
            control_instruction=control_instruction,
            prompt_wav_path=ref_path,
            prompt_text=prompt_text.strip(),
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
        )

        return _waveform_to_response(
            wav,
            _get_output_sample_rate(load_model()),
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


@app.get("/health", summary="Health check")
async def health():
    model = tts_model
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_loaded": model is not None,
        "output_sample_rate": _get_output_sample_rate(model),
    }
