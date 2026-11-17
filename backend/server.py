"""FastAPI service contract consumed by the Antigravity visual frontend."""

from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

import psutil
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.audio_pipeline import (SUPPORTED_LANGUAGES, clone_and_synthesize, extract_audio,
                                    transcribe_audio, translate_text)
from backend.lipsync_engine import run_lipsync
from backend.memory_utils import get_device
from backend.visual_telemetry import generate_spectrogram_comparison

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="NeuralDub Studio API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

Job = dict[str, Any]
_jobs: dict[str, Job] = {}
_job_lock = threading.Lock()


def _ram_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / 1024**2, 1)


def _update(job_id: str, stage: str, percent: int, **values: Any) -> None:
    with _job_lock:
        _jobs[job_id].update(stage=stage, percent=percent, ram_mb=_ram_mb(), **values)


def _url(path: Path) -> str:
    return "/outputs/" + path.resolve().relative_to(OUTPUTS_DIR.resolve()).as_posix()


def _source_language(whisper_language: str) -> str:
    return {"en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "ja": "jpn_Jpan"}.get(whisper_language, "eng_Latn")


def _process_dub(job_id: str, video_path: Path, target_language: str, use_finetuned_weights: bool) -> None:
    """Run one job sequentially; every model stage frees memory before the next."""
    job_dir = OUTPUTS_DIR / job_id
    original = job_dir / "original_audio.wav"
    cloned = job_dir / "cloned_speech.wav"
    spectrogram = job_dir / "spectrogram.png"
    dubbed = job_dir / "dubbed_video.mp4"
    try:
        _update(job_id, "extracting_audio", 5)
        extract_audio(str(video_path), str(original))
        _update(job_id, "transcribing", 20)
        transcript = transcribe_audio(str(original))
        _update(job_id, "translating", 42)
        translation = translate_text(transcript["text"], _source_language(transcript["language"]), target_language)
        _update(job_id, "cloning_voice", 60)
        clone_and_synthesize(translation, str(original), target_language, str(cloned))
        _update(job_id, "generating_telemetry", 70)
        voice_metrics = generate_spectrogram_comparison(str(original), str(cloned), str(spectrogram))
        _update(job_id, "lip_syncing", 78)
        fine_tuned = MODELS_DIR / "unet_finetuned.pth"
        result = run_lipsync(str(video_path), str(cloned), str(dubbed),
                             str(fine_tuned) if use_finetuned_weights and fine_tuned.is_file() else None)
        _update(job_id, "complete", 100, done=True, result={
            "dubbed_video_url": _url(Path(result["dubbed_video"])),
            "mouth_zoom_url": _url(Path(result["mouth_zoom"])),
            "spectrogram_url": _url(spectrogram),
            "transcript": transcript,
            "translation": translation,
            "voice_metrics": voice_metrics,
            "sync_metrics": result["sync_metrics"],
        })
    except Exception as exc:
        _update(job_id, "failed", 100, done=True, error=str(exc))
        traceback.print_exc()


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Return runtime hardware and fine-tuned checkpoint availability."""
    memory = psutil.virtual_memory()
    fine_tuned = (MODELS_DIR / "unet_finetuned.pth").is_file()
    return {"device": get_device(), "available_ram_mb": round(memory.available / 1024**2, 1),
            # Models are always unloaded between jobs by design; availability is
            # the useful health signal for a lazily-loaded fine-tuned UNet.
            "finetuned_weights_available": fine_tuned, "finetuned_weights_loaded": False}


@app.post("/api/dub")
async def dub(background_tasks: BackgroundTasks, video_file: UploadFile = File(...),
              target_language: str = Form(...), use_finetuned_weights: bool = Form(True)) -> dict[str, str]:
    """Queue a complete four-stage dubbing job and immediately return its ID."""
    if target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=422, detail=f"Unsupported target language: {target_language}")
    suffix = Path(video_file.filename or "input.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        raise HTTPException(status_code=415, detail="Upload a supported video file.")
    job_id = uuid.uuid4().hex
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    upload_path = job_dir / f"input{suffix}"
    with upload_path.open("wb") as file_handle:
        shutil.copyfileobj(video_file.file, file_handle)
    await video_file.close()
    with _job_lock:
        _jobs[job_id] = {"stage": "queued", "percent": 0, "ram_mb": _ram_mb(), "done": False, "error": None}
    background_tasks.add_task(_process_dub, job_id, upload_path, target_language, use_finetuned_weights)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str) -> Job:
    """Return an immutable snapshot of the requested job's live progress."""
    with _job_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job ID")
        return dict(job)


@app.get("/api/training-metrics")
def training_metrics() -> dict[str, Any]:
    """Serve fine-tuning data, providing chart-safe placeholder values pre-training."""
    metrics_path = MODELS_DIR / "training_metrics.json"
    if metrics_path.is_file():
        try:
            with metrics_path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail=f"Invalid training metrics: {exc}") from exc
    return {
        "status": "not_trained", "epochs": [1, 2, 3, 4, 5],
        "losses": {"L1": [0.48, 0.39, 0.32, 0.28, 0.25], "Perceptual_VGG": [0.71, 0.60, 0.51, 0.45, 0.40],
                   "GAN": [1.20, 1.08, 0.98, 0.91, 0.87], "SyncNet": [0.83, 0.70, 0.61, 0.55, 0.50]},
        "benchmarks": {"before": {"LSE-D": 8.1, "LSE-C": 4.0}, "after": {"LSE-D": 7.2, "LSE-C": 4.7}},
    }


# Mount last so explicit /api/* and /outputs/* routes keep their precedence.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
