"""MuseTalk 1.5 subprocess wrapper with bounded-memory visual telemetry.

MuseTalk is intentionally executed in its own Python process.  That process
owns the VAE, Whisper audio encoder, DWPose and UNet, then exits immediately;
this is the most reliable way to return MPS unified memory on an 8 GB Mac.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from backend.audio_pipeline import _run_ffmpeg
from backend.memory_utils import flush_memory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "musetalk"
REPO_DIR = Path(__file__).resolve().parent / "musetalk_repo"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_musetalk_weights() -> Path:
    """Fetch the MuseTalk source and official V1.5 checkpoints when absent."""
    if not REPO_DIR.is_dir():
        if shutil.which("git") is None:
            raise RuntimeError("git is required to download MuseTalk on first use.")
        result = subprocess.run(["git", "clone", "--depth", "1", "https://github.com/TMElyralab/MuseTalk.git", str(REPO_DIR)],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"Could not clone MuseTalk: {result.stderr.strip()}")
    required = MODELS_DIR / "musetalkV15" / "unet.pth"
    if not required.is_file():
        try:
            from huggingface_hub import snapshot_download

            # The official model repository contains musetalkV15, sd-vae,
            # whisper and dwpose folders in the layout expected by MuseTalk.
            snapshot_download("TMElyralab/MuseTalk", local_dir=str(MODELS_DIR))
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub to download MuseTalk weights.") from exc
    if not required.is_file():
        raise RuntimeError("MuseTalk download completed but musetalkV15/unet.pth is missing.")
    return MODELS_DIR


def _newest_video(directory: Path, excluded: Path) -> Path:
    videos = [path for path in directory.rglob("*.mp4") if path.resolve() != excluded.resolve()]
    if not videos:
        raise RuntimeError("MuseTalk completed without creating an MP4 result.")
    return max(videos, key=lambda path: path.stat().st_mtime)


def _create_comparison(original_video: Path, synced_video: Path, output: Path) -> list[float]:
    """Create 256px mouth comparison video and a visual/audio alignment timeline."""
    try:
        import cv2
        import numpy as np
        import librosa
    except ImportError as exc:
        raise RuntimeError("Install opencv-python, numpy and librosa for lip telemetry.") from exc

    original = cv2.VideoCapture(str(original_video))
    synced = cv2.VideoCapture(str(synced_video))
    fps = original.get(cv2.CAP_PROP_FPS) or 25.0
    count = int(min(original.get(cv2.CAP_PROP_FRAME_COUNT), synced.get(cv2.CAP_PROP_FRAME_COUNT)))
    if count <= 0:
        raise RuntimeError("Could not read frames for mouth comparison.")
    temporary = output.with_suffix(".silent.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (512, 256))
    movement: list[float] = []
    previous: Any | None = None
    for _ in range(count):
        ok_a, frame_a = original.read()
        ok_b, frame_b = synced.read()
        if not (ok_a and ok_b):
            break
        def mouth(frame: Any) -> Any:
            height, width = frame.shape[:2]
            # Lower-central crop is stable and avoids retaining a face detector.
            crop = frame[int(height * .48):int(height * .92), int(width * .20):int(width * .80)]
            return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
        left, right = mouth(frame_a), mouth(frame_b)
        gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        movement.append(0.0 if previous is None else float(np.mean(cv2.absdiff(gray, previous))))
        previous = gray
        writer.write(np.hstack((left, right)))
    original.release(); synced.release(); writer.release()

    # Compare mouth motion with audio energy at matching frame timestamps.
    audio, sample_rate = librosa.load(str(synced_video), sr=16000, mono=True)
    hop = max(1, round(sample_rate / fps))
    energy = librosa.feature.rms(y=audio, frame_length=hop * 2, hop_length=hop)[0]
    timeline: list[float] = []
    for index, value in enumerate(movement):
        start, end = max(0, index - 3), min(len(movement), index + 4)
        visual = np.asarray(movement[start:end], dtype=float)
        audible = np.asarray(energy[start:min(end, len(energy))], dtype=float)
        length = min(visual.size, audible.size)
        if length < 2 or np.std(visual[:length]) == 0 or np.std(audible[:length]) == 0:
            timeline.append(0.0)
        else:
            timeline.append(round(float((np.corrcoef(visual[:length], audible[:length])[0, 1] + 1) / 2), 4))
    _run_ffmpeg(["ffmpeg", "-y", "-i", str(temporary), "-i", str(synced_video), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(output)])
    temporary.unlink(missing_ok=True)
    return timeline


def run_lipsync(video_path: str, audio_path: str, output_video_path: str,
                custom_unet_path: str | None = None) -> dict[str, Any]:
    """Run MuseTalk V1.5 at 256px and return output paths plus sync confidence."""
    video, audio = Path(video_path).resolve(), Path(audio_path).resolve()
    output = Path(output_video_path).resolve()
    if not video.is_file() or not audio.is_file():
        raise FileNotFoundError("Both input video and synthesized audio must exist.")
    if custom_unet_path and not Path(custom_unet_path).is_file():
        custom_unet_path = None
    output.parent.mkdir(parents=True, exist_ok=True); OUTPUTS_DIR.mkdir(exist_ok=True)
    config_file: Path | None = None
    try:
        weights = ensure_musetalk_weights()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            config_file = Path(handle.name)
            # MuseTalk's inference runner consumes a mapping of named tasks.
            json.dump({"task_0": {"video_path": str(video), "audio_path": str(audio), "bbox_shift": 0,
                                   "result_dir": str(output.parent),
                                   "unet_model_path": custom_unet_path or str(weights / "musetalkV15" / "unet.pth"),
                                   "resolution": 256}}, handle)
        command = [sys.executable, "-m", "scripts.inference", "--inference_config", str(config_file)]
        process = subprocess.run(command, cwd=REPO_DIR, capture_output=True, text=True, check=False)
        if process.returncode:
            raise RuntimeError(f"MuseTalk inference failed: {(process.stderr or process.stdout).strip()}")
        rendered = _newest_video(REPO_DIR / "results", output)
        if rendered.resolve() != output.resolve():
            shutil.move(str(rendered), str(output))
        # Keep job artifacts together so simultaneous API jobs cannot overwrite
        # each other's comparison clip.
        mouth_zoom = output.parent / "mouth_zoom.mp4"
        scores = _create_comparison(video, output, mouth_zoom)
        return {"dubbed_video": str(output), "mouth_zoom": str(mouth_zoom), "sync_metrics": scores}
    finally:
        if config_file is not None:
            config_file.unlink(missing_ok=True)
        # Model objects lived only in the completed child process; still clear
        # parent-side accelerator allocator before the server starts another job.
        flush_memory()
