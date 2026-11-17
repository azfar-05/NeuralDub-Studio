"""Visual quality telemetry for the original and synthesized speech tracks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    size = min(left.size, right.size)
    if size == 0:
        return 0.0
    left, right = left[:size], right[:size]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def generate_spectrogram_comparison(orig_wav: str, cloned_wav: str, output_png: str) -> dict[str, float]:
    """Save a high-contrast dual Mel plot and return normalized pitch/energy scores."""
    try:
        import librosa
        import librosa.display
    except ImportError as exc:
        raise RuntimeError("Install librosa to generate visual telemetry.") from exc

    original_path, clone_path = Path(orig_wav), Path(cloned_wav)
    if not original_path.is_file() or not clone_path.is_file():
        raise FileNotFoundError("Both original and cloned WAV files are required.")
    output = Path(output_png).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    original, sample_rate = librosa.load(str(original_path), sr=16000, mono=True)
    cloned, _ = librosa.load(str(clone_path), sr=16000, mono=True)
    original_mel = librosa.feature.melspectrogram(y=original, sr=sample_rate, n_mels=128)
    cloned_mel = librosa.feature.melspectrogram(y=cloned, sr=sample_rate, n_mels=128)
    original_db = librosa.power_to_db(original_mel, ref=np.max)
    cloned_db = librosa.power_to_db(cloned_mel, ref=np.max)

    original_pitch, _, _ = librosa.pyin(original, fmin=65, fmax=500, sr=sample_rate)
    cloned_pitch, _, _ = librosa.pyin(cloned, fmin=65, fmax=500, sr=sample_rate)
    original_pitch = np.nan_to_num(original_pitch, nan=0.0)
    cloned_pitch = np.nan_to_num(cloned_pitch, nan=0.0)
    original_energy = librosa.feature.rms(y=original)[0]
    cloned_energy = librosa.feature.rms(y=cloned)[0]
    metrics = {
        "pitch_similarity": round(max(0.0, _cosine_similarity(original_pitch, cloned_pitch)), 4),
        "energy_similarity": round(max(0.0, _cosine_similarity(original_energy, cloned_energy)), 4),
    }

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, facecolor="#10131a")
    for axis, data, title in zip(axes, (original_db, cloned_db), ("Original Speech", "Cloned Speech")):
        image = librosa.display.specshow(data, sr=sample_rate, x_axis="time", y_axis="mel", ax=axis,
                                         cmap="magma", vmin=-80, vmax=0)
        axis.set_title(title, color="white", fontweight="bold")
        axis.tick_params(colors="white")
        axis.set_facecolor("#10131a")
        for spine in axis.spines.values():
            spine.set_color("#64748b")
    figure.colorbar(image, ax=axes, format="%+2.0f dB", pad=0.02).ax.yaxis.set_tick_params(color="white")
    figure.suptitle(f"Mel-Spectrogram Comparison  |  Pitch {metrics['pitch_similarity']:.2%}  Energy {metrics['energy_similarity']:.2%}", color="white")
    figure.tight_layout()
    figure.savefig(output, dpi=160, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return metrics
