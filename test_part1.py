"""Command-line smoke test for the Part 1 audio dubbing pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import psutil

from backend.audio_pipeline import clone_and_synthesize, extract_audio, transcribe_audio, translate_text
from backend.visual_telemetry import generate_spectrogram_comparison

PROJECT_ROOT = Path(__file__).resolve().parent


def report_ram(step: str) -> None:
    """Print the resident memory used by this CLI process."""
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"[{step}] RAM: {memory_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeuralDub Part 1 audio pipeline.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--lang", required=True, choices=("eng_Latn", "hin_Deva", "kan_Knda", "spa_Latn"))
    args = parser.parse_args()

    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    original_wav = extract_audio(args.video, str(outputs / "original_audio.wav"))
    report_ram("audio extraction")
    transcript = transcribe_audio(original_wav)
    report_ram("transcription")
    print(f"Detected {transcript['language']}: {transcript['text']}")
    whisper_languages = {
        "en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda", "es": "spa_Latn",
    }
    source_lang = whisper_languages.get(transcript["language"], "eng_Latn")
    translated = translate_text(transcript["text"], source_lang, args.lang)
    report_ram("translation")
    print(f"Translation: {translated}")
    cloned_wav = clone_and_synthesize(translated, original_wav, args.lang, str(outputs / "cloned_speech.wav"))
    report_ram("voice synthesis")
    metrics = generate_spectrogram_comparison(original_wav, cloned_wav, str(outputs / "spectrogram.png"))
    report_ram("spectrogram")
    print(f"Created {cloned_wav} and {outputs / 'spectrogram.png'}")
    print(f"Similarity metrics: {metrics}")


if __name__ == "__main__":
    main()
