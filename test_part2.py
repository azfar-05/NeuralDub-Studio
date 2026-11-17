"""End-to-end smoke test for NeuralDub's four-stage Part 2 pipeline."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import psutil

from backend.audio_pipeline import clone_and_synthesize, extract_audio, transcribe_audio, translate_text
from backend.lipsync_engine import run_lipsync
from backend.visual_telemetry import generate_spectrogram_comparison

ROOT = Path(__file__).resolve().parent


def measure(label: str, action):
    start = time.perf_counter(); result = action()
    ram = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"{label}: {time.perf_counter() - start:.2f}s | RAM {ram:.1f} MB")
    return result, ram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--lang", required=True, choices=("eng_Latn", "hin_Deva", "kan_Knda", "spa_Latn"))
    args = parser.parse_args()
    output = ROOT / "outputs" / "part2_smoke"; output.mkdir(parents=True, exist_ok=True)
    peak = 0.0
    original, ram = measure("1. extract", lambda: extract_audio(args.video, str(output / "original.wav"))); peak = max(peak, ram)
    transcript, ram = measure("1. transcribe", lambda: transcribe_audio(original)); peak = max(peak, ram)
    lang_map = {"en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda", "es": "spa_Latn"}
    translation, ram = measure("2. translate", lambda: translate_text(transcript["text"], lang_map.get(transcript["language"], "eng_Latn"), args.lang)); peak = max(peak, ram)
    cloned, ram = measure("3. clone", lambda: clone_and_synthesize(translation, original, args.lang, str(output / "cloned.wav"))); peak = max(peak, ram)
    _, ram = measure("3. spectrogram", lambda: generate_spectrogram_comparison(original, cloned, str(output / "spectrogram.png"))); peak = max(peak, ram)
    _, ram = measure("4. lipsync", lambda: run_lipsync(args.video, cloned, str(output / "dubbed.mp4"))); peak = max(peak, ram)
    print(f"Peak process RAM: {peak:.1f} MB")


if __name__ == "__main__":
    main()
