"""Evaluate visual fidelity and optionally official SyncNet LSE scores for video pairs.

SSIM/PSNR compare aligned frames from an original and dubbed render.  LSE-D and
LSE-C are deliberately delegated to an official SyncNet evaluation script: the
metrics are undefined without its pretrained audiovisual embedding network.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _open_video(path: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return capture


def frame_fidelity(original_path: str, dubbed_path: str, max_frames: int | None = None) -> dict[str, Any]:
    """Return mean SSIM/PSNR over equally indexed, resolution-matched frames."""
    original, dubbed = _open_video(original_path), _open_video(dubbed_path)
    ssim_values: list[float] = []
    psnr_values: list[float] = []
    try:
        while max_frames is None or len(ssim_values) < max_frames:
            ok_original, original_frame = original.read()
            ok_dubbed, dubbed_frame = dubbed.read()
            if not (ok_original and ok_dubbed):
                break
            dubbed_frame = cv2.resize(dubbed_frame, (original_frame.shape[1], original_frame.shape[0]), interpolation=cv2.INTER_AREA)
            original_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            dubbed_gray = cv2.cvtColor(dubbed_frame, cv2.COLOR_BGR2GRAY)
            ssim_values.append(float(structural_similarity(original_gray, dubbed_gray, data_range=255)))
            psnr_values.append(float(peak_signal_noise_ratio(original_frame, dubbed_frame, data_range=255)))
    finally:
        original.release()
        dubbed.release()
    if not ssim_values:
        raise RuntimeError("No mutually readable video frames were found.")
    return {"SSIM": round(float(np.mean(ssim_values)), 6), "PSNR": round(float(np.mean(psnr_values)), 4), "frames": len(ssim_values)}


def syncnet_lse(video_path: str, evaluator: str | None) -> dict[str, float | None]:
    """Run an official SyncNet evaluator and parse its reported LSE-D/LSE-C."""
    if not evaluator:
        return {"LSE-D": None, "LSE-C": None}
    script = Path(evaluator)
    if not script.is_file():
        raise FileNotFoundError(f"SyncNet evaluator does not exist: {script}")
    result = subprocess.run(["python", str(script), "--videofile", video_path], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"SyncNet evaluator failed: {result.stderr.strip() or result.stdout.strip()}")
    output = result.stdout + "\n" + result.stderr
    distance = re.search(r"LSE[- ]?D\s*[:=]\s*([0-9.]+)", output, re.I)
    confidence = re.search(r"LSE[- ]?C\s*[:=]\s*([0-9.]+)", output, re.I)
    if not distance or not confidence:
        raise RuntimeError("SyncNet completed but did not print parseable LSE-D and LSE-C values.")
    return {"LSE-D": float(distance.group(1)), "LSE-C": float(confidence.group(1))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired SSIM/PSNR and optional official SyncNet LSE.")
    parser.add_argument("--original", required=True, help="Ground-truth/original video")
    parser.add_argument("--dubbed", required=True, help="Generated dubbed video")
    parser.add_argument("--syncnet-evaluator", help="Path to official SyncNet evaluation script")
    parser.add_argument("--max-frames", type=int, help="Optional cap for quick checks")
    parser.add_argument("--output", help="Write JSON report to this path")
    args = parser.parse_args()
    report: dict[str, Any] = frame_fidelity(args.original, args.dubbed, args.max_frames)
    report.update(syncnet_lse(args.dubbed, args.syncnet_evaluator))
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
