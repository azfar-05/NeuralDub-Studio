"""Audio extraction, transcription, translation, and voice-cloning stages.

All heavyweight dependencies are imported inside their stage and discarded in a
``finally`` block.  This is intentional: an 8 GB unified-memory Mac should
never retain ASR, translation, and voice models at the same time.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from backend.memory_utils import flush_memory, get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

SUPPORTED_LANGUAGES = {"eng_Latn", "hin_Deva", "kan_Knda", "spa_Latn", "fra_Latn", "deu_Latn", "deu_Deva", "jpn_Jpan"}
LANGUAGE_ALIASES = {"deu_Deva": "deu_Latn"}  # UI compatibility; NLLB's official German code is deu_Latn.
EDGE_VOICES = {
    "eng_Latn": "en-US-AriaNeural",
    "hin_Deva": "hi-IN-SwaraNeural",
    "kan_Knda": "kn-IN-SapnaNeural",
    "spa_Latn": "es-ES-ElviraNeural",
    "fra_Latn": "fr-FR-DeniseNeural",
    "deu_Latn": "de-DE-KatjaNeural",
    "jpn_Jpan": "ja-JP-NanamiNeural",
}
EDGE_VOICE_BY_PITCH = {
    "hin_Deva": ("hi-IN-MadhurNeural", "hi-IN-SwaraNeural"),
    "kan_Knda": ("kn-IN-GaganNeural", "kn-IN-SapnaNeural"),
    "eng_Latn": ("en-US-AndrewMultilingualNeural", "en-US-AvaMultilingualNeural"),
    "spa_Latn": ("es-ES-AlvaroNeural", "es-ES-ElviraNeural"),
}


def _require_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    return candidate


def _run_ffmpeg(arguments: list[str]) -> None:
    """Run ffmpeg and include its diagnostic text in errors."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or is not on PATH.")
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {detail}")


def extract_audio(video_path: str, output_wav: str) -> str:
    """Extract a 16 kHz mono PCM WAV track from a video using ffmpeg."""
    source = _require_file(video_path, "Video file")
    destination = Path(output_wav).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(destination),
    ])
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("ffmpeg completed but did not produce a WAV file.")
    return str(destination)


def _split_long_segments(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split multi-sentence Whisper blocks into natural clause/sentence beats."""
    refined: list[dict[str, Any]] = []
    for seg in raw_segments:
        text = seg["text"].strip()
        start, end = float(seg["start"]), float(seg["end"])
        if not text or end <= start:
            continue
        parts = [p.strip() for p in re.split(r"(?<=[.?!।])\s+", text) if p.strip()]
        if len(parts) <= 1 or (end - start) < 3.0:
            refined.append({"id": len(refined), "start": start, "end": end, "text": text})
            continue
        total_chars = sum(len(p) for p in parts)
        cursor = start
        for idx, part in enumerate(parts):
            share = (end - start) * (len(part) / max(1, total_chars))
            part_end = end if idx == len(parts) - 1 else min(end, cursor + share)
            # Leave a 0.14s natural breath pause between split sentences
            spoken_end = part_end - 0.14 if idx < len(parts) - 1 and (part_end - cursor) > 0.6 else part_end
            refined.append({"id": len(refined), "start": round(cursor, 3), "end": round(spoken_end, 3), "text": part})
            cursor = part_end
    return refined


def transcribe_audio(wav_path: str) -> dict[str, Any]:
    """Transcribe a WAV with Faster-Whisper and release the model immediately."""
    source = _require_file(wav_path, "Audio file")
    model: Any | None = None
    try:
        from faster_whisper import WhisperModel

        runtime_device = "cuda" if get_device() == "cuda" else "cpu"
        compute_type = "float16" if runtime_device == "cuda" else "int8"
        model = WhisperModel(
            "small", device=runtime_device, compute_type=compute_type,
            download_root=str(MODELS_DIR / "faster-whisper"),
        )
        segments, info = model.transcribe(
            str(source),
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 220, "speech_pad_ms": 80},
            condition_on_previous_text=False,
        )
        raw_segments = [
            {"id": index, "start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()}
            for index, segment in enumerate(segments)
        ]
        segment_data = _split_long_segments(raw_segments)
        return {
            "text": " ".join(item["text"] for item in segment_data).strip(),
            "language": str(info.language),
            "segments": segment_data,
        }
    except ImportError as exc:
        raise RuntimeError("Install faster-whisper to use transcription.") from exc
    finally:
        if model is not None:
            del model
        flush_memory()


def _translate_batch(texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
    """Translate all segments in a single NLLB-200 model load."""
    if not texts:
        return []
    if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language. Use one of: {sorted(SUPPORTED_LANGUAGES)}")
    src_lang, tgt_lang = LANGUAGE_ALIASES.get(src_lang, src_lang), LANGUAGE_ALIASES.get(tgt_lang, tgt_lang)
    if src_lang == tgt_lang:
        return [t.strip() for t in texts]

    model: Any | None = None
    tokenizer: Any | None = None
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        checkpoint = "facebook/nllb-200-distilled-600M"
        cache_dir = MODELS_DIR / "nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=str(cache_dir), src_lang=src_lang)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, cache_dir=str(cache_dir))
        device = get_device()
        try:
            model.to(device)
        except RuntimeError:
            if device != "mps":
                raise
            device = "cpu"
            model.to(device)

        results: list[str] = []
        bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.inference_mode():
            for text in texts:
                if not text.strip():
                    results.append("")
                    continue
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                inputs = {name: value.to(device) for name, value in inputs.items()}
                try:
                    tokens = model.generate(
                        **inputs, forced_bos_token_id=bos_id,
                        max_new_tokens=512, num_beams=2, do_sample=False,
                    )
                except RuntimeError:
                    if device != "mps":
                        raise
                    device = "cpu"
                    model.to(device)
                    inputs = {name: value.to(device) for name, value in inputs.items()}
                    tokens = model.generate(
                        **inputs, forced_bos_token_id=bos_id,
                        max_new_tokens=512, num_beams=2, do_sample=False,
                    )
                results.append(tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip())
        return results
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers to use translation.") from exc
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        flush_memory()


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate a single string with NLLB-200."""
    if not text.strip():
        return ""
    return _translate_batch([text], src_lang, tgt_lang)[0]


def _ensure_openvoice_checkpoints() -> Path:
    """Download the official OpenVoice V2 checkpoint snapshot on first use."""
    destination = MODELS_DIR / "OpenVoiceV2"
    if (destination / "converter").is_dir():
        return destination
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("myshell-ai/OpenVoiceV2", local_dir=str(destination))
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to download OpenVoice V2 checkpoints.") from exc
    return destination


def _reference_voice_profile(reference_wav: Path | None) -> tuple[str | None, str]:
    """Estimate voice pitch and activity density for native Edge voice/rate choice."""
    if reference_wav is None:
        return None, "+0%"
    try:
        import librosa
        import numpy as np

        audio, sample_rate = librosa.load(str(reference_wav), sr=22050, mono=True)
        pitches = librosa.yin(audio, fmin=65, fmax=400, sr=sample_rate)
        voiced = pitches[np.isfinite(pitches) & (pitches > 65)]
        median_pitch = float(np.median(voiced)) if voiced.size else 170.0
        _, intervals = librosa.effects.split(audio, top_db=25)
        active = sum(end - start for start, end in intervals) / max(1, len(audio))
        rate = "+4%" if active > .85 else "-4%" if active < .55 else "+0%"
        return "high" if median_pitch >= 165 else "low", rate
    except Exception:
        return None, "+0%"


def _edge_tts_to_wav(
    text: str,
    lang: str,
    destination: Path,
    pitch_band: str | None = None,
    rate: str = "+0%",
) -> None:
    """Create a temporary 24 kHz WAV using Edge Neural TTS."""
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Install edge-tts for this target language.") from exc

    mp3 = destination.with_suffix(".edge.mp3")
    voices = EDGE_VOICE_BY_PITCH.get(lang)
    voice = voices[1 if pitch_band == "high" else 0] if voices and pitch_band else EDGE_VOICES[lang]

    # Ensure terminal punctuation so neural TTS uses natural falling/rising intonation
    spoken = text.strip()
    if spoken and spoken[-1] not in ".?!।,;:":
        spoken += "."

    async def synthesize() -> None:
        await edge_tts.Communicate(spoken, voice, rate=rate, pitch="-1Hz").save(str(mp3))

    asyncio.run(synthesize())
    try:
        _run_ffmpeg(["ffmpeg", "-y", "-i", str(mp3), "-ac", "1", "-ar", "24000", str(destination)])
    finally:
        mp3.unlink(missing_ok=True)


def _configure_openvoice_whisper(se_extractor: Any) -> Any:
    """Initialize OpenVoice's hidden Faster-Whisper singleton safely on M1."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install faster-whisper for OpenVoice reference extraction.") from exc

    runtime_device = "cuda" if get_device() == "cuda" else "cpu"
    compute_type = "float16" if runtime_device == "cuda" else "int8"
    previous = getattr(se_extractor, "model", None)
    if previous is not None:
        se_extractor.model = None
        del previous
        flush_memory()
    model = WhisperModel(
        se_extractor.model_size, device=runtime_device, compute_type=compute_type,
        download_root=str(MODELS_DIR / "faster-whisper"),
    )
    se_extractor.model = model
    return model


def _clean_reference_for_embedding(reference_wav: Path, output_wav: Path) -> tuple[Path, float]:
    """Create up to 30 seconds of voiced, RMS-normalized reference speech."""
    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Install librosa and soundfile for OpenVoice reference cleanup.") from exc

    audio, sample_rate = librosa.load(str(reference_wav), sr=22050, mono=True)
    total_seconds = len(audio) / sample_rate
    intervals = librosa.effects.split(audio, top_db=28)
    if not len(intervals):
        raise RuntimeError("Reference audio contains no usable speech after silence trimming.")
    maximum_samples, gap_samples = 30 * sample_rate, int(.15 * sample_rate)
    pieces: list[np.ndarray] = []
    used = 0
    for start, end in intervals:
        if used >= maximum_samples:
            break
        piece = audio[start:min(end, start + maximum_samples - used)]
        if piece.size:
            pieces.append(piece)
            used += len(piece)
        if pieces and used < maximum_samples:
            gap = min(gap_samples, maximum_samples - used)
            pieces.append(np.zeros(gap, dtype=audio.dtype))
            used += gap
    trimmed = np.concatenate(pieces) if pieces else np.empty(0, dtype=audio.dtype)
    if not trimmed.size:
        raise RuntimeError("Reference audio contains no voiced samples after silence trimming.")
    rms = float(np.sqrt(np.mean(np.square(trimmed))))
    if rms > 1e-6:
        target_rms = 10 ** (-20 / 20)
        gain = min(target_rms / rms, .99 / max(float(np.max(np.abs(trimmed))), 1e-6))
        trimmed = trimmed * gain
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), trimmed, sample_rate, subtype="PCM_16")
    return output_wav, total_seconds


def _audio_duration(path: str | Path) -> float:
    """Return exact WAV duration without loading it into a model."""
    import soundfile as sf
    info = sf.info(str(path))
    return float(info.frames / info.samplerate)


def _media_duration(path: str | Path) -> float:
    """Read container duration so the mastered track matches the video exactly."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except ValueError:
        return 0.0


def _fit_segment_to_slot(audio: Any, sample_rate: int, slot_seconds: float) -> Any:
    """Fit synthesized speech into its slot with smooth fade-out and no mid-word cutoffs."""
    import librosa
    import numpy as np

    if slot_seconds <= 0 or not len(audio):
        return np.empty(0, dtype=audio.dtype)

    # Trim leading/trailing TTS padding silence so speech starts right on cue
    trimmed, _ = librosa.effects.trim(audio, top_db=32)
    if len(trimmed) > int(0.15 * sample_rate):
        audio = trimmed

    duration = len(audio) / sample_rate
    if duration > slot_seconds:
        # Allow up to 1.32x natural time-compression so full sentences fit without clipping
        rate = min(1.32, max(1.0, duration / slot_seconds))
        audio = librosa.effects.time_stretch(audio, rate=rate)

    target_samples = round(slot_seconds * sample_rate)
    if len(audio) > target_samples:
        audio = audio[:target_samples].copy()

    # Apply a 25 ms cosine fade-in/out to prevent clicks at segment boundaries
    fade_len = min(int(0.025 * sample_rate), len(audio) // 4)
    if fade_len > 1:
        fade = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_len, dtype=audio.dtype)))
        audio[:fade_len] *= fade
        audio[-fade_len:] *= fade[::-1]
    return audio


def _studio_finish(audio: Any, sample_rate: int) -> Any:
    """Apply warm studio de-essing/EQ plus RMS/peak broadcast normalization."""
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    if not len(audio):
        return audio
    # Gentle low-pass at 8.5 kHz removes metallic neural-vocoder buzz while keeping crisp speech
    cutoff = min(8500.0, sample_rate * .45)
    filtered = sosfiltfilt(butter(4, cutoff, btype="low", fs=sample_rate, output="sos"), audio)
    rms = float(np.sqrt(np.mean(np.square(filtered))))
    if rms > 1e-7:
        filtered = filtered * ((10 ** (-18 / 20)) / rms)
    peak = float(np.max(np.abs(filtered)))
    ceiling = 10 ** (-1 / 20)
    if peak > ceiling:
        filtered = filtered * (ceiling / peak)
    return filtered.astype("float32")


def _try_indic_f5(text: str, reference_wav: Path, destination: Path) -> bool:
    """Optionally run IndicF5 when explicitly enabled and installed."""
    if os.getenv("NEURALDUB_USE_INDIC_F5", "0") != "1":
        return False
    engine: Any | None = None
    try:
        from f5_tts.api import F5TTS

        engine = F5TTS(model="ai4bharat/IndicF5")
        engine.infer(ref_file=str(reference_wav), ref_text="", gen_text=text, file_wave=str(destination))
        return destination.is_file() and destination.stat().st_size > 0
    except Exception:
        destination.unlink(missing_ok=True)
        return False
    finally:
        if engine is not None:
            del engine
        flush_memory()


def _clone_segments_batch(
    segments: list[dict[str, Any]],
    reference_wav: Path,
    tgt_lang: str,
    work_dir: Path,
) -> list[Path]:
    """Synthesize and voice-clone all segments in a single OpenVoice model session."""
    tgt_lang = LANGUAGE_ALIASES.get(tgt_lang, tgt_lang)
    clean_reference = work_dir / "clean_reference.wav"
    target_embedding_path = work_dir / "target_se.pth"
    converter: Any | None = None
    se_extractor_module: Any | None = None
    reference_whisper: Any | None = None
    outputs: list[Path] = []

    try:
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter

        se_extractor_module = se_extractor
        checkpoint_root = _ensure_openvoice_checkpoints()
        device = get_device()
        converter_dir = checkpoint_root / "converter"
        converter = ToneColorConverter(str(converter_dir / "config.json"), device=device)
        converter.load_ckpt(str(converter_dir / "checkpoint.pth"))

        cleaned_reference, reference_seconds = _clean_reference_for_embedding(reference_wav, clean_reference)
        pitch_band, rate = _reference_voice_profile(cleaned_reference)

        reference_whisper = _configure_openvoice_whisper(se_extractor)
        if reference_seconds <= 45:
            target_se = converter.extract_se([str(cleaned_reference)], se_save_path=str(target_embedding_path))
        else:
            target_se, _ = se_extractor.get_se(str(reference_wav), converter, vad=False)

        for idx, segment in enumerate(segments):
            text = segment["translation"].strip()
            if not text:
                continue
            temp_src = work_dir / f"src_{idx:03d}.wav"
            out_clip = work_dir / f"segment_{idx:03d}.wav"
            try:
                used_indic_f5 = tgt_lang in {"hin_Deva", "kan_Knda"} and _try_indic_f5(
                    text, cleaned_reference, temp_src
                )
                if not used_indic_f5:
                    # Use neural Edge-TTS for every language (much warmer prosody than legacy VITS BaseSpeakerTTS)
                    _edge_tts_to_wav(text, tgt_lang, temp_src, pitch_band=pitch_band, rate=rate)

                source_embedding = converter.extract_se([str(temp_src)])
                # tau=0.70 transfers stronger target speaker warmth & eliminates robotic formant artifacts
                converter.convert(
                    audio_src_path=str(temp_src),
                    src_se=source_embedding,
                    tgt_se=target_se,
                    output_path=str(out_clip),
                    message="@MyShell",
                    tau=0.70,
                )
                if out_clip.is_file() and out_clip.stat().st_size > 0:
                    outputs.append(out_clip)
            finally:
                temp_src.unlink(missing_ok=True)
        return outputs
    except ImportError as exc:
        raise RuntimeError("Install OpenVoice V2 and its dependencies to clone voices.") from exc
    finally:
        clean_reference.unlink(missing_ok=True)
        target_embedding_path.unlink(missing_ok=True)
        if converter is not None:
            del converter
        if se_extractor_module is not None:
            se_extractor_module.model = None
        if reference_whisper is not None:
            del reference_whisper
        flush_memory()


def clone_and_synthesize(translated_text: str, reference_wav: str, tgt_lang: str, output_wav: str) -> str:
    """Single-clip helper preserved for compatibility."""
    if not translated_text.strip():
        raise ValueError("Cannot synthesize empty translated text.")
    if tgt_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported target language: {tgt_lang}")
    reference = _require_file(reference_wav, "Reference WAV")
    destination = Path(output_wav).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    clips = _clone_segments_batch([{"translation": translated_text}], reference, tgt_lang, destination.parent)
    if not clips:
        raise RuntimeError("OpenVoice completed without creating synthesized audio.")
    shutil.move(str(clips[0]), str(destination))
    return str(destination)


def run_audio_pipeline(
    video_path: str,
    target_language: str,
    output_wav: str,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Create a timestamp-aligned dubbed track with natural sentence pauses."""
    if target_language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported target language: {target_language}")
    destination = Path(output_wav).expanduser().resolve()
    original_wav = destination.with_name("original_audio.wav")
    work_dir = destination.with_name(f"{destination.stem}_segments")
    work_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("extracting_audio")
    extract_audio(video_path, str(original_wav))

    if progress_callback:
        progress_callback("transcribing")
    transcript = transcribe_audio(str(original_wav))
    source_map = {
        "en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda",
        "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "ja": "jpn_Jpan",
    }
    source_language = source_map.get(transcript["language"], "eng_Latn")
    segments = [
        segment for segment in transcript["segments"]
        if segment.get("text", "").strip() and float(segment["end"]) > float(segment["start"])
    ]
    if not segments:
        raise RuntimeError("Whisper produced no speech segments to dub.")

    if progress_callback:
        progress_callback("translating")
    translations = _translate_batch([seg["text"] for seg in segments], source_language, target_language)
    translated_segments = [
        {**segment, "translation": translation}
        for segment, translation in zip(segments, translations)
    ]

    if progress_callback:
        progress_callback("cloning_voice")
    import numpy as np
    import soundfile as sf

    clip_paths = _clone_segments_batch(translated_segments, original_wav, target_language, work_dir)
    if not clip_paths:
        raise RuntimeError("No cloned segment audio was generated.")

    master: Any | None = None
    master_rate = 0
    master_duration = max(_audio_duration(original_wav), _media_duration(video_path))
    prev_end_sample = 0

    for segment, clip in zip(translated_segments, clip_paths):
        audio, sample_rate = sf.read(str(clip), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        if master is None:
            master_rate = sample_rate
            master = np.zeros(round(master_duration * master_rate), dtype="float32")
        if sample_rate != master_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=master_rate)

        slot_seconds = float(segment["end"]) - float(segment["start"])
        audio = _fit_segment_to_slot(audio, master_rate, slot_seconds)

        # Enforce at least a 0.12s breath gap after the previous sentence so segments never blend continuously
        min_start = prev_end_sample + int(0.12 * master_rate) if prev_end_sample > 0 else 0
        start = max(min_start, round(float(segment["start"]) * master_rate))
        end = min(len(master), start + len(audio))
        if end > start:
            master[start:end] += audio[:end - start]
            prev_end_sample = end

    if master is None:
        raise RuntimeError("No cloned segment audio was generated.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), _studio_finish(master, master_rate), master_rate, subtype="PCM_16")
    return {
        "original_wav": str(original_wav),
        "cloned_wav": str(destination),
        "transcript": transcript,
        "translation": " ".join(segment["translation"] for segment in translated_segments),
        "translated_segments": translated_segments,
    }