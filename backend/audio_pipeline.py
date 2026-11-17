"""Audio extraction, transcription, translation, and voice-cloning stages.

All heavyweight dependencies are imported inside their stage and discarded in a
``finally`` block.  This is intentional: an 8 GB unified-memory Mac should
never retain ASR, translation, and voice models at the same time.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

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


def transcribe_audio(wav_path: str) -> dict[str, Any]:
    """Transcribe a WAV with Faster-Whisper and release the model immediately."""
    source = _require_file(wav_path, "Audio file")
    model: Any | None = None
    try:
        from faster_whisper import WhisperModel

        # CTranslate2 has no MPS backend; CPU int8 avoids duplicating memory on macOS.
        runtime_device = "cuda" if get_device() == "cuda" else "cpu"
        compute_type = "float16" if runtime_device == "cuda" else "int8"
        model = WhisperModel("small", device=runtime_device, compute_type=compute_type,
                             download_root=str(MODELS_DIR / "faster-whisper"))
        segments, info = model.transcribe(str(source), beam_size=5)
        segment_data = [
            {"id": index, "start": float(segment.start), "end": float(segment.end),
             "text": segment.text.strip()}
            for index, segment in enumerate(segments)
        ]
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


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text with NLLB-200, caching its checkpoint under ``models/``."""
    if not text.strip():
        return ""
    if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language. Use one of: {sorted(SUPPORTED_LANGUAGES)}")
    src_lang, tgt_lang = LANGUAGE_ALIASES.get(src_lang, src_lang), LANGUAGE_ALIASES.get(tgt_lang, tgt_lang)

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
        model.to(device)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_new_tokens=512,
            )
        return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers to use translation.") from exc
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        flush_memory()


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


def _edge_tts_to_wav(text: str, lang: str, destination: Path) -> None:
    """Create a temporary 24 kHz WAV using Edge TTS for languages MeloTTS lacks."""
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Install edge-tts for this target language.") from exc

    mp3 = destination.with_suffix(".edge.mp3")

    async def synthesize() -> None:
        await edge_tts.Communicate(text, EDGE_VOICES[lang]).save(str(mp3))

    asyncio.run(synthesize())
    try:
        _run_ffmpeg(["ffmpeg", "-y", "-i", str(mp3), "-ac", "1", "-ar", "24000", str(destination)])
    finally:
        mp3.unlink(missing_ok=True)


def _try_indic_f5(text: str, reference_wav: Path, destination: Path) -> bool:
    """Optionally run IndicF5 when explicitly enabled and installed.

    IndicF5 is deliberately opt-in because its checkpoint can exceed the safe
    memory budget on an 8 GB M1.  Any unavailable package, model, or network
    failure returns ``False`` and lets the compact OpenVoice route continue.
    """
    if os.getenv("NEURALDUB_USE_INDIC_F5", "0") != "1":
        return False
    engine: Any | None = None
    try:
        from f5_tts.api import F5TTS

        engine = F5TTS(model="ai4bharat/IndicF5")
        # IndicF5 accepts an empty reference transcript; it infers vocal style
        # from the supplied short reference audio.
        engine.infer(ref_file=str(reference_wav), ref_text="", gen_text=text,
                     file_wave=str(destination))
        return destination.is_file() and destination.stat().st_size > 0
    except Exception:
        destination.unlink(missing_ok=True)
        return False
    finally:
        if engine is not None:
            del engine
        flush_memory()


def clone_and_synthesize(translated_text: str, reference_wav: str, tgt_lang: str, output_wav: str) -> str:
    """Synthesize translated speech and transfer the reference speaker's timbre.

    OpenVoice V2's converter is used for every language.  OpenVoice/MeloTTS is
    used where available; Edge TTS supplies a neutral source voice for Kannada
    and other unsupported Melo languages before tone-colour conversion.
    """
    if not translated_text.strip():
        raise ValueError("Cannot synthesize empty translated text.")
    if tgt_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported target language: {tgt_lang}")
    tgt_lang = LANGUAGE_ALIASES.get(tgt_lang, tgt_lang)
    reference = _require_file(reference_wav, "Reference WAV")
    destination = Path(output_wav).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".source.wav")
    converter: Any | None = None
    tts_model: Any | None = None
    try:
        import torch
        from openvoice import se_extractor
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter

        checkpoint_root = _ensure_openvoice_checkpoints()
        device = get_device()
        converter_dir = checkpoint_root / "converter"
        converter = ToneColorConverter(str(converter_dir / "config.json"), device=device)
        converter.load_ckpt(str(converter_dir / "checkpoint.pth"))
        # Retain 5--8 second references intact: VAD can otherwise remove useful
        # speaker identity cues from short clips.
        target_se, _ = se_extractor.get_se(str(reference), converter, vad=False)

        # OpenVoice V2 ships native base speakers for English and Spanish.
        native_language = {"eng_Latn": "EN_NEWEST", "spa_Latn": "ES"}.get(tgt_lang)
        if native_language:
            base_dir = checkpoint_root / "base_speakers" / native_language
            tts_model = BaseSpeakerTTS(str(base_dir / "config.json"), device=device)
            tts_model.load_ckpt(str(base_dir / "checkpoint.pth"))
            tts_model.tts(translated_text, str(temporary), speaker="default", language="English")
            source_embedding = torch.load(base_dir / "en_default_se.pth", map_location=device)
        else:
            used_indic_f5 = tgt_lang in {"hin_Deva", "kan_Knda"} and _try_indic_f5(
                translated_text, reference, temporary
            )
            if not used_indic_f5:
                _edge_tts_to_wav(translated_text, tgt_lang, temporary)
            source_embedding, _ = se_extractor.get_se(str(temporary), converter, vad=True)

        converter.convert(
            audio_src_path=str(temporary), src_se=source_embedding, tgt_se=target_se,
            output_path=str(destination), message="@NeuralDubStudio", tau=0.45,
        )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("OpenVoice completed without creating synthesized audio.")
        return str(destination)
    except ImportError as exc:
        raise RuntimeError("Install OpenVoice V2 and its dependencies to clone voices.") from exc
    finally:
        temporary.unlink(missing_ok=True)
        if tts_model is not None:
            del tts_model
        if converter is not None:
            del converter
        flush_memory()
