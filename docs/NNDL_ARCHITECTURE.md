# NeuralDub Studio: NNDL Architecture and Experimental Method

This document records the architecture, training objective, evaluation protocol, and memory strategy used by NeuralDub Studio. It is intended to accompany the NNDL submission and to make the reported experiment reproducible.

## 1. End-to-end architecture

```mermaid
flowchart TB
    subgraph Input
        video[Source talking-head video]
        ref[16 kHz reference speech]
    end
    subgraph Audio pipeline — one model at a time
        asr[Faster-Whisper small\ntext + source language + segments]
        nllb[NLLB-200 distilled 600M\ntranslated text]
        synth[Native/Edge TTS or optional IndicF5\nsource-language speech]
        ov[OpenVoice V2 ToneColorConverter\nreference speaker embedding]
    end
    subgraph Latent visual synthesis — isolated process
        pose[DWPose face alignment\n256 x 256 crop]
        wa[Whisper-tiny audio features]
        vae[sd-vae-ft-mse\n4-channel latent encoder/decoder]
        unet[MuseTalk V1.5 conditional\n2D spatio-temporal UNet]
        blend[Face parsing / smooth blending]
    end
    subgraph Delivery
        api[FastAPI background job]
        ui[Antigravity Studio UI]
        telemetry[Spectrogram + pitch/energy\nframe sync confidence]
    end
    video --> asr --> nllb --> synth --> ov
    ref --> ov
    video --> pose --> vae --> unet --> blend
    ov --> wa --> unet
    blend --> api --> ui
    ref --> telemetry
    ov --> telemetry --> api
```

### Inference stages

1. **Audio extraction:** `ffmpeg` produces a 16 kHz mono PCM WAV. This makes ASR and reference extraction deterministic.
2. **ASR:** Faster-Whisper `small` produces source text, language, and segment timestamps. On MPS hosts Faster-Whisper runs CPU `int8`, because CTranslate2 does not offer a native MPS backend.
3. **Machine translation:** NLLB-200 Distilled 600M maps the source language token to `eng_Latn`, `hin_Deva`, `kan_Knda`, or `spa_Latn`.
4. **Speech synthesis/identity transfer:** Native OpenVoice base speakers are used where available. Indian-language IndicF5 can be enabled explicitly; otherwise Edge TTS produces a source voice and OpenVoice V2 transfers tone color from the reference audio. The short-reference embedding uses `vad=False`; OpenVoice conversion uses `tau=0.45` to strengthen identity transfer.
5. **Lip-sync:** MuseTalk V1.5 receives the source frames and synthesized audio. DWPose supplies face alignment; Whisper-tiny provides audio features; the SD VAE operates on 4-channel latent faces; the UNet inpaints motion around a lower-face mask at 256 x 256. Face parsing/smooth blending composites the generated region back into the original frame.
6. **Delivery:** FastAPI exposes job status and static artifacts to the Antigravity UI. The UI can plot per-frame visual/audio alignment confidence and display spectrogram telemetry.

## 2. Fine-tuning model and NNDL objective

For each temporal clip, let (x_t\) be the target face image and let (z_t = E_{\mathrm{VAE}}(x_t)\) be its frozen SD-VAE latent. The lower half of (z_t\) is masked to form (ẑ_t). Whisper features aligned to the video frame are denoted (a_t). The conditional MuseTalk UNet produces

\[
\hat z_{1:T} = f_\theta(\tilde z_{1:T}, a_{1:T}, \tau),
\]

where (τ) is the diffusion timestep. VAE and Whisper parameters are frozen. Only UNet cross-attention and decoder/up-block parameters are updated, which reduces trainable memory and limits identity drift.

The training objective is

\[
\mathcal{L} = \mathcal{L}_1 + \lambda_{\mathrm{vgg}}\mathcal{L}_{\mathrm{VGG}}
+ \lambda_{\mathrm{sync}}\mathcal{L}_{\mathrm{Sync}}
+ \lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{Temp}}.
\]

The notebook uses (λ_{\mathrm{vgg}}=0.20), (λ_{\mathrm{sync}}=0.10), and (λ_{\mathrm{temp}}=0.05). These should be treated as experiment hyperparameters, not fixed universal constants.

### Latent reconstruction

\[
\mathcal{L}_1 = \frac{1}{T}\sum_{t=1}^{T}\lVert\hat z_t - z_t\rVert_1.
\]

It anchors the generated latent to the ground-truth face and stabilizes early fine-tuning.

### Perceptual VGG/LPIPS loss

Let (D) be the frozen VAE decoder and (Φ_l) a VGG/LPIPS feature at layer (l). Then

\[
\mathcal{L}_{\mathrm{VGG}} = \sum_l w_l\,
\left\lVert \Phi_l(D(\hat z_t)) - \Phi_l(x_t) \right\rVert_1.
\]

This term targets lip contours, teeth boundaries, and local texture that are underweighted by a pure latent L1 loss.

### Audio-visual sync loss

With a *pretrained frozen* SyncNet video embedding (v_t) and audio embedding (u_t),

\[
\mathcal{L}_{\mathrm{Sync}} = 1 - \frac{1}{T}\sum_{t=1}^{T}
\frac{v_t^\top u_t}{\lVert v_t\rVert_2\lVert u_t\rVert_2 + \epsilon}.
\]

The reported full experiment must use a real pretrained SyncNet checkpoint. A zeroed SyncNet term is allowed only as a clearly labelled ablation; it must not be reported as the full system.

### Temporal consistency

\[
\mathcal{L}_{\mathrm{Temp}} = \frac{1}{T-1}\sum_{t=2}^{T}
\left\lVert(\hat z_t - \hat z_{t-1}) - (z_t-z_{t-1})\right\rVert_1.
\]

Matching latent motion differences penalizes frame-to-frame mouth jitter while preserving natural motion.

### Optimization protocol

| Item | Setting |
| --- | --- |
| Trainable modules | UNet cross-attention and decoder/up blocks |
| Frozen modules | SD VAE, Whisper audio encoder, LPIPS/VGG, SyncNet |
| Optimizer | AdamW, learning rate (10^{-5}), weight decay (10^{-2}) |
| Scheduler | Cosine annealing, minimum LR (10^{-6}) |
| Precision | CUDA fp16 autocast and `GradScaler` |
| Clip length | 8-frame random contiguous clips in notebook baseline |
| Validation | Fixed held-out speaker/video split, never used for optimization |

## 3. Quantitative evaluation

Baseline and fine-tuned models must render the *same held-out clips* with the same input audio, crop policy, FPS, and deterministic seed. Save raw per-video measurements before computing aggregates.

| Metric | Definition and direction | Measurement protocol |
| --- | --- | --- |
| **LSE-D** | SyncNet lip-sync embedding distance; lower is better. | Official pretrained SyncNet evaluator on each generated video/audio pair. |
| **LSE-C** | SyncNet synchronization confidence; higher is better. | Official pretrained SyncNet evaluator. Never infer this from optical flow. |
| **FID** | Fréchet distance between Inception feature distributions; lower is better. | `pytorch-fid` on equally sized, held-out real and generated face-frame directories. |
| **SSIM** | Mean structural similarity in ([-1,1]); higher is better. | Grayscale, resolution-matched paired frames. |
| **PSNR** | (10\log_{10}(MAX^2/\mathrm{MSE})) in dB; higher is better. | RGB, resolution-matched paired frames. |

`training/evaluate_metrics.py` computes SSIM and PSNR on paired videos and can invoke an external official SyncNet script to obtain LSE-D/LSE-C. It returns `null` for LSE values if no evaluator is provided—rather than inventing values. FID is intentionally calculated in the Colab evaluation cell because it requires directories of frame distributions.

The ablation study reruns the same held-out evaluation with one loss term disabled at a time:

- (-\mathcal{L}_{\mathrm{VGG}}): tests visual texture supervision.
- (-\mathcal{L}_{\mathrm{Sync}}): tests audio-video alignment supervision.
- (-\mathcal{L}_{\mathrm{Temp}}): tests jitter reduction.
- (-\mathcal{L}_1): optional instability control; report only if training remains stable.

Plot loss curves, baseline/fine-tuned metrics, and ablations under `training/plots/`. Report mean and standard deviation across held-out clips where the dataset is large enough.

## 4. Apple Silicon memory engineering

The M1 MacBook Air has a unified 8 GB memory pool, so GPU and CPU allocations compete. NeuralDub never intentionally keeps ASR, NLLB, voice cloning, and MuseTalk networks in memory together.

```text
extract (ffmpeg) ──► ASR model ──flush──► NLLB model ──flush──►
voice model(s) ──flush──► MuseTalk child process ──exits──► telemetry
```

### In-process stages

`transcribe_audio`, `translate_text`, and `clone_and_synthesize` load models locally inside their function, perform one stage, delete model references in `finally`, run `gc.collect()`, then call the accelerator cache clear routine:

- CUDA: `torch.cuda.empty_cache()`.
- Apple MPS: `torch.mps.empty_cache()`.
- CPU: garbage collection still runs.

`memory_utils.get_device()` prioritizes CUDA, then MPS, then CPU. Faster-Whisper chooses CPU `int8` on Apple Silicon to avoid unsupported CTranslate2 MPS use and reduce unified-memory pressure.

### MuseTalk process boundary

MuseTalk is the heaviest stage: VAE, Whisper encoder, DWPose, and UNet coexist during inference. `backend/lipsync_engine.py` launches the MuseTalk inference runner in a child process. When that runner exits, its address space and any allocator pools are reclaimed by the operating system. The parent then calls the same cache flush routine before processing another job.

This is stronger than only deleting Python variables, which does not always promptly return MPS/CUDA cached allocations. It is a memory-control design, not a guarantee: model version, input duration/resolution, other applications, and framework release affect the actual peak. For low-memory local demos use short clips, 256 px face crops, one job at a time, and close unrelated applications.

### Training vs. local inference

Fine-tuning belongs on Colab GPU. The notebook freezes VAE/Whisper/LPIPS/SyncNet, updates a subset of UNet parameters, uses fp16 and gradient accumulation, and exports an fp16 state dictionary. The local server loads custom weights only when a job requests them; it does not preload them at API startup.

## 5. Reproducible execution checklist

1. Record dataset licenses, speaker consent, exact train/validation split, GPU type, package versions, model hashes, random seed, and hyperparameters.
2. Cache raw preprocessing outputs and never mix identities or clips across train and validation sets.
3. Run the base and fine-tuned model on identical held-out inputs.
4. Run every ablation from the same initialization/split where feasible.
5. Save raw metrics and figures, then export `unet_finetuned.pth` and `training_metrics.json`.
6. Copy both files into local `models/`, restart the API, and verify `/api/health` reports the checkpoint as available.

For command-level instructions, see the project [README](../README.md).
