/**
 * NeuralDub Studio — Frontend Application Controller
 *
 * Orchestrates source video upload/preview, built-in sample clip loading,
 * live 4-stage pipeline polling, synchronized side-by-side playback,
 * acoustic/visual diagnostics, and fine-tuning benchmark telemetry.
 */

const API = window.location.origin.startsWith("http")
  ? window.location.origin
  : "http://localhost:8000";

const $ = (selector) => document.querySelector(selector);

let jobTimer;
let healthTimer;
let syncChart;
let lossChart;
let objectUrl;

/* ==========================================================================
   1. Pipeline Stage Definitions & Helpers
   ========================================================================== */

const stages = {
  extracting_audio: ["stage1", "Extracting 16 kHz mono PCM source audio via ffmpeg"],
  transcribing: ["stage1", "Transcribing speech & sentence timestamps with Faster-Whisper (int8 CPU)"],
  translating: ["stage2", "Translating clause segments with NLLB-200 Distilled 600M"],
  cloning_voice: ["stage3", "Synthesizing natural prosody & cloning vocal timbre with OpenVoice V2 (tau=0.70)"],
  generating_telemetry: ["stage3", "Generating Mel-spectrogram & pitch/energy telemetry"],
  lip_syncing: ["stage4", "Rendering 256×256 latent lip-synced frames with MuseTalk 1.5 UNet"],
  complete: ["stage4", "Render package ready — side-by-side comparison unlocked"],
  failed: ["stage4", "Pipeline stopped"],
};

const displayUrl = (path) =>
  path ? (path.startsWith("http") ? path : `${API}${path}`) : "";

/* ==========================================================================
   2. Source File & Sample Clip Management
   ========================================================================== */

function setFile(file) {
  if (!file) return;
  if (!/video\/(mp4|quicktime)|\.(mp4|mov)$/i.test(file.type + file.name)) {
    return showError("Select an MP4 or MOV video source.");
  }

  $("#videoInput")._file = file;
  $("#fileName").textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
  $("#fileMeta").classList.remove("hidden");
  $("#inputPreview").classList.remove("hidden");

  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  $("#inputPreview").src = objectUrl;

  $("#emptyState").classList.remove("hidden");
  showError("");
}

function clearFile() {
  $("#videoInput").value = "";
  $("#videoInput")._file = null;
  $("#fileMeta").classList.add("hidden");
  $("#inputPreview").classList.add("hidden");
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
}

async function loadSampleClip() {
  const sampleBtn = $("#loadSample");
  if (sampleBtn) sampleBtn.disabled = true;
  showError("");

  try {
    const response = await fetch(`${API}/api/sample`);
    if (!response.ok) throw new Error("Sample video unavailable on server.");
    const blob = await response.blob();
    const file = new File([blob], "sample.mp4", { type: "video/mp4" });
    setFile(file);
    $("#stageDescription").textContent = "Loaded built-in sample.mp4 — ready to generate dubbed video.";
  } catch (error) {
    showError(error.message || "Could not load sample clip.");
  } finally {
    if (sampleBtn) sampleBtn.disabled = false;
  }
}

function showError(message) {
  $("#requestError").textContent = message;
}

function resetPipeline() {
  document.querySelectorAll(".stage-card").forEach((card) => {
    card.className = "stage-card";
    card.querySelector(".stage-state").textContent = "WAITING";
  });
  $("#overallProgress").style.width = "0%";
  $("#jobStatus").className = "badge neutral";
  $("#jobStatus").textContent = "QUEUED";
}

/* ==========================================================================
   3. Runtime Health Polling (/api/health)
   ========================================================================== */

async function checkHealth() {
  try {
    const response = await fetch(`${API}/api/health`);
    const health = await response.json();
    $("#backendDevice").textContent = (health.device || "cpu").toUpperCase();
    $("#availableRam").textContent = `${Math.round(health.available_ram_mb || 0)} MB`;
    $("#checkpointStatus").textContent = health.finetuned_weights_available
      ? "CUSTOM FINE-TUNED UNET"
      : "BASE MUSETALK 1.5";
    document.querySelector(".status-dot").classList.add("online");
  } catch {
    $("#backendDevice").textContent = "OFFLINE";
    $("#availableRam").textContent = "—";
    $("#checkpointStatus").textContent = "SERVER UNREACHABLE";
    document.querySelector(".status-dot").classList.remove("online");
  }
}

/* ==========================================================================
   4. Live Job Submission & Stage Progress Polling (/api/dub, /api/status)
   ========================================================================== */

function updateProgress(job) {
  const [active, description] = stages[job.stage] || [null, "Waiting for worker"];
  $("#overallProgress").style.width = `${job.percent || 0}%`;
  $("#jobRam").textContent = job.ram_mb ? `${job.ram_mb} MB` : "—";
  $("#stageDescription").textContent = job.error || description;

  const order = ["stage1", "stage2", "stage3", "stage4"];
  const activeIndex = order.indexOf(active);

  order.forEach((stage, index) => {
    const card = document.querySelector(`[data-stage="${stage}"]`);
    card.className = "stage-card";
    const label = card.querySelector(".stage-state");

    if (job.stage === "failed" && stage === active) {
      card.classList.add("error");
      label.textContent = "ERROR";
    } else if (index < activeIndex || (job.done && job.stage === "complete")) {
      card.classList.add("done");
      label.textContent = "COMPLETE";
    } else if (stage === active && !job.done) {
      card.classList.add("active");
      label.textContent = "ACTIVE";
    } else {
      label.textContent = "WAITING";
    }
  });

  $("#jobStatus").className = `badge ${job.done ? (job.error ? "error" : "done") : "active"}`;
  $("#jobStatus").textContent = job.error
    ? "FAILED"
    : job.done
      ? "COMPLETE"
      : (job.stage || "RUNNING").replaceAll("_", " ").toUpperCase();
}

async function pollJob(jobId) {
  try {
    const response = await fetch(`${API}/api/status/${jobId}`);
    if (!response.ok) throw new Error("Job status unavailable");
    const job = await response.json();
    updateProgress(job);

    if (job.done) {
      clearInterval(jobTimer);
      if (job.error) {
        showError(job.error);
      } else {
        showOutputs(job.result);
      }
    }
  } catch (error) {
    clearInterval(jobTimer);
    showError(error.message);
  }
}

async function submitJob() {
  const file = $("#videoInput")._file;
  if (!file) return showError("Add a source video (or click 'Load Studio Sample Clip') before generating.");

  const button = $("#generateButton");
  button.disabled = true;
  button.querySelector("span").textContent = "Initializing Pipeline…";
  showError("");
  resetPipeline();
  $("#outputs").classList.add("hidden");

  const data = new FormData();
  data.append("video_file", file);
  data.append("target_language", $("#language").value);
  data.append("use_finetuned_weights", $("#fineTuned").checked);

  try {
    const response = await fetch(`${API}/api/dub`, { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Unable to start render.");

    $("#jobId").textContent = payload.job_id.slice(0, 10).toUpperCase();
    await pollJob(payload.job_id);
    jobTimer = setInterval(() => pollJob(payload.job_id), 1500);
  } catch (error) {
    showError(error.message);
    $("#jobStatus").textContent = "ERROR";
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Generate Dubbed Video";
  }
}

/* ==========================================================================
   5. Output Presentation & Multimodal Diagnostics
   ========================================================================== */

function makeSyncChart(values) {
  if (syncChart) syncChart.destroy();
  const context = $("#syncChart");
  syncChart = new Chart(context, {
    type: "line",
    data: {
      labels: values.map((_, i) => i + 1),
      datasets: [
        {
          label: "Alignment confidence",
          data: values,
          borderColor: "#22d3ee",
          backgroundColor: "rgba(34,211,238,.12)",
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.32,
        },
      ],
    },
    options: chartOptions("Frame", "Confidence", 0.0, 1),
  });
}

function showOutputs(result) {
  $("#emptyState").classList.add("hidden");
  $("#outputs").classList.remove("hidden");

  $("#sourceOutput").src = objectUrl || "";
  $("#dubbedOutput").src = displayUrl(result.dubbed_video_url);
  $("#mouthZoom").src = displayUrl(result.mouth_zoom_url);
  $("#spectrogram").src = displayUrl(result.spectrogram_url);
  $("#downloadDub").href = displayUrl(result.dubbed_video_url);

  $("#transcript").textContent =
    result.transcript?.text || result.transcript || "No transcript returned.";
  $("#translation").textContent =
    result.translation || "No translation returned.";

  if ($("#detectedLangBadge") && result.transcript?.language) {
    $("#detectedLangBadge").textContent = `Detected: ${result.transcript.language.toUpperCase()} · Faster-Whisper`;
  }

  if ($("#voiceMetricsBadge") && result.voice_metrics) {
    const pitch = Math.round((result.voice_metrics.pitch_similarity || 0) * 100);
    const energy = Math.round((result.voice_metrics.energy_similarity || 0) * 100);
    $("#voiceMetricsBadge").textContent = `Pitch Similarity ${pitch}% · Energy ${energy}%`;
  }

  const audioComparison = $("#audioComparison");
  if (audioComparison && result.original_audio_url && result.cloned_audio_url) {
    $("#originalAudio").src = displayUrl(result.original_audio_url);
    $("#clonedAudio").src = displayUrl(result.cloned_audio_url);
    audioComparison.classList.remove("hidden");
  }

  makeSyncChart(result.sync_metrics || []);
  $("#outputs").scrollIntoView({ behavior: "smooth", block: "start" });
}

function chartOptions(xTitle, yTitle, min, max) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: "#9fb0c6",
          usePointStyle: true,
          boxWidth: 8,
          font: { family: "Manrope", size: 11 },
        },
      },
      tooltip: {
        backgroundColor: "#0b1220",
        borderColor: "#334155",
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        title: { display: true, text: xTitle, color: "#8ea0b8", font: { size: 10 } },
        ticks: { color: "#8ea0b8", maxTicksLimit: 8 },
        grid: { color: "rgba(148,163,184,.08)" },
      },
      y: {
        min,
        max,
        title: { display: true, text: yTitle, color: "#8ea0b8", font: { size: 10 } },
        ticks: { color: "#8ea0b8" },
        grid: { color: "rgba(148,163,184,.08)" },
      },
    },
  };
}

/* ==========================================================================
   6. Benchmark Telemetry & Optimization Curves (/api/training-metrics)
   ========================================================================== */

function renderMetrics(data) {
  const totalLoss = data.losses?.Total || [];
  const l1Loss = data.losses?.L1 || [];
  const base = {
    ...(data.benchmarks?.before || data.base || {}),
    TOTAL_LOSS: totalLoss[0],
    L1_LOSS: l1Loss[0],
  };
  const tuned = {
    ...(data.benchmarks?.after || data.finetuned || {}),
    TOTAL_LOSS: totalLoss.at(-1),
    L1_LOSS: l1Loss.at(-1),
  };

  const descriptors = [
    {
      key: "FID",
      label: "FID",
      low: true,
      decimals: 2,
      caption: "Visual realism distance (lower is better)",
      tip: "Fréchet Inception Distance: measures realism vs. real faces. Lower FID means sharper, more natural lip renders.",
    },
    {
      key: "SSIM",
      label: "SSIM",
      decimals: 2,
      caption: "Structural fidelity (higher is better)",
      tip: "Structural Similarity (0–1): measures mouth texture and edge accuracy. Higher SSIM means closer facial structure.",
    },
    {
      key: "PSNR",
      label: "PSNR",
      decimals: 2,
      caption: "Signal fidelity in dB (higher is better)",
      tip: "Peak Signal-to-Noise Ratio (dB): measures pixel-level fidelity. Higher PSNR means cleaner, less noisy synthesis.",
    },
    {
      key: "TOTAL_LOSS",
      label: "TRAIN LOSS",
      low: true,
      decimals: 4,
      caption: "Composite 4-term UNet objective",
      tip: "Combined L1, VGG perceptual, SyncNet, and temporal loss. Lower values show stronger UNet convergence.",
    },
    {
      key: "L1_LOSS",
      label: "LATENT L1",
      low: true,
      decimals: 4,
      caption: "Mouth-region latent pixel error",
      tip: "Direct L1 error on masked lower-face latents. Lower L1 means tighter frame-by-frame mouth shape accuracy.",
    },
  ];

  const template = $("#metricTemplate");
  const target = $("#metricCards");
  target.innerHTML = "";

  descriptors.forEach(({ key, label, low, decimals, caption, tip }) => {
    const node = template.content.cloneNode(true);
    const rawBefore = base[key];
    const rawAfter = tuned[key];
    const before = rawBefore == null ? NaN : Number(rawBefore);
    const after = rawAfter == null ? NaN : Number(rawAfter);
    const delta = Number.isFinite(before) && Number.isFinite(after) ? after - before : NaN;

    node.querySelector(".metric-name").textContent = label;
    const tipButton = node.querySelector(".info-tip");
    if (tipButton && tip) {
      tipButton.dataset.tip = tip;
      tipButton.setAttribute("aria-label", `${label} details`);
    }
    node.querySelector(".metric-base").textContent = Number.isFinite(before)
      ? `BASE ${before.toFixed(decimals)}`
      : "BASE —";
    node.querySelector(".metric-value").textContent = Number.isFinite(after)
      ? after.toFixed(decimals)
      : "—";

    const change = node.querySelector(".metric-delta");
    const better = low ? delta < 0 : delta > 0;
    change.textContent = Number.isFinite(delta)
      ? `${better ? "▲" : "▼"} ${Math.abs(delta).toFixed(decimals)} vs base`
      : "Awaiting benchmark run";

    if (Number.isFinite(delta) && !better) {
      change.classList.add("negative");
    }
    node.querySelector(".metric-caption").textContent = caption;
    target.append(node);
  });
}

function renderLossChart(losses = {}, validation = []) {
  if (lossChart) lossChart.destroy();
  const palette = ["#22d3ee", "#818cf8", "#f59e0b", "#f472b6", "#34d399"];
  const labels = Array.from(
    { length: Math.max(...Object.values(losses).map((v) => v.length), validation.length, 0) },
    (_, i) => i + 1
  );

  const niceNames = {
    L1: "L1 Reconstruction",
    Perceptual_VGG: "Perceptual VGG",
    SyncNet: "SyncNet Alignment",
    Temporal: "Temporal Consistency",
    Total: "Total Loss",
  };

  const datasets = Object.entries(losses)
    .filter(([key]) => niceNames[key])
    .map(([key, values], index) => ({
      label: niceNames[key],
      data: values,
      borderColor: palette[index],
      borderWidth: 2,
      tension: 0.3,
      pointRadius: 2.5,
    }));

  if (validation.length) {
    datasets.push({
      label: "Validation Total",
      data: validation,
      borderColor: "#f8fafc",
      borderDash: [5, 4],
      pointRadius: 1.5,
    });
  }

  lossChart = new Chart($("#lossChart"), {
    type: "line",
    data: { labels, datasets },
    options: chartOptions("Epoch", "Loss"),
  });
}

async function loadBenchmarks() {
  try {
    const response = await fetch(`${API}/api/training-metrics`);
    if (!response.ok) throw new Error("Telemetry unavailable");
    const data = await response.json();
    renderMetrics(data);
    renderLossChart(data.losses, data.validation_loss || []);
    $("#trainingState").className = "badge done";
    $("#trainingState").textContent = data.status === "not_trained" ? "REFERENCE" : "TRAINED";
  } catch {
    $("#trainingState").className = "badge error";
    $("#trainingState").textContent = "OFFLINE";
  }
}

/* ==========================================================================
   7. Event Listeners & Initialization
   ========================================================================== */

document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab,.workspace").forEach((el) => el.classList.remove("active"));
    tab.classList.add("active");
    $(`#${tab.dataset.tab}`).classList.add("active");
    if (tab.dataset.tab === "benchmarks") loadBenchmarks();
  })
);

$("#videoInput").addEventListener("change", (event) => setFile(event.target.files[0]));
$("#clearFile").addEventListener("click", clearFile);
$("#generateButton").addEventListener("click", submitJob);

if ($("#loadSample")) {
  $("#loadSample").addEventListener("click", loadSampleClip);
}

$("#dropzone").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("#dropzone").classList.add("dragover");
});

$("#dropzone").addEventListener("dragleave", () => $("#dropzone").classList.remove("dragover"));

$("#dropzone").addEventListener("drop", (event) => {
  event.preventDefault();
  $("#dropzone").classList.remove("dragover");
  setFile(event.dataTransfer.files[0]);
});

$("#playBoth").addEventListener("click", () => {
  const source = $("#sourceOutput");
  const output = $("#dubbedOutput");
  source.currentTime = 0;
  output.currentTime = 0;
  source.play().catch(() => {});
  output.play().catch(() => {});
});

checkHealth();
healthTimer = setInterval(checkHealth, 5000);
loadBenchmarks();

// Support click/tap toggling on static and dynamically rendered info-tip buttons
document.addEventListener("click", (event) => {
  const tip = event.target.closest(".info-tip");
  if (tip) {
    event.preventDefault();
    event.stopPropagation();
    const wasOpen = tip.classList.contains("open");
    document.querySelectorAll(".info-tip.open").forEach((el) => el.classList.remove("open"));
    if (!wasOpen) tip.classList.add("open");
    return;
  }
  document.querySelectorAll(".info-tip.open").forEach((el) => el.classList.remove("open"));
});
