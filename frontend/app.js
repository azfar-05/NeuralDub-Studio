const API = window.location.origin.startsWith("http") ? window.location.origin : "http://localhost:8000";
const $ = (selector) => document.querySelector(selector);
let jobTimer, healthTimer, syncChart, lossChart, objectUrl;
const stages = {
  extracting_audio: ["stage1", "Extracting 16 kHz source audio"], transcribing: ["stage1", "Transcribing source speech"],
  translating: ["stage2", "Translating with NLLB-200"], cloning_voice: ["stage3", "Synthesizing and matching speaker timbre"],
  generating_telemetry: ["stage3", "Generating acoustic diagnostics"], lip_syncing: ["stage4", "Rendering latent lip motion"],
  complete: ["stage4", "Render package ready"], failed: ["stage4", "Pipeline stopped"]
};
const displayUrl = (path) => path ? (path.startsWith("http") ? path : `${API}${path}`) : "";

function setFile(file) {
  if (!file) return;
  if (!/video\/(mp4|quicktime)|\.(mp4|mov)$/i.test(file.type + file.name)) return showError("Select an MP4 or MOV video source.");
  $("#videoInput")._file = file; $("#fileName").textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
  $("#fileMeta").classList.remove("hidden"); $("#inputPreview").classList.remove("hidden");
  if (objectUrl) URL.revokeObjectURL(objectUrl); objectUrl = URL.createObjectURL(file); $("#inputPreview").src = objectUrl;
  $("#emptyState").classList.remove("hidden"); showError("");
}
function clearFile() { $("#videoInput").value = ""; $("#videoInput")._file = null; $("#fileMeta").classList.add("hidden"); $("#inputPreview").classList.add("hidden"); if (objectUrl) URL.revokeObjectURL(objectUrl); objectUrl = null; }
function showError(message) { $("#requestError").textContent = message; }
function resetPipeline() { document.querySelectorAll(".stage-card").forEach((card) => { card.className = "stage-card"; card.querySelector(".stage-state").textContent = "WAITING"; }); $("#overallProgress").style.width = "0%"; $("#jobStatus").className = "badge neutral"; $("#jobStatus").textContent = "QUEUED"; }

async function checkHealth() {
  try { const health = await (await fetch(`${API}/api/health`)).json(); $("#backendDevice").textContent = (health.device || "cpu").toUpperCase(); $("#availableRam").textContent = `${Math.round(health.available_ram_mb || 0)} MB`; $("#checkpointStatus").textContent = health.finetuned_weights_available ? "CUSTOM FINE-TUNED UNET" : "BASE MUSETALK 1.5"; document.querySelector(".status-dot").classList.add("online"); }
  catch { $("#backendDevice").textContent = "OFFLINE"; $("#availableRam").textContent = "—"; $("#checkpointStatus").textContent = "SERVER UNREACHABLE"; document.querySelector(".status-dot").classList.remove("online"); }
}

function updateProgress(job) {
  const [active, description] = stages[job.stage] || [null, "Waiting for worker"];
  $("#overallProgress").style.width = `${job.percent || 0}%`; $("#jobRam").textContent = job.ram_mb ? `${job.ram_mb} MB` : "—"; $("#stageDescription").textContent = job.error || description;
  const order = ["stage1", "stage2", "stage3", "stage4"], activeIndex = order.indexOf(active);
  order.forEach((stage, index) => { const card = document.querySelector(`[data-stage="${stage}"]`); card.className = "stage-card"; const label = card.querySelector(".stage-state"); if (job.stage === "failed" && stage === active) { card.classList.add("error"); label.textContent = "ERROR"; } else if (index < activeIndex || job.done && job.stage === "complete") { card.classList.add("done"); label.textContent = "COMPLETE"; } else if (stage === active && !job.done) { card.classList.add("active"); label.textContent = "ACTIVE"; } else label.textContent = "WAITING"; });
  $("#jobStatus").className = `badge ${job.done ? (job.error ? "error" : "done") : "active"}`; $("#jobStatus").textContent = job.error ? "FAILED" : job.done ? "COMPLETE" : (job.stage || "RUNNING").replaceAll("_", " ").toUpperCase();
}

async function pollJob(jobId) {
  try { const response = await fetch(`${API}/api/status/${jobId}`); if (!response.ok) throw new Error("Job status unavailable"); const job = await response.json(); updateProgress(job); if (job.done) { clearInterval(jobTimer); if (job.error) showError(job.error); else showOutputs(job.result); } }
  catch (error) { clearInterval(jobTimer); showError(error.message); }
}

async function submitJob() {
  const file = $("#videoInput")._file; if (!file) return showError("Add a source video before generating.");
  const button = $("#generateButton"); button.disabled = true; button.querySelector("span").textContent = "Initializing Pipeline…"; showError(""); resetPipeline(); $("#outputs").classList.add("hidden");
  const data = new FormData(); data.append("video_file", file); data.append("target_language", $("#language").value); data.append("use_finetuned_weights", $("#fineTuned").checked);
  try { const response = await fetch(`${API}/api/dub`, { method: "POST", body: data }); const payload = await response.json(); if (!response.ok) throw new Error(payload.detail || "Unable to start render."); $("#jobId").textContent = payload.job_id.slice(0, 10).toUpperCase(); await pollJob(payload.job_id); jobTimer = setInterval(() => pollJob(payload.job_id), 1500); }
  catch (error) { showError(error.message); $("#jobStatus").textContent = "ERROR"; }
  finally { button.disabled = false; button.querySelector("span").textContent = "Generate Dubbed Video"; }
}

function makeSyncChart(values) { if (syncChart) syncChart.destroy(); const context = $("#syncChart"); syncChart = new Chart(context, { type: "line", data: { labels: values.map((_, i) => i + 1), datasets: [{ label: "Alignment confidence", data: values, borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,.12)", fill: true, borderWidth: 2, pointRadius: 0, tension: .32 }] }, options: chartOptions("Frame", "Confidence", .0, 1) }); }
function showOutputs(result) { $("#emptyState").classList.add("hidden"); $("#outputs").classList.remove("hidden"); $("#sourceOutput").src = objectUrl || ""; $("#dubbedOutput").src = displayUrl(result.dubbed_video_url); $("#mouthZoom").src = displayUrl(result.mouth_zoom_url); $("#spectrogram").src = displayUrl(result.spectrogram_url); $("#downloadDub").href = displayUrl(result.dubbed_video_url); $("#transcript").textContent = result.transcript?.text || result.transcript || "No transcript returned."; $("#translation").textContent = result.translation || "No translation returned."; makeSyncChart(result.sync_metrics || []); $("#outputs").scrollIntoView({ behavior: "smooth", block: "start" }); }
function chartOptions(xTitle, yTitle, min, max) { return { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: "#9fb0c6", usePointStyle: true, boxWidth: 8, font: { family: "Manrope", size: 10 } } }, tooltip: { backgroundColor: "#101827", borderColor: "#334155", borderWidth: 1 } }, scales: { x: { title: { display: true, text: xTitle, color: "#718198", font: { size: 10 } }, ticks: { color: "#718198", maxTicksLimit: 8 }, grid: { color: "rgba(148,163,184,.08)" } }, y: { min, max, title: { display: true, text: yTitle, color: "#718198", font: { size: 10 } }, ticks: { color: "#718198" }, grid: { color: "rgba(148,163,184,.08)" } } } }; }

function renderMetrics(data) { const base = data.benchmarks?.before || data.base || {}, tuned = data.benchmarks?.after || data.finetuned || {}; const descriptors = [{ key: "LSE-D", label: "LSE-D", low: true, caption: "Lip sync error distance" }, { key: "LSE-C", label: "LSE-C", caption: "Lip sync confidence" }, { key: "FID", label: "FID", low: true, caption: "Visual realism distance" }, { key: "SSIM", label: "SSIM", caption: "Structural fidelity" }, { key: "PSNR", label: "PSNR", caption: "Signal fidelity (dB)" }]; const template = $("#metricTemplate"), target = $("#metricCards"); target.innerHTML = ""; descriptors.forEach(({ key, label, low, caption }) => { const node = template.content.cloneNode(true); const before = Number(base[key]), after = Number(tuned[key]); const delta = Number.isFinite(before) && Number.isFinite(after) ? after - before : NaN; node.querySelector(".metric-name").textContent = label; node.querySelector(".metric-base").textContent = Number.isFinite(before) ? `BASE ${before.toFixed(2)}` : "BASE —"; node.querySelector(".metric-value").textContent = Number.isFinite(after) ? after.toFixed(2) : "—"; const change = node.querySelector(".metric-delta"); const better = low ? delta < 0 : delta > 0; change.textContent = Number.isFinite(delta) ? `${better ? "▲" : "▼"} ${Math.abs(delta).toFixed(2)} vs base` : "Awaiting benchmark run"; if (Number.isFinite(delta) && !better) change.classList.add("negative"); node.querySelector(".metric-caption").textContent = caption; target.append(node); }); }
function renderLossChart(losses = {}, validation = []) { if (lossChart) lossChart.destroy(); const palette = ["#22d3ee", "#818cf8", "#f59e0b", "#f472b6", "#34d399"]; const labels = Array.from({ length: Math.max(...Object.values(losses).map(v => v.length), validation.length, 0) }, (_, i) => i + 1); const niceNames = { L1: "L1 Reconstruction", Perceptual_VGG: "Perceptual VGG", SyncNet: "SyncNet Alignment", Temporal: "Temporal Consistency", Total: "Total Loss" }; const datasets = Object.entries(losses).filter(([key]) => niceNames[key]).map(([key, values], index) => ({ label: niceNames[key], data: values, borderColor: palette[index], borderWidth: 2, tension: .3, pointRadius: 2 })); if (validation.length) datasets.push({ label: "Validation Total", data: validation, borderColor: "#f8fafc", borderDash: [5, 4], pointRadius: 1 }); lossChart = new Chart($("#lossChart"), { type: "line", data: { labels, datasets }, options: chartOptions("Epoch", "Loss") }); }
async function loadBenchmarks() { try { const response = await fetch(`${API}/api/training-metrics`); if (!response.ok) throw new Error("Telemetry unavailable"); const data = await response.json(); renderMetrics(data); renderLossChart(data.losses, data.validation_loss || []); $("#trainingState").className = "badge done"; $("#trainingState").textContent = data.status === "not_trained" ? "REFERENCE" : "TRAINED"; } catch (error) { $("#trainingState").className = "badge error"; $("#trainingState").textContent = "OFFLINE"; } }

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => { document.querySelectorAll(".tab,.workspace").forEach(el => el.classList.remove("active")); tab.classList.add("active"); $(`#${tab.dataset.tab}`).classList.add("active"); if (tab.dataset.tab === "benchmarks") loadBenchmarks(); }));
$("#videoInput").addEventListener("change", (event) => setFile(event.target.files[0])); $("#clearFile").addEventListener("click", clearFile); $("#generateButton").addEventListener("click", submitJob); $("#dropzone").addEventListener("dragover", (event) => { event.preventDefault(); $("#dropzone").classList.add("dragover"); }); $("#dropzone").addEventListener("dragleave", () => $("#dropzone").classList.remove("dragover")); $("#dropzone").addEventListener("drop", (event) => { event.preventDefault(); $("#dropzone").classList.remove("dragover"); setFile(event.dataTransfer.files[0]); });
$("#playBoth").addEventListener("click", () => { const source = $("#sourceOutput"), output = $("#dubbedOutput"); source.currentTime = 0; output.currentTime = 0; source.play().catch(() => {}); output.play().catch(() => {}); });
checkHealth(); healthTimer = setInterval(checkHealth, 5000); loadBenchmarks();
