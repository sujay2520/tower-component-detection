const API_BASE = "http://localhost:5000";

// DOM Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const progressBar = document.getElementById("progress-bar");
const progressFill = document.getElementById("progress-fill");
const previewSection = document.getElementById("preview-section");
const previewImg = document.getElementById("preview-img");
const executeBtn = document.getElementById("execute-btn");
const resetBtn = document.getElementById("reset-btn");
const resultSection = document.getElementById("result-section");
const resultImg = document.getElementById("result-img");
const qualityCard = document.getElementById("quality-card");
const qualityStatus = document.getElementById("quality-status");
const qualityDetails = document.getElementById("quality-details");
const statsGrid = document.getElementById("stats-grid");
const detectionsBody = document.getElementById("detections-body");
const infoFilename = document.getElementById("info-filename");
const infoFilesize = document.getElementById("info-filesize");
const infoStatus = document.getElementById("info-status");

let currentJob = null;
let currentFile = null;

// --- Upload Handlers ---
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function handleFile(file) {
  currentFile = file;
  resultSection.style.display = "none";
  qualityCard.classList.add("hidden");

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
  };
  reader.readAsDataURL(file);

  infoFilename.textContent = file.name;
  infoFilesize.textContent = formatBytes(file.size);
  infoStatus.innerHTML = '<span class="status-badge status-processing">⏳ Uploading...</span>';

  previewSection.style.display = "block";
  executeBtn.disabled = true;
  progressBar.style.display = "block";
  progressFill.style.width = "0%";

  uploadFile(file);
}

function uploadFile(file) {
  const formData = new FormData();
  formData.append("image", file);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `${API_BASE}/upload`);

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = (e.loaded / e.total) * 100;
      progressFill.style.width = pct + "%";
    }
  };

  xhr.onload = () => {
    progressBar.style.display = "none";
    if (xhr.status === 200) {
      const data = JSON.parse(xhr.responseText);
      currentJob = data;
      infoStatus.innerHTML = '<span class="status-badge status-passed">✅ Uploaded — ready to process</span>';
      executeBtn.disabled = false;
    } else {
      infoStatus.innerHTML = '<span class="status-badge status-rejected">❌ Upload failed</span>';
    }
  };

  xhr.onerror = () => {
    progressBar.style.display = "none";
    infoStatus.innerHTML = '<span class="status-badge status-rejected">❌ Connection error — is the backend running?</span>';
  };

  xhr.send(formData);
}

// --- Execute Detection ---
executeBtn.addEventListener("click", async () => {
  if (!currentJob) return;

  executeBtn.disabled = true;
  executeBtn.innerHTML = '<span class="spinner"></span> Processing...';
  infoStatus.innerHTML = '<span class="status-badge status-processing">🔍 Running quality check & detection...</span>';
  resultSection.style.display = "none";
  qualityCard.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentJob)
    });
    const data = await res.json();

    if (data.status === "rejected") {
      showRejected(data);
      return;
    }

    if (data.error) {
      infoStatus.innerHTML = `<span class="status-badge status-rejected">❌ ${data.error}</span>`;
      return;
    }

    showResults(data);

  } catch (err) {
    infoStatus.innerHTML = `<span class="status-badge status-rejected">❌ Error: ${err.message}</span>`;
  } finally {
    executeBtn.disabled = false;
    executeBtn.innerHTML = '🔍 Run Detection';
  }
});

function showRejected(data) {
  infoStatus.innerHTML = `<span class="status-badge status-rejected">🚫 Image Rejected</span>`;

  qualityCard.classList.remove("hidden");
  qualityStatus.innerHTML = `<span class="status-badge status-rejected">❌ ${data.reason}</span>`;

  if (data.quality_details) {
    let html = "";
    for (const [key, val] of Object.entries(data.quality_details)) {
      html += `<div class="quality-item"><div class="ql">${key}</div><div class="qv">${val}</div></div>`;
    }
    qualityDetails.innerHTML = html;
  }
}

function showResults(data) {
  infoStatus.innerHTML = '<span class="status-badge status-passed">✅ Detection Complete</span>';

  // Quality card
  if (data.quality_details) {
    qualityCard.classList.remove("hidden");
    qualityStatus.innerHTML = '<span class="status-badge status-passed">✅ Quality check passed</span>';
    let html = "";
    for (const [key, val] of Object.entries(data.quality_details)) {
      html += `<div class="quality-item"><div class="ql">${key}</div><div class="qv">${val}</div></div>`;
    }
    qualityDetails.innerHTML = html;
  }

  // Result image
  resultImg.src = API_BASE + data.output_image_url;
  resultSection.style.display = "block";

  // Stats
  let statsHtml = `<div class="stat-card"><div class="stat-value">${data.detection_count || data.detections.length}</div><div class="stat-label">Detections</div></div>`;

  for (const [cls, conf] of Object.entries(data.average_confidence)) {
    statsHtml += `<div class="stat-card"><div class="stat-value">${conf !== null ? (conf * 100).toFixed(1) + "%" : "—"}</div><div class="stat-label">${cls.replace("_", " ")}</div></div>`;
  }
  statsGrid.innerHTML = statsHtml;

  // Detections table
  let tableHtml = "";
  for (const det of data.detections) {
    const confPct = (det.confidence * 100).toFixed(1);
    const barWidth = Math.max(confPct, 10);
    const box = det.box.map(v => Math.round(v)).join(", ");
    tableHtml += `<tr>
      <td><strong>${det.class.replace("_", " ")}</strong></td>
      <td><span class="conf-bar" style="width:${barWidth}px"></span>${confPct}%</td>
      <td style="font-family: monospace; font-size: 0.85rem;">[${box}]</td>
    </tr>`;
  }
  if (data.detections.length === 0) {
    tableHtml = '<tr><td colspan="3" style="text-align:center;color:var(--text-muted)">No detections found</td></tr>';
  }
  detectionsBody.innerHTML = tableHtml;
}

// --- Reset ---
resetBtn.addEventListener("click", () => {
  currentJob = null;
  currentFile = null;
  previewSection.style.display = "none";
  resultSection.style.display = "none";
  qualityCard.classList.add("hidden");
  fileInput.value = "";
  progressBar.style.display = "none";
});
