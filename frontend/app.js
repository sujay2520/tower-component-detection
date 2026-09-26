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
const bannerContainer = document.getElementById("classification-banner-container");
const qualityCard = document.getElementById("quality-card");
const qualityStatus = document.getElementById("quality-status");
const qualityDetails = document.getElementById("quality-details");
const statsGrid = document.getElementById("stats-grid");
const detectionsBody = document.getElementById("detections-body");
const infoFilename = document.getElementById("info-filename");
const infoFilesize = document.getElementById("info-filesize");
const infoStatus = document.getElementById("info-status");
const systemBadge = document.getElementById("system-status-badge");

let currentJob = null;
let currentFile = null;

// Health check on startup
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.model_loaded) {
      systemBadge.innerHTML = '✅ Model Loaded & Active';
      systemBadge.style.background = 'rgba(34, 197, 94, 0.25)';
    } else {
      systemBadge.innerHTML = '⏳ Model Training in Progress...';
      systemBadge.style.background = 'rgba(234, 179, 8, 0.25)';
    }
  } catch (e) {
    systemBadge.innerHTML = '❌ Backend Offline';
    systemBadge.style.background = 'rgba(239, 68, 68, 0.25)';
  }
}
checkBackendHealth();
setInterval(checkBackendHealth, 8000);

// --- Navigation Tabs ---
function switchTab(tabId) {
  const detectTab = document.getElementById("tab-detect");
  const metricsTab = document.getElementById("tab-metrics");
  const detectView = document.getElementById("view-detect");
  const metricsView = document.getElementById("view-metrics");

  if (tabId === "detect") {
    detectTab.classList.add("active");
    metricsTab.classList.remove("active");
    detectView.classList.remove("hidden");
    metricsView.classList.add("hidden");
  } else {
    metricsTab.classList.add("active");
    detectTab.classList.remove("active");
    metricsView.classList.remove("hidden");
    detectView.classList.add("hidden");
    loadMetrics();
  }
}

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
  if (!bytes) return "0 B";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function handleFile(file) {
  currentFile = file;
  resultSection.classList.add("hidden");
  qualityCard.classList.add("hidden");

  // Show local preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
  };
  reader.readAsDataURL(file);

  infoFilename.textContent = file.name;
  infoFilesize.textContent = formatBytes(file.size);
  infoStatus.innerHTML = '<span class="status-badge status-processing">⏳ Uploading to server...</span>';

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
      infoStatus.innerHTML = '<span class="status-badge status-passed">✅ Image Ready for Analysis</span>';
      executeBtn.disabled = false;
    } else {
      infoStatus.innerHTML = '<span class="status-badge status-rejected">❌ Upload failed</span>';
    }
  };

  xhr.onerror = () => {
    progressBar.style.display = "none";
    infoStatus.innerHTML = '<span class="status-badge status-rejected">❌ Cannot connect to backend (port 5000)</span>';
  };

  xhr.send(formData);
}

// --- Execute Detection ---
executeBtn.addEventListener("click", async () => {
  if (!currentJob) return;

  executeBtn.disabled = true;
  executeBtn.innerHTML = '<span class="spinner"></span> Analyzing Architecture...';
  infoStatus.innerHTML = '<span class="status-badge status-processing">🔬 Running quality filter & YOLO detection...</span>';
  resultSection.classList.add("hidden");
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
    infoStatus.innerHTML = `<span class="status-badge status-rejected">❌ Execution error: ${err.message}</span>`;
  } finally {
    executeBtn.disabled = false;
    executeBtn.innerHTML = '🔍 Run Detection & Classification';
  }
});

function showRejected(data) {
  // Determine rejection type icon
  let icon = "🚫";
  let title = "Image Rejected by Quality Filter";
  const reason = (data.reason || "").toLowerCase();
  if (reason.includes("blur")) { icon = "📷"; title = "Image Rejected — Too Blurry"; }
  else if (reason.includes("overexpos")) { icon = "☀️"; title = "Image Rejected — Overexposed"; }
  else if (reason.includes("underexpos")) { icon = "🌑"; title = "Image Rejected — Underexposed"; }
  else if (reason.includes("contrast")) { icon = "🔲"; title = "Image Rejected — Low Contrast"; }
  else if (reason.includes("resolution")) { icon = "🔍"; title = "Image Rejected — Too Small"; }

  infoStatus.innerHTML = `<span class="status-badge status-rejected">${icon} ${title}</span>`;

  qualityCard.classList.remove("hidden");
  qualityStatus.innerHTML = `<span class="status-badge status-rejected">❌ ${data.reason}</span>`;

  if (data.quality_details) {
    let html = "";
    for (const [key, val] of Object.entries(data.quality_details)) {
      const label = key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
      html += `<div class="quality-item"><div class="ql">${label}</div><div class="qv">${val}</div></div>`;
    }
    html += `<div class="quality-item" style="margin-top:8px;color:#f59e0b;font-size:13px;">💡 Please upload a clearer, well-lit image for accurate tower detection.</div>`;
    qualityDetails.innerHTML = html;
  }
}

function showResults(data) {
  infoStatus.innerHTML = '<span class="status-badge status-passed">✅ Tower Analysis Complete</span>';

  // Quality check details
  if (data.quality_details) {
    qualityCard.classList.remove("hidden");
    qualityStatus.innerHTML = '<span class="status-badge status-passed">✅ Image Quality Passed Standard Thresholds</span>';
    let html = "";
    for (const [key, val] of Object.entries(data.quality_details)) {
      if (key !== "status") {
        html += `<div class="quality-item"><div class="ql">${key.replace("_", " ")}</div><div class="qv">${val}</div></div>`;
      }
    }
    qualityDetails.innerHTML = html;
  }

  // 1. PROMINENT HERO CLASSIFICATION BANNER
  const isSupporting = data.primary_classification === "supporting_tower";
  const bannerThemeClass = isSupporting ? "supporting" : "monopole";
  const confPct = (data.classification_confidence * 100).toFixed(1);

  bannerContainer.innerHTML = `
    <div class="classification-banner ${bannerThemeClass}">
      <div class="banner-icon">${data.classification_icon || (isSupporting ? "🗼" : "📍")}</div>
      <div class="banner-content">
        <div class="banner-tag">${data.classification_method || "Deep YOLO Detection"}</div>
        <div class="banner-title">${data.classification_title}</div>
        <div class="banner-desc">${data.classification_description}</div>
      </div>
      <div class="banner-badge">
        <div class="banner-conf-val">${confPct}%</div>
        <div class="banner-conf-lbl">Confidence</div>
      </div>
    </div>
  `;

  // 2. Output annotated image
  resultImg.src = API_BASE + data.output_image_url;
  resultSection.classList.remove("hidden");

  // 3. Quick stats (unified single confidence level matching the hero banner)
  const sa = data.structural_analysis;
  let verificationCards = '';
  if (sa && sa.details) {
    const d = sa.details;
    const edgeCount = d.total_lines || 0;
    const texStd = d.texture_std ? d.texture_std.toFixed(0) : 'N/A';
    const skyPct = d.sky_ratio ? (d.sky_ratio * 100).toFixed(0) : 'N/A';
    const saConf = sa.confidence ? (sa.confidence * 100).toFixed(0) : 'N/A';
    const saType = sa.tower_type ? sa.tower_type.replace('_', ' ').toUpperCase() : 'N/A';
    const methodMatch = data.classification_method && data.classification_method.includes('Confirmed') ? 'CONFIRMED' : 
                        data.classification_method && data.classification_method.includes('Override') ? 'OVERRIDE' : 'CHECKED';
    verificationCards = `
    <div class="stat-card">
      <div class="stat-value">${edgeCount}</div>
      <div class="stat-label">EDGES DETECTED (HOUGH)</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${texStd}</div>
      <div class="stat-label">TEXTURE COMPLEXITY</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${skyPct}%</div>
      <div class="stat-label">SKY RATIO</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${methodMatch}</div>
      <div class="stat-label">STRUCTURAL ANALYSIS</div>
    </div>
    `;
  }

  let statsHtml = `
    <div class="stat-card">
      <div class="stat-value">${confPct}%</div>
      <div class="stat-label">${data.primary_classification ? data.primary_classification.replace("_", " ").toUpperCase() : "TOWER"} CONFIDENCE</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${data.detection_count || 0}</div>
      <div class="stat-label">DETECTED COMPONENTS</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${isSupporting ? "Lattice Truss" : "Tubular Column"}</div>
      <div class="stat-label">STRUCTURAL PROFILE</div>
    </div>
    ${verificationCards}
  `;
  statsGrid.innerHTML = statsHtml;

  // 4. Detections table
  let tableHtml = "";
  if (data.detections && data.detections.length > 0) {
    for (const det of data.detections) {
      const dPct = (det.confidence * 100).toFixed(1);
      const isSupp = det.class === "supporting_tower";
      const badgeClass = isSupp ? "badge-supporting" : "badge-monopole";
      const icon = isSupp ? "🗼 " : "📍 ";
      const box = det.box.map(v => Math.round(v)).join(", ");

      tableHtml += `
        <tr>
          <td><span class="badge-class ${badgeClass}">${icon}${det.class.replace("_", " ").toUpperCase()}</span></td>
          <td><span class="conf-bar" style="width: ${Math.max(dPct, 12)}px"></span><strong>${dPct}%</strong></td>
          <td style="font-family: monospace; font-size: 0.85rem; color: #334155;">[${box}]</td>
        </tr>
      `;
    }
  } else {
    tableHtml = `
      <tr>
        <td colspan="3" style="text-align: center; color: var(--text-muted); padding: 18px;">
          Overall classified as <strong>${data.classification_title}</strong> via structural gradient & edge analysis.
        </td>
      </tr>
    `;
  }
  detectionsBody.innerHTML = tableHtml;

  // Smooth scroll to results
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// --- Reset Button ---
resetBtn.addEventListener("click", () => {
  currentJob = null;
  currentFile = null;
  previewSection.style.display = "none";
  resultSection.classList.add("hidden");
  qualityCard.classList.add("hidden");
  fileInput.value = "";
  progressBar.style.display = "none";
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// --- Metrics & Graphs Loader ---
async function loadMetrics() {
  try {
    // 1. Fetch JSON metrics
    const mRes = await fetch(`${API_BASE}/metrics`);
    if (mRes.ok) {
      const m = await mRes.json();
      document.getElementById("m-map50").textContent = m.mAP50 ? (m.mAP50 * 100).toFixed(1) + "%" : "—";
      document.getElementById("m-map50-95").textContent = (m.mAP50_95 || m["mAP50-95"]) ? ((m.mAP50_95 || m["mAP50-95"]) * 100).toFixed(1) + "%" : "—";
      document.getElementById("m-precision").textContent = m.precision ? (m.precision * 100).toFixed(1) + "%" : "—";
      document.getElementById("m-recall").textContent = m.recall ? (m.recall * 100).toFixed(1) + "%" : "—";
    }

    // 2. Fetch list of graph images
    const gRes = await fetch(`${API_BASE}/graphs`);
    if (gRes.ok) {
      const gData = await gRes.json();
      const gallery = document.getElementById("graphs-gallery");
      if (gData.graphs && gData.graphs.length > 0) {
        let gHtml = "";
        for (const g of gData.graphs) {
          gHtml += `
            <div class="gallery-card">
              <h4>${g.title}</h4>
              <a href="${API_BASE}${g.url}" target="_blank" title="Click to view full resolution">
                <img src="${API_BASE}${g.url}" alt="${g.title}">
              </a>
            </div>
          `;
        }
        gallery.innerHTML = gHtml;
      } else {
        document.getElementById("graphs-gallery").innerHTML = `
          <div style="color: var(--text-muted); padding: 20px;">
            Training plots will be populated once the training process completes.
          </div>
        `;
      }
    }
  } catch (err) {
    console.error("Failed to load metrics:", err);
  }
}
