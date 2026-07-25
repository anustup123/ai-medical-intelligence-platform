// Base URL of the API. When the frontend is served BY the FastAPI app itself
// (as configured in api/main.py) this can stay empty (relative paths).
// If you deploy the frontend separately, set this to your API's public URL.
const API_BASE = "";

let selectedFile = null;

// ---- Tab switching ----------------------------------------------------
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`${btn.dataset.tab}-tab`).classList.add("active");
    if (btn.dataset.tab === "history") loadHistory();
  });
});

// ---- File selection ----------------------------------------------------
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const dropzoneText = document.getElementById("dropzoneText");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusMsg = document.getElementById("statusMsg");

dropzone.addEventListener("dragover", e => e.preventDefault());
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", e => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
  if (!["image/jpeg", "image/png"].includes(file.type)) {
    setStatus("Please select a JPEG or PNG image.", true);
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    dropzoneText.hidden = true;
  };
  reader.readAsDataURL(file);
  analyzeBtn.disabled = false;
  setStatus("");
}

function setStatus(msg, isError = false) {
  statusMsg.textContent = msg;
  statusMsg.classList.toggle("error", isError);
}

// ---- Analyze -----------------------------------------------------------
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  setStatus("Running deep learning inference + Grad-CAM + LLM report generation…");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    renderResult(data);
    setStatus("Analysis complete.");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    analyzeBtn.disabled = false;
  }
});

function renderResult(data) {
  const card = document.getElementById("resultCard");
  card.hidden = false;

  document.getElementById("originalPreview").src = previewImg.src;
  document.getElementById("gradcamPreview").src = `${API_BASE}${data.gradcam_image_url}`;

  const badge = document.getElementById("predictionBadge");
  badge.textContent = data.predicted_class;
  badge.className = "prediction-badge " + (data.predicted_class === "NORMAL" ? "normal" : "pneumonia");

  const confPct = (data.confidence * 100).toFixed(1);
  document.getElementById("confidenceValue").textContent = `${confPct}%`;
  document.getElementById("confidenceFill").style.width = `${confPct}%`;

  const probDiv = document.getElementById("probBreakdown");
  probDiv.innerHTML = "";
  Object.entries(data.probabilities).forEach(([label, prob]) => {
    const row = document.createElement("div");
    row.innerHTML = `<span>${label}</span><span>${(prob * 100).toFixed(1)}%</span>`;
    probDiv.appendChild(row);
  });

  document.getElementById("attentionRegion").textContent = data.attention_region;
  document.getElementById("llmReport").textContent = data.llm_report;

  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---- History -----------------------------------------------------------
async function loadHistory() {
  const listEl = document.getElementById("historyList");
  listEl.innerHTML = "<p>Loading…</p>";
  try {
    const res = await fetch(`${API_BASE}/api/history?limit=50`);
    const items = await res.json();
    if (!items.length) {
      listEl.innerHTML = "<p>No predictions yet.</p>";
      return;
    }
    listEl.innerHTML = "";
    items.forEach(item => {
      const el = document.createElement("div");
      el.className = "history-item";
      el.innerHTML = `
        <img src="${API_BASE}${item.gradcam_image_url}" alt="Grad-CAM">
        <div class="meta">
          <div class="label">${item.predicted_class} — ${(item.confidence * 100).toFixed(1)}%</div>
          <div class="sub">${item.original_filename || "unnamed"} · ${new Date(item.created_at).toLocaleString()}</div>
        </div>
      `;
      listEl.appendChild(el);
    });
  } catch (err) {
    listEl.innerHTML = `<p>Failed to load history: ${err.message}</p>`;
  }
}

document.getElementById("refreshHistoryBtn").addEventListener("click", loadHistory);
