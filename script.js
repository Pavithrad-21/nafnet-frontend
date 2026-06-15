// ─────────────────────────────────────────────────────────────────────────────
// CONFIG — replace YOUR_HF_USERNAME with your actual Hugging Face username
//          and YOUR_SPACE_NAME with the Space name you chose when deploying.
//
// Example: https://janedoe-nafnet-deblur.hf.space
// ─────────────────────────────────────────────────────────────────────────────
const API_BASE = "https://pavithraduraisamy-nafnet-deblur.hf.space";
const API_URL  = `${API_BASE}/deblur`;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropZone     = document.getElementById("dropZone");
const dropText     = document.getElementById("dropText");
const fileInput    = document.getElementById("fileInput");
const deblurBtn    = document.getElementById("deblurBtn");
const statusEl     = document.getElementById("status");
const warmupNote   = document.getElementById("warmupNote");
const resultsEl    = document.getElementById("results");
const originalImg  = document.getElementById("originalImg");
const deblurredImg = document.getElementById("deblurredImg");
const metaInfo     = document.getElementById("metaInfo");
const downloadBtn  = document.getElementById("downloadBtn");
const resetBtn     = document.getElementById("resetBtn");

let selectedFile    = null;
let isFirstRequest  = true;   // show warmup warning on first call

// ── Drag & Drop ───────────────────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    setFile(file);
  } else {
    showStatus("Please drop an image file (JPG, PNG, WEBP).", true);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  dropZone.classList.add("has-file");
  dropText.textContent = `✅ ${file.name}`;
  deblurBtn.disabled = false;
  hideStatus();
  warmupNote.hidden = true;
  resultsEl.hidden  = true;
}

// ── Deblur ────────────────────────────────────────────────────────────────────
deblurBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  deblurBtn.disabled = true;
  showStatus("⏳ Sending image to NAF-Net model…");

  // Warn user about cold-start weight download on first call
  if (isFirstRequest) {
    warmupNote.hidden = false;
    isFirstRequest = false;
  }

  // Show original preview immediately
  originalImg.src = URL.createObjectURL(selectedFile);

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const t0 = Date.now();
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${response.status}`);
    }

    const blob          = await response.blob();
    const inferenceTime = response.headers.get("X-Inference-Time") ?? "?";
    const totalTime     = ((Date.now() - t0) / 1000).toFixed(2);

    // Display deblurred result
    const resultURL    = URL.createObjectURL(blob);
    deblurredImg.src   = resultURL;
    metaInfo.textContent =
      `Model inference: ${inferenceTime}s · Total round-trip: ${totalTime}s · ${selectedFile.name}`;

    // Wire download button
    downloadBtn.onclick = () => {
      const a        = document.createElement("a");
      a.href         = resultURL;
      const baseName = selectedFile.name.replace(/\.[^.]+$/, "");
      a.download     = `deblurred_${baseName}.png`;
      a.click();
    };

    warmupNote.hidden = true;
    hideStatus();
    resultsEl.hidden  = false;

  } catch (err) {
    warmupNote.hidden = true;
    showStatus(`❌ ${err.message}`, true);
    deblurBtn.disabled = false;
  }
});

// ── Reset ─────────────────────────────────────────────────────────────────────
resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  dropZone.classList.remove("has-file", "dragover");
  dropText.textContent = "Drag & drop a blurred image here";
  deblurBtn.disabled   = true;
  resultsEl.hidden     = true;
  warmupNote.hidden    = true;
  hideStatus();
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function showStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.className   = isError ? "status error" : "status";
  statusEl.hidden      = false;
}
function hideStatus() {
  statusEl.hidden = true;
}
