"""
NAFNet Deblurring API — FastAPI Backend
Project: Image Deblurring with NAF-Net

Fix log vs original project:
  1. Lazy model loading — app starts cleanly before weights are ready
  2. Robust chunked weight download: .tmp file + size validation + cleanup on failure
  3. State-dict key handling: tries 'params', 'params_ema', 'state_dict'; strips 'module.' prefix
  4. Correct architecture: 29.16M params (middle_blk_num=12, enc=[2,2,4,8], dec=[2,2,2,2])
  5. content_type None guard — prevents 422 when client omits Content-Type
  6. _model_error cache — stops repeated failed load attempts on every request
  7. resize_if_needed uses max(1, ...) — prevents zero-dimension crash on tiny images
  8. try/except around inference — returns clean 500 instead of raw crash
  9. try/except in get_model — corrupted weight file is deleted so next deploy can retry
 10. Removed unused JSONResponse import
 11. asynccontextmanager / lifespan removed — was imported but unused

Endpoint: POST /deblur  →  returns deblurred PNG
"""

import io
import os
import time
import urllib.request
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from basicsr.models.archs.NAFNet_arch import create_nafnet_gopro

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "NAFNet-GoPro-width64.pth"
MAX_SIZE   = 1280   # resize long edge if image is very large

# Pretrained NAFNet-GoPro-width32 weights hosted on Hugging Face Hub
WEIGHTS_URL = (
    "https://huggingface.co/nyanko7/nafnet-models/resolve/main/"
    "NAFNet-GoPro-width64.pth"
)

# ── Global model handle — loaded lazily on first /deblur request ──────────────
_model       = None
_model_error = None   # cached error string; prevents retrying a known-bad state


# ── Weight download ───────────────────────────────────────────────────────────
def download_weights() -> None:
    """Download model weights from Hugging Face Hub if not already present."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        print(f"Weights already present: {MODEL_PATH}")
        return

    print(f"Downloading NAFNet-GoPro-width32.pth …")
    tmp_path = MODEL_PATH + ".tmp"
    try:
        req = urllib.request.Request(
            WEIGHTS_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp, \
                open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Weight download failed: {exc}") from exc

    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1_000_000:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError("Weight download produced an empty or missing file.")

    os.rename(tmp_path, MODEL_PATH)
    print("Weights downloaded successfully.")


# ── State-dict helpers ────────────────────────────────────────────────────────
def _extract_state_dict(checkpoint: object) -> dict:
    """Pull the actual weight dict from various checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ("params", "params_ema", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
        return checkpoint          # raw state_dict with no wrapper key
    # Bare tensor dict (older PyTorch saves)
    return checkpoint  # type: ignore[return-value]


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix added by DataParallel / DistributedDataParallel."""
    if any(k.startswith("module.") for k in state_dict):
        return {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
    return state_dict


# ── Lazy model loader ─────────────────────────────────────────────────────────
def get_model():
    """Load model on first call; return cached instance on subsequent calls."""
    global _model, _model_error

    if _model is not None:
        return _model

    # Surface a previously cached load failure immediately — don't hammer disk.
    if _model_error is not None:
        raise RuntimeError(_model_error)

    try:
        download_weights()

        model = create_nafnet_gopro()
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = _extract_state_dict(checkpoint)
        state_dict = _strip_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"NAFNet loaded on {DEVICE} | params: {n_params:.2f}M")

        _model = model
        return _model

    except Exception as exc:
        _model_error = str(exc)
        # Delete a potentially corrupt weight file so a fresh redeploy can retry.
        if os.path.exists(MODEL_PATH):
            try:
                os.remove(MODEL_PATH)
            except OSError:
                pass
        raise RuntimeError(f"Model failed to load: {exc}") from exc


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Image Deblurring with NAF-Net",
    description=(
        "Upload a blurred image → receive a deblurred PNG.\n"
        "Powered by NAFNet-GoPro-width32 (ECCV 2022, 29.16M params)."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your GitHub Pages URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Image helpers ─────────────────────────────────────────────────────────────
def resize_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > MAX_SIZE:
        scale = MAX_SIZE / max(w, h)
        # max(1, ...) prevents zero-dimension on pathologically small images
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.LANCZOS,
        )
    return img


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0   # HWC → [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
    return tensor.unsqueeze(0).to(DEVICE)             # 1CHW


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service":      "Image Deblurring with NAF-Net",
        "model":        "NAFNet-GoPro-width32",
        "params":       "29.16M",
        "device":       str(DEVICE),
        "model_loaded": _model is not None,
        "docs":         "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/deblur")
async def deblur(file: UploadFile = File(...)):
    """
    Upload a blurred image (JPG / PNG / WEBP / etc.) → returns deblurred PNG.

    Response header `X-Inference-Time` contains model runtime in seconds.
    First call also downloads model weights (~600 MB, ~60 s one-time cost).
    """
    # Guard: content_type can be None for some HTTP clients → avoid AttributeError
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Uploaded file must be an image "
                f"(received content_type='{content_type}')."
            ),
        )

    # Read and decode image
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        # convert handles RGBA, palette, grayscale → uniform RGB tensor
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image file.")

    img = resize_if_needed(img)

    # Load model (lazy, downloads weights on first call)
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}")

    # Run inference
    try:
        t0 = time.time()
        with torch.no_grad():
            inp = pil_to_tensor(img)
            out = model(inp)
        elapsed = round(time.time() - t0, 2)
        result_img = tensor_to_pil(out)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # Encode result as PNG
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="image/png",
        headers={"X-Inference-Time": str(elapsed)},
    )
