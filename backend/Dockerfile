# ── Image Deblurring with NAF-Net — Hugging Face Spaces Backend ──────────────
#
# Model weights (~600 MB) are downloaded lazily from Hugging Face Hub on the
# first POST /deblur request. The app starts immediately without them.
#
FROM python:3.10-slim

WORKDIR /app

# System deps — libgl1 required by some Pillow JPEG/WebP codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run as non-root for security
RUN useradd -m appuser
USER appuser

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# App starts immediately; model downloads on first /deblur call (~600MB, ~60s)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
