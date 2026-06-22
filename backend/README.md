---
title: Image Deblurring with NAF-Net
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Image Deblurring with NAF-Net — FastAPI Backend

`POST /deblur` — upload a blurred image, receive a deblurred PNG.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/deblur` | Upload blurred image → returns deblurred PNG |
| `GET` | `/` | Service info & model status |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

## Model

| Property | Value |
|---|---|
| Architecture | NAFNet-width32 |
| Parameters | 29.16M |
| Config | `middle_blk_num=12, enc=[2,2,4,8], dec=[2,2,2,2]` |
| Pretrained on | GoPro Large Dataset |
| Paper | ECCV 2022 |

## Notes

- Model weights (~600 MB) download from Hugging Face Hub on the **first** `/deblur` request (~60 s one-time). All subsequent requests are fast.
- Response header `X-Inference-Time` contains model-only inference time in seconds.
- Images are auto-converted to RGB; RGBA, grayscale, and palette images are handled correctly.
- Images larger than 1280 px on the long edge are resized before inference.
