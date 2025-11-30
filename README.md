# Occlusion-Removal-AI

A repo scaffold for a **live camera occlusion removal & predictive reconstruction** system.

This package includes:
- Segmentation (YOLOv8/SAM wrapper)
- Inpainting (fast OpenCV fallback + hooks for LaMa / E2FGVI / Stable Diffusion inpainting)
- Depth estimation (MiDaS fallback)
- Tracking (centroid tracker fallback, with ByteTrack + RAFT integration hooks)
- Pipeline that composes everything and outputs a cleaned live feed

See `config.yaml` for configuration.

