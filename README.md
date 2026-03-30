# Gaussian Toolkit

Video-to-3D scene pipeline built on [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio). Upload a video, get a textured polygonal mesh + USD scene.

## What It Does

1. Upload a video of a space (museum, gallery, room)
2. Claude Code orchestrates the pipeline autonomously
3. Get: textured mesh (OBJ), USD scene, depth maps, preview renders

## Quick Start

```bash
docker compose -f docker-compose.consolidated.yml up -d
# Open http://localhost:7860
# Provision Claude Code via terminal tab
# Upload a video
```

## Architecture

Single Docker container with:
- LichtFeld Studio (3DGS training, 70+ MCP tools)
- COLMAP (Structure-from-Motion)
- gsplat (depth rendering for mesh extraction)
- ComfyUI (SAM3D, FLUX inpainting)
- Claude Code (agentic orchestrator)
- Flask web UI with preview carousel

## Pipeline Stages

| Stage | Tool | Output |
|-------|------|--------|
| Frame extraction | PyAV | JPEG frames |
| COLMAP SfM | COLMAP 4.1.0 | Camera poses + sparse model |
| 3DGS Training | LichtFeld Studio | Trained gaussian PLY |
| Preview renders | gsplat | RGB + depth map previews |
| Segmentation | SAM2/SAM3 | Per-object masks |
| Mesh extraction | gsplat + TSDF | Textured OBJ mesh |
| USD assembly | OpenUSD | Hierarchical scene |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 12 GB | 48 GB |
| System RAM | 32 GB | 64 GB+ |
| Disk | 50 GB free | 200 GB+ |

## Ports

| Port | Service |
|------|---------|
| 7860 | Web UI (upload + monitor) |
| 7681 | Terminal (Claude Code) |
| 8188 | ComfyUI |
| 5901 | VNC (Blender) |

## Project Structure

See [BOUNDARIES.md](BOUNDARIES.md) for ownership map.

## License

Upstream LichtFeld Studio: GPL-3.0
Pipeline additions: GPL-3.0 (derivative work)
