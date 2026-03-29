# Gaussian Toolkit

**Video-to-3D-scene pipeline built on LichtFeld Studio.**

Gaussian Toolkit is our fork of [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) (MrNeRF) that adds a complete video-to-structured-3D pipeline. LichtFeld Studio is the upstream product -- a native workstation for 3D Gaussian Splatting training, editing, and export. Gaussian Toolkit extends it with automated video ingestion, object segmentation, mesh extraction, and USD scene assembly, all running inside a single consolidated Docker container on a dedicated GPU workstation.

We do not modify upstream LichtFeld code. Our additions live in separate directories (`src/pipeline/`, `src/web/`, `docker/`, `scripts/`, `research/`). See [BOUNDARIES.md](BOUNDARIES.md) for the full separation policy.

---

## Quick Start

```bash
# 1. Clone and checkout the gaussian-toolkit branch
git clone <repo-url> && cd LichtFeld-Studio
git checkout gaussian-toolkit

# 2. Set your HuggingFace token (needed for model downloads)
export HF_TOKEN=hf_your_token_here

# 3. Build and start the container
docker compose -f docker-compose.consolidated.yml up -d

# 4. Open the web interface
#    http://localhost:7860
```

Services exposed by the container:

| Port  | Service              |
|-------|----------------------|
| 7860  | Web upload interface |
| 8188  | ComfyUI              |
| 45677 | LichtFeld MCP server |
| 5901  | VNC remote desktop   |

Upload a video at `:7860`, and the pipeline will produce a structured USD scene with per-object meshes.

---

## What This Fork Adds

LichtFeld Studio provides 3DGS training, visualisation, editing, and export. Gaussian Toolkit adds everything needed to go from a raw video file to a fully decomposed 3D scene:

### Pipeline Modules (24 files in `src/pipeline/`)

| Module | Purpose | Status |
|--------|---------|--------|
| `orchestrator.py` | End-to-end pipeline driver (PyAV frame extraction, COLMAP orchestration) | Working |
| `cli.py` / `__main__.py` | CLI entry point | Working |
| `config.py` | YAML/dict pipeline configuration | Working |
| `mcp_client.py` | LichtFeld MCP client for training control | Working |
| `sam2_segmentor.py` | SAM2 2D segmentation (grid-point prompts) | Working |
| `sam3_segmentor.py` | SAM3 segmentation wrapper | In Progress |
| `sam3d_client.py` | SAM3D concept segmentation client (4M concepts) | In Progress |
| `mask_projector.py` | 2D mask to 3D Gaussian label projection (98.3% coverage) | Working |
| `mesh_extractor.py` | Marching Cubes + TSDF mesh extraction (22K verts, 49K faces) | Working |
| `mesh_cleaner.py` | Decimation, hole filling, manifold repair | Working |
| `texture_baker.py` | UV unwrapping + texture bake (xatlas) | In Progress |
| `material_assigner.py` | PBR material assignment | In Progress |
| `usd_assembler.py` | USD scene assembly (59 prims, variant sets) | Working |
| `multiview_renderer.py` | Camera orbit renders for Hunyuan3D input | Working |
| `hunyuan3d_client.py` | Hunyuan3D 2.0 multi-view to textured mesh | Working |
| `comfyui_inpainter.py` | FLUX inpainting via ComfyUI for background recovery | Working |
| `person_remover.py` | Person removal from training views | Working |
| `frame_selector.py` | Keyframe selection for segmentation | Working |
| `frame_quality.py` | Blur/exposure quality scoring | Working |
| `coordinate_transform.py` | COLMAP <-> 3DGS <-> USD coordinate transforms | Working |
| `colmap_parser.py` | COLMAP binary model reader | Working (needs hardening) |
| `quality_gates.py` | Per-stage pass/fail quality checks | Working |

### Web Interface (`src/web/`)

Flask application on port 7860 with video upload, job tracking, and result download.

### Deployment (`docker/`, `Dockerfile.consolidated`, `docker-compose.consolidated.yml`)

Single consolidated Docker container running all services under supervisord. Designed for a dedicated GPU workstation (tested on dual RTX 6000 Ada, 96GB VRAM, 251GB RAM, Threadripper PRO 48-core).

### Research (`research/`)

15 research documents covering tool landscape, pipeline architecture decisions, segmentation methods, and mesh extraction approaches. This is research context, not product documentation.

### Utility Scripts (`scripts/`)

Pipeline runners, test harnesses, MCP bridge, hardware tracing, gallery assembly.

---

## Pipeline Architecture

```
Video (.mp4/.mov) or Web Upload (:7860)
    |
    v [Stage 1] Frame extraction (PyAV)
JPEG Frames
    |
    v [Stage 2] COLMAP SfM (feature extract -> match -> sparse -> undistort)
COLMAP Dataset
    |
    v [Stage 3] 3DGS Training (LichtFeld MCP, 7k iter, ~2m15s)
Trained Gaussian Splat (1M gaussians)
    |
    v [Stage 4] SAM2/SAM3 segmentation on training views
2D Object Masks
    |
    v [Stage 5] Mask projection to 3D Gaussians (98.3% coverage)
Per-Object PLY Files (33 objects in test scene)
    |
    v [Stage 6] Per-object mesh creation (Hunyuan3D 2.0 or TSDF fallback)
Textured Meshes
    |
    v [Stage 7] Background inpainting (FLUX via ComfyUI)
Clean Background Views
    |
    v [Stage 8] USD scene assembly (variant sets: Gaussian + Mesh)
USD Scene (59 prims) + PLY/SOG/SPZ/HTML exports
```

---

## Directory Boundaries

```
LichtFeld-Studio/
  src/
    core/          # UPSTREAM - LichtFeld core (do not modify)
    app/           # UPSTREAM - LichtFeld application
    mcp/           # UPSTREAM - LichtFeld MCP server
    rendering/     # UPSTREAM - LichtFeld rendering
    training/      # UPSTREAM - LichtFeld training
    pipeline/      # OURS - 24 pipeline modules
    web/           # OURS - Flask web interface
  research/        # OURS - Research documents (not product)
  docker/          # OURS - Docker configuration
  scripts/         # OURS - Utility scripts
  Dockerfile.consolidated     # OURS
  docker-compose.consolidated.yml  # OURS
  README.md        # UPSTREAM - Do not overwrite
```

See [BOUNDARIES.md](BOUNDARIES.md) for the complete policy.

---

## Feature Status Summary

| Category | Feature | Status |
|----------|---------|--------|
| Ingestion | Video frame extraction | Working |
| Ingestion | Web upload interface | Working |
| Reconstruction | COLMAP SfM | Working |
| Reconstruction | 3DGS training via MCP | Working |
| Segmentation | SAM2 (grid-point prompts) | Working |
| Segmentation | SAM3 (text+visual, 4M concepts) | In Progress |
| Mesh | TSDF extraction (Open3D) | Working |
| Mesh | Per-object Hunyuan3D 2.0 | Working |
| Mesh | Texture baking (xatlas) | In Progress |
| Mesh | Material assignment | In Progress |
| Scene | USD assembly with variant sets | Working |
| Scene | Background inpainting (FLUX) | Working |
| Infra | Consolidated Docker | Working |
| Infra | MCP bridge | Working |
| Infra | Quality gates | Working |
| Research | Audio-to-scene-graph naming | Planned |

---

## Known Limitations

1. **COLMAP is the bottleneck** -- ~20 min on 32 cores for 15 frames. No GPU BA solver available.
2. **SAM2 prompt quality varies** -- Grid-point prompts need tuning per scene. SAM3 upgrade will eliminate this.
3. **No UV texture baking yet** -- Meshes are vertex-coloured. xatlas integration is in progress.
4. **USD Gaussian variants are path references** -- Not embedded splat data.
5. **Single-machine deployment only** -- The consolidated container assumes all GPUs are local.

---

## MCP Integration

The pipeline can be driven programmatically through LichtFeld's MCP server (70+ tools on port 45677). See [AGENTS.md](AGENTS.md) for the agent operating guide.

```bash
# Example: start training from CLI
lfs-mcp call scene.load_dataset '{"path": "/opt/output/colmap"}'
lfs-mcp call training.start
lfs-mcp call training.get_state
```

---

## Contributing

Upstream contributions (LichtFeld core) go to [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio). Pipeline, web, Docker, and research changes stay on the `gaussian-toolkit` branch in this repository.
