# Code Boundaries

This document defines what is upstream (LichtFeld Studio), what is our addition (Gaussian Toolkit), and what is experimental. It exists to prevent identity drift between the two projects and to make merge decisions obvious.

## Upstream: LichtFeld Studio (MrNeRF)

LichtFeld Studio is the upstream product. It is a native C++23/CUDA workstation for 3D Gaussian Splatting developed by MrNeRF. We fork it; we do not modify it.

**Upstream directories -- do not modify on the gaussian-toolkit branch:**

| Directory | Contents |
|-----------|----------|
| `src/core/` | Core data structures, Gaussian representation, scene graph |
| `src/app/` | Application entry point, GUI tools, main loop |
| `src/mcp/` | Built-in MCP HTTP server (JSON-RPC, tool/resource registries) |
| `src/rendering/` | Rasterization, camera, viewport |
| `src/training/` | Training loop, optimizers, schedulers |
| `src/geometry/` | Spatial data structures |
| `src/io/` | Import/export (PLY, SOG, SPZ, HTML) |
| `src/sequencer/` | Timeline/animation |
| `src/visualizer/` | GUI framework, panels, assets |
| `src/python/` | Embedded Python plugin runtime |
| `cmake/` | Build system configuration |
| `external/` | Git submodules (OpenMesh, nvImageCodec, libvterm, etc.) |
| `eval/` | Evaluation scripts |
| `tools/` | CLI wrappers shipped by upstream |
| `tests/` | Upstream test suite |
| `CMakeLists.txt` | Root build file |
| `vcpkg.json` | C++ dependency manifest |
| `README.md` | Upstream README (do not overwrite) |
| `CONTRIBUTING.md` | Upstream contributing guide |
| `LICENSE` | GPL-3.0 |
| `THIRD_PARTY_LICENSES.md` | Upstream third-party notices |

**Merge policy:** Periodically rebase or merge from upstream `main`. Conflicts in upstream directories are resolved in favour of upstream.

## Our Addition: Gaussian Toolkit

Everything below is written and maintained by us on the `gaussian-toolkit` branch. Upstream does not contain these directories.

### Pipeline (`src/pipeline/`) -- 24 modules

The video-to-structured-3D pipeline. Takes a video file and produces a USD scene with per-object Gaussian and mesh representations.

Core modules: `orchestrator.py`, `cli.py`, `config.py`, `mcp_client.py`, `quality_gates.py`
Segmentation: `sam2_segmentor.py`, `sam3_segmentor.py`, `sam3d_client.py`, `mask_projector.py`
Mesh: `mesh_extractor.py`, `mesh_cleaner.py`, `texture_baker.py`, `material_assigner.py`
Rendering: `multiview_renderer.py`, `hunyuan3d_client.py`, `comfyui_inpainter.py`
Utilities: `frame_quality.py`, `frame_selector.py`, `coordinate_transform.py`, `colmap_parser.py`, `person_remover.py`

### Web Interface (`src/web/`)

Flask application (port 7860) for video upload, job tracking, and result download. 5 files: `app.py`, `job_manager.py`, `pipeline_runner.py`, `static/`, `templates/`.

### Deployment

| File | Purpose |
|------|---------|
| `Dockerfile.consolidated` | Single container with all services |
| `docker-compose.consolidated.yml` | Compose file for the consolidated container |
| `docker/Dockerfile` | Base container (older, superseded) |
| `docker/docker-compose.yml` | Base compose (older, superseded) |
| `docker/entrypoint.sh` | Container entry script |
| `docker/supervisord.conf` | Process manager configuration |
| `docker/run_docker.sh` | Launch helper |

**The default deployment story is:** `docker compose -f docker-compose.consolidated.yml up -d`. One container, one command, all services. The older `docker/` files exist for reference but the consolidated approach is canonical.

### Scripts (`scripts/`)

Pipeline runners, test harnesses, and utilities:
- `run_gallery_pipeline.py` -- Full gallery pipeline
- `run_object_separation.py` -- Object extraction (33 objects, 98.3% coverage)
- `run_tsdf_mesh.py` -- TSDF mesh extraction
- `assemble_gallery_usd.py` -- USD scene assembly
- `lichtfeld_mcp_bridge.py` -- stdio MCP bridge for Claude Desktop/Codex
- `hardware_trace.py` -- GPU/RAM/CPU logging
- `test_*.py` -- Test harnesses for individual pipeline stages

### Research (`research/`)

15 documents covering landscape analysis, pipeline design, component integration, and architecture decisions. This is research context supporting development decisions. It is not product documentation and should not be treated as user-facing.

### Documentation (`docs/architecture/`, `docs/integration/`, `docs/workflows/`, `docs/troubleshooting/`)

Architecture overviews, integration guides, and workflow documentation written for the gaussian-toolkit branch.

### Other Our Files

| File | Purpose |
|------|---------|
| `GAUSSIAN_TOOLKIT_README.md` | Authoritative README for the gaussian-toolkit branch |
| `BOUNDARIES.md` | This file |
| `AGENTS.md` | Agent operating guide for MCP-driven workflows |

## Experimental

These components are built but not yet validated end-to-end. They may change significantly or be removed.

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| SAM3 concept segmentation | `src/pipeline/sam3_segmentor.py`, `sam3d_client.py` | Client built, Docker integration pending | Replacing SAM2 grid-point prompts with text+visual concept prompts (4M concepts) |
| Hunyuan3D 2.0 mesh creation | `src/pipeline/hunyuan3d_client.py` | Client built, per-object workflow untested at scale | Multi-view renders to textured mesh |
| Texture baking | `src/pipeline/texture_baker.py` | Skeleton written | Depends on clean mesh extraction + xatlas |
| Material assignment | `src/pipeline/material_assigner.py` | Skeleton written | Depends on texture baking |
| FLUX background inpainting | `src/pipeline/comfyui_inpainter.py` | Client built, tested on single scenes | ComfyUI workflow dependency |
| Audio-to-scene-graph naming | Planned (not yet started) | Planned | Extract audio from video, transcribe with Whisper, use transcript to name objects in the USD scene graph |

## Decision Framework

When deciding where new code goes:

1. **Does it modify how LichtFeld trains, renders, or exports Gaussians?** -- Propose it upstream. Do not put it on `gaussian-toolkit`.
2. **Does it extend the video-to-scene pipeline?** -- Put it in `src/pipeline/`.
3. **Does it add a web endpoint or UI page?** -- Put it in `src/web/`.
4. **Does it change container configuration?** -- Put it in `docker/` or update `Dockerfile.consolidated`.
5. **Is it a research exploration or literature review?** -- Put it in `research/`.
6. **Is it a one-off script or test harness?** -- Put it in `scripts/`.
