# Existing Capabilities: What We Already Have

## LichtFeld Studio (v0.4.2)

### Reconstruction
- COLMAP 4.1.0 (CUDA-accelerated, headless)
- SplatReady plugin (video → COLMAP pipeline)
- Training strategies: MCMC, MRNF, IGS+
- Pose optimisation: direct + MLP modes
- PPISP correction (exposure, vignetting, colour, CRF)
- SH degrees 0-3

### MCP Server (70+ tools)
- Training: load_dataset, start, get_state, loss_history, ask_advisor
- Camera: get, set_view, reset, list, go_to_dataset_camera
- Render: capture (base64 PNG), settings.get/set
- Selection: rect, polygon, lasso, ring, brush, click, by_description (LLM)
- Export: PLY, SOG, SPZ, USD, HTML (async with status/cancel)
- Scene graph: list_nodes, select, visibility, rename, reparent, add_group, duplicate, merge
- History: undo/redo with transactions
- Gaussians: read/write GPU tensor data
- Editor: Python code execution
- Events: pub/sub system
- Plugins: invoke/list

### File Format Support
- Import: PLY, SOG, SPZ, USD/USDA/USDC/USDZ, OBJ, FBX, glTF, GLB, STL, DAE
- Export: PLY, SOG, SPZ, USD, HTML
- Checkpoint: .resume format

### Scene Graph
- Node types: SPLAT, MESH, GROUP, POINTCLOUD
- Hierarchical tree with visibility, locking, renaming
- Duplication, merging, reparenting

### mesh2splat (EA, BSD-3)
- Mesh → Gaussian conversion
- PBR texture support (diffuse, metallic-roughness, normal)
- Integrated in rendering pipeline

## COLMAP (4.1.0)

- Feature extraction (SIFT, GPU-accelerated)
- Exhaustive/sequential/spatial matching
- Sparse mapper
- Model alignment
- Image undistortion
- Dense reconstruction (CUDA)
- Format export (TXT, BIN)

## ComfyUI Docker

- FLUX, SDXL, SD 1.5 models
- Inpainting workflows
- LoRA loading
- ControlNet
- API access via HTTP
- Custom node ecosystem

## Blender (5.0.1)

- MCP socket server
- Python scripting
- USD import/export
- Mesh operations
- Material/shader assignment
- Rendering (Cycles, EEVEE)
- Modifier stack

## CLI Tools

- `lichtfeld-studio` — Full application
- `lfs-mcp` — MCP HTTP wrapper
- `video2splat` — Full video-to-training pipeline
- `colmap` — SfM toolkit

## Python Environment

- PyAV 17.0.0 (video frame extraction)
- Pillow (image processing)
- piexif (EXIF GPS embedding)
- OpenCV available via pacman

## Docker Phase 2.7

All tools pre-built in container image with CUDA support. Reproducible builds.

## RuVector Memory

3DGS stack knowledge stored:
- 13 memory entries (tools, APIs, workflows)
- 10 patterns (workflow, build, troubleshooting)
- 13-node, 17-edge knowledge graph
