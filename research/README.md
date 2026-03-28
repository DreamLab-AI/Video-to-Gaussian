# Research: Video to Structured 3D USD Scene

## Objective

Reconstruct 3D polygonal scenes in USD format from video, via Gaussian Splatting. Identify all objects in the scene, isolate them, reconstruct each as individual textured 3D meshes with metadata, and assemble into a hierarchical scene graph.

## Pipeline Status

### Complete

| Component | Module | Status |
|-----------|--------|--------|
| Video frame extraction | `orchestrator.py` (PyAV) | Tested, working |
| COLMAP SfM (feature extract, match, sparse, undistort) | `orchestrator.py` + COLMAP 4.1.0 | Tested, working |
| 3DGS Training via LichtFeld MCP | `mcp_client.py` | Tested, 7k iter in 2m15s |
| SAM2 2D segmentation | `sam2_segmentor.py` | Tested, 13 frames in 46s |
| Mask projection (2D masks to 3D Gaussians) | `mask_projector.py` | Tested, per-Gaussian voting |
| Mesh extraction (Marching Cubes) | `mesh_extractor.py` | Tested, basic extraction |
| Mesh cleaning (decimation, hole fill) | `mesh_cleaner.py` | Tested |
| USD scene assembly | `usd_assembler.py` | Tested, variant sets |
| Quality gates (per-stage pass/fail) | `quality_gates.py` | Tested |
| CLI entry point | `cli.py` + `__main__.py` | Working |
| Pipeline configuration | `config.py` | YAML/dict based |
| Coordinate transforms | `coordinate_transform.py` | COLMAP <-> 3DGS <-> USD |
| Frame quality scoring | `frame_quality.py` | Blur/exposure filtering |
| MCP bridge script | `scripts/lichtfeld_mcp_bridge.py` | Working |
| Hardware tracing | `scripts/hardware_trace.py` | GPU/RAM/CPU logging |

### In Progress

| Component | Module | Status | Blocker |
|-----------|--------|--------|---------|
| Mask projection refinement | `mask_projector.py` | Functional but noisy | Needs depth-weighted voting for thin structures |
| TSDF mesh extraction | `mesh_extractor.py` | Marching Cubes only | TSDF fusion requires Open3D integration for watertight meshes |
| ComfyUI inpainting integration | `comfyui_inpainter.py` | Skeleton written | Needs workflow JSON and API endpoint wiring |
| Texture baking | `texture_baker.py` | Skeleton written | Depends on clean mesh extraction |
| Material assignment | `material_assigner.py` | Skeleton written | Depends on texture baking |
| COLMAP output parsing | `colmap_parser.py` | Basic binary reader | Needs robust error handling for malformed models |

### Known Issues

1. **COLMAP sparse reconstruction is the bottleneck** — 15-20 minutes on 32 cores for 15 frames. No GPU acceleration available for the sparse BA solver. Workaround: use fewer frames or switch to incremental mapper.

2. **SAM2 prompt strategy** — Currently using grid-point prompts. Quality depends heavily on prompt placement. Needs automatic foreground detection or user-guided bounding boxes for reliable multi-object segmentation.

3. **Mask projection noise on thin geometry** — The Gaussian-space voting from 2D masks produces noisy labels on thin structures (branches, wires). Depth-weighted voting and multi-view consistency checks are needed.

4. **Mesh extraction produces non-manifold geometry** — Marching Cubes on the Gaussian density field can produce self-intersecting faces. PyMeshFix is available but not yet wired into the pipeline.

5. **ComfyUI inpainting not yet integrated** — The remote ComfyUI-API endpoint (192.168.2.48:3001) is running but the pipeline does not yet call it. Background recovery for removed objects is manual.

6. **No texture UV unwrapping** — Meshes are vertex-coloured only. xatlas is installed but not integrated for proper UV unwrapping and texture baking.

7. **USD variant sets are placeholder** — The assembler creates Gaussian and Mesh variant sets but the Gaussian variant currently stores only a path reference, not embedded splat data.

## Target Pipeline

```
Video -> Frames -> COLMAP SfM -> 3DGS Training -> Object Segmentation
    -> Per-Object Mesh Extraction -> Background Recovery -> USD Scene Assembly
```

## Research Structure

```
research/
├── README.md                          # This file
├── landscape/
│   ├── tool-catalogue.md              # 28 tools assessed with viability scores
│   ├── segmentation-methods.md        # 3D Gaussian segmentation SOTA
│   ├── mesh-extraction-methods.md     # Gaussian-to-mesh conversion SOTA
│   └── field-overview.md              # Landscape synthesis and gap analysis
├── pipelines/
│   ├── proposed-pipeline.md           # Recommended end-to-end architecture
│   ├── alternative-pipelines.md       # Alternative approaches considered
│   └── pipeline-comparison.md         # Comparative analysis
├── components/
│   ├── gaussian-grouping.md           # Primary segmentation method
│   ├── sugar-mesh-extraction.md       # Primary mesh extraction method
│   ├── sof-mesh-extraction.md         # Alternative: SOF Marching Tetrahedra
│   ├── inpainting-recovery.md         # Background recovery via diffusion
│   ├── usd-assembly.md               # Scene graph composition
│   └── quality-control.md            # Agent quality decision trees
├── references/
│   ├── papers.md                      # Academic references
│   ├── repos.md                       # GitHub repositories
│   └── existing-capabilities.md       # What LichtFeld/COLMAP already provide
└── decisions/
    ├── prd.md                         # Product Requirements Document
    ├── adr-001-pipeline-architecture.md
    └── ddd-domain-model.md
```

## Key Findings

### Critical Path

The minimum viable pipeline requires exactly 4 components:

1. **SplatReady** (installed) — Video to COLMAP dataset
2. **Gaussian Grouping** (Apache-2.0) — Per-object Gaussian labelling during training
3. **SuGaR** (CVPR 2024, 3.3k stars) — Per-object textured mesh extraction
4. **USD Assembly Script** (new) — OpenUSD scene graph composition

### Segment-First vs Reconstruct-Then-Segment

Evidence strongly favours **reconstruct-then-segment** for our use case:
- Gaussian Grouping and SAGA both operate on trained 3DGS models
- Post-hoc segmentation allows quality gating before decomposition
- Pre-segmentation (inpainting objects out of training images) is viable for background recovery but not for initial decomposition

### Hybrid Approach (Recommended)

1. Train full scene with Gaussian Grouping (joint reconstruction + segmentation)
2. Extract per-object Gaussians using learned identity encodings
3. Run SuGaR per-object for textured mesh extraction
4. Inpaint removed objects from training views via ComfyUI/FLUX
5. Retrain clean background Gaussian
6. Assemble multi-object USD scene with variant sets (Gaussian + Mesh per object)

### Gap Analysis

| Capability | Status | Primary Tool |
|-----------|--------|--------------|
| Video to Frames | Complete | SplatReady / PyAV |
| COLMAP SfM | Complete | COLMAP 4.1.0 |
| 3DGS Training | Complete | LichtFeld Studio MCP |
| Object Segmentation | Complete (basic) | SAM2 + mask projection |
| Mesh Extraction | Complete (basic) | Marching Cubes + Trimesh |
| Background Inpainting | **In Progress** | ComfyUI + FLUX |
| Texture Baking | **In Progress** | xatlas + custom baker |
| USD Assembly | Complete (basic) | OpenUSD Python |
| Agentic Orchestration | Complete | LichtFeld MCP (70+ tools) |
