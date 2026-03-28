# Research: Video to Structured 3D USD Scene

## Objective

Reconstruct 3D polygonal scenes in USD format from video, via Gaussian Splatting. Identify all objects in the scene, isolate them, reconstruct each as individual textured 3D meshes with metadata, and assemble into a hierarchical scene graph.

## Target Pipeline

```
Video → Frames → COLMAP SfM → 3DGS Training → Object Segmentation
    → Per-Object Mesh Extraction → Background Recovery → USD Scene Assembly
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
| Video → Frames | Exists | SplatReady |
| COLMAP SfM | Exists | COLMAP 4.1.0 |
| 3DGS Training | Exists | LichtFeld Studio |
| Object Segmentation | **New** | Gaussian Grouping |
| Mesh Extraction | **New** | SuGaR / SOF |
| Background Inpainting | **New** | ComfyUI + FLUX |
| USD Assembly | **Extend** | OpenUSD Python |
| Agentic Orchestration | Exists | LichtFeld MCP (70+ tools) |
