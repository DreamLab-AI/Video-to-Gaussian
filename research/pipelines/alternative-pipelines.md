# Alternative Pipeline Approaches

## Pipeline A: Segment-First (Image-Space Decomposition)

```mermaid
graph TB
    V[Video] --> F[Frames]
    F --> SAM[SAM2 Object Segmentation<br/>Per-frame instance masks]
    SAM --> INP[Inpaint Objects Out<br/>FLUX/SDXL per frame]
    INP --> COL1[COLMAP Background<br/>Clean environment]
    COL1 --> GS1[Train Background Gaussian]

    SAM --> OBJ[Per-Object Frame Crops]
    OBJ --> TRIPO[TripoSR / InstantMesh<br/>Per-object 3D mesh]

    GS1 --> USD[USD Assembly]
    TRIPO --> USD
```

**Pros**: Clean separation between environment and objects. No ghosting artefacts.
**Cons**: Inpainting quality is critical. Object poses must be recovered separately (no COLMAP for individual objects). TripoSR quality is lower than SuGaR from multi-view Gaussians.
**Verdict**: Viable but lower quality than reconstruct-then-segment. Good fallback for scenes with very few objects.

## Pipeline B: Full Scene Mesh + Post-Hoc Segmentation

```mermaid
graph TB
    V[Video] --> COL[COLMAP]
    COL --> GS[Train 3DGS]
    GS --> MESH[Full Scene Mesh<br/>SuGaR / SOF]
    MESH --> SEG[Mesh Segmentation<br/>SAM3D / PointNet++]
    SEG --> SPLIT[Split Mesh by Labels]
    SPLIT --> USD[USD Assembly]
```

**Pros**: Simplest pipeline. No per-object Gaussian manipulation needed.
**Cons**: Mesh segmentation is harder than Gaussian segmentation (boundaries are less clean). Loss of Gaussian view-dependent appearance. No dual representation (Gaussian + Mesh) per object.
**Verdict**: The hardest path. Polygonal segmentation is a solved problem but results are noisier than Gaussian-space segmentation.

## Pipeline C: Multi-View + Single-Image-to-3D Hybrid

```mermaid
graph TB
    V[Video] --> COL[COLMAP + 3DGS Training]
    COL --> REN[Render Best Views<br/>Per detected object]
    REN --> MV[Multi-View Generation<br/>Zero123++ / SV3D]
    MV --> IM[InstantMesh<br/>Per-object mesh]

    COL --> BG[Background Gaussian<br/>Object-masked training]

    IM --> USD[USD Assembly]
    BG --> USD
```

**Pros**: Leverages Gaussian renders as high-quality input for single-image-to-3D. Multi-view generation fills occluded regions.
**Cons**: Quality depends on generated views (hallucination risk). Extra compute for view generation.
**Verdict**: Good for partially occluded objects where direct Gaussian extraction produces incomplete geometry. Complement to main pipeline.

## Pipeline D: Direct MeshSplatting (No Intermediate Gaussians)

```mermaid
graph TB
    V[Video] --> COL[COLMAP]
    COL --> MS[MeshSplatting Training<br/>+ SAM2 supervision]
    MS --> MESH[Per-Object Meshes<br/>Directly from training]
    MESH --> USD[USD Assembly]
```

**Pros**: Single training step produces segmented meshes directly. No Gaussian-to-mesh conversion needed.
**Cons**: MeshSplatting is newer (2025), less tested. Produces PLY with vertex colours (no UV maps). Requires SAM2 during training.
**Verdict**: Most elegant long-term solution. Monitor MeshSplatting maturity. Currently too new for production pipeline.

## Pipeline E: SLAM-Based Real-Time Decomposition

```mermaid
graph TB
    V[Video Stream] --> SLAM[MonoGS SLAM<br/>Real-time Gaussian mapping]
    SLAM --> SEG[Online SAM2 Segmentation<br/>Per-frame masks → Gaussian labels]
    SEG --> ACC[Accumulate Object Gaussians<br/>Streaming decomposition]
    ACC --> MESH[Batch Mesh Extraction<br/>After capture complete]
    MESH --> USD[USD Assembly]
```

**Pros**: No offline COLMAP step. Real-time capable. Progressive reconstruction.
**Cons**: SLAM quality < offline SfM. Real-time SAM2 is compute-intensive. No pose refinement.
**Verdict**: Future direction for live capture workflows. Not suitable for production quality.

## Comparison

| Pipeline | Quality | Speed | Automation | Maturity | Our Pick |
|----------|---------|-------|------------|----------|----------|
| **Proposed (Hybrid)** | High | Medium | Full | Medium | **Primary** |
| A: Segment-First | Medium | Fast | Full | High | Fallback |
| B: Mesh Segmentation | Low-Medium | Fast | Full | High | Avoid |
| C: Multi-View Hybrid | Medium-High | Slow | Full | Medium | Supplement |
| D: Direct MeshSplat | High | Fast | Full | Low | Future |
| E: SLAM Real-Time | Low-Medium | Real-time | Full | Low | Future |
