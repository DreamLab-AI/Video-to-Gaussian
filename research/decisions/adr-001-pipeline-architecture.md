# ADR-001: Pipeline Architecture for Video-to-Scene Reconstruction

## Status

Proposed

## Context

We need to convert video recordings into structured USD scenes with individually addressable 3D mesh objects. The field offers multiple approaches:

1. **Segment-first**: Segment objects in 2D frames, inpaint backgrounds, reconstruct separately
2. **Reconstruct-then-segment**: Build monolithic Gaussian scene, segment in 3D, extract per-object
3. **Co-training**: Joint reconstruction and segmentation during training
4. **Direct meshing**: Skip Gaussians entirely, train meshes from images (MeshSplatting)
5. **SLAM-based**: Real-time simultaneous reconstruction and decomposition

We have an existing stack: LichtFeld Studio (3DGS training, 70+ MCP tools, USD export), COLMAP (SfM), SplatReady (video-to-COLMAP), ComfyUI (diffusion models), Blender (3D editing).

## Decision

**We adopt a hybrid co-training + post-hoc approach** using Gaussian Grouping for joint reconstruction and segmentation, SuGaR for mesh extraction, and ComfyUI FLUX for background inpainting.

The pipeline is:

```
Video → SplatReady → COLMAP → Gaussian Grouping → Per-Object Extraction
    → SuGaR Meshing → Background Inpainting → Background Retrain → USD Assembly
```

Orchestrated by an agent swarm communicating via LichtFeld's MCP server.

## Rationale

### Why co-training (Gaussian Grouping) over post-hoc (SAGA)?

- Co-training produces cleaner object boundaries because the segmentation regulariser influences Gaussian placement during optimisation
- Post-hoc methods inherit any boundary ambiguity from the original training
- Gaussian Grouping (Apache-2.0) is production-safe and the most mature co-training approach
- SAGA serves as a refinement tool for difficult cases, not the primary path

### Why SuGaR over SOF/GOF for meshing?

- SuGaR is the only method producing UV-mapped textured meshes (OBJ + MTL + diffuse PNG)
- SOF/GOF produce vertex-coloured geometry requiring separate texture baking
- SuGaR's output is directly compatible with USD `UsdPreviewSurface` materials
- SOF is retained as a fallback for scenes where geometric accuracy matters more than textures

### Why background inpainting via ComfyUI?

- We already have ComfyUI infrastructure in the Docker stack
- FLUX inpainting produces photorealistic fill
- The Gaussian Splats Repair LoRA is specifically trained for Gaussian render artefacts
- Agent orchestration of ComfyUI workflows is straightforward via HTTP API

### Why not segment-first (Pipeline A)?

- Requires inpainting BEFORE reconstruction, which means any inpainting artefact propagates through the entire pipeline
- Object pose recovery is harder without a full scene reconstruction as reference
- Lower mesh quality from single-image-to-3D vs. multi-view Gaussian extraction

### Why not direct meshing (MeshSplatting)?

- Too new (2025), limited community testing
- Produces PLY with vertex colours (no UV maps)
- Requires specific training setup incompatible with LichtFeld's existing pipeline
- Retained as future direction for Phase 7+

### Why agentic orchestration?

- Each stage has quality decisions that benefit from adaptive control
- The pipeline is not deterministic — different videos require different parameters
- Quality gates prevent poor-quality intermediates from propagating
- Agent retries with adjusted parameters handle edge cases automatically
- LichtFeld's 70+ MCP tools provide complete programmatic control

## Consequences

### Positive
- Clean architecture with well-defined stage boundaries
- All critical-path dependencies are Apache-2.0 licensed
- Leverages existing LichtFeld infrastructure (no new rendering engine)
- Dual representation (Gaussian + Mesh) per object in USD output
- Agentic control enables fully automated processing

### Negative
- Gaussian Grouping requires retraining (cannot reuse existing trained models for segmentation)
- SuGaR license is unspecified (risk for commercial use)
- Background inpainting adds ~15 min per scene + ComfyUI dependency
- Multiple Python dependencies alongside C++ core (mixed runtime)

### Risks
- If Gaussian Grouping quality is insufficient, fallback to SAGA + manual refinement
- If SuGaR mesh quality is insufficient, fallback to SOF + xatlas texture baking
- If ComfyUI is unavailable, fallback to LaMa inpainting (lower quality, no GPU model)

## Alternatives Considered

See [[alternative-pipelines]] for full analysis of 5 alternative approaches.

## Related Decisions

- ADR-002 (pending): Coordinate transform chain specification
- ADR-003 (pending): USD scene graph schema and variant set design
- ADR-004 (pending): Agent communication protocol and state machine design
