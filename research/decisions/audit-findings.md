# Pipeline Audit Findings

**Date**: 2026-03-30
**Auditor**: Code Review Agent (Opus 4.6)
**Scope**: Full end-to-end audit of `src/pipeline/` in Docker container environment
**Python**: 3.14.3 (system default at `/usr/bin/python3`)
**Torch**: 2.10.0+cu128, CUDA available
**gsplat**: installed at `~/.local/lib/python3.14/site-packages/gsplat/`

---

## Executive Summary

All pipeline module imports succeed. gsplat and torch ARE available on the correct Python 3.14 path. The core problems are:

1. **COLMAP sequential matcher uses default overlap=10** -- far too low for walkthrough video. Registration rate tanks.
2. **Mesh stage gsplat path succeeds** in import but the fallback chain silently degrades to convex hull when any exception occurs (818 vertices = convex hull signature).
3. **No depth map or multi-view preview save stage exists** in the pipeline. The multiview_renderer `_save_views` only saves RGBA PNGs, not depth colormaps. The CLI orchestrator never calls multiview_renderer at all.
4. **Segmentation falls back to full_scene** (single object) because SAM3 BPE vocab is at wrong path and the fallback chain treats this as acceptable.
5. **LichtFeld binary config default is wrong** (`/opt/gaussian-toolkit/build/LichtFeld-Studio` does not exist; actual is at `/home/devuser/workspace/gaussians/LichtFeld-Studio/build/LichtFeld-Studio`).

---

## Finding 1: COLMAP Sequential Matcher -- Low Registration Rate (13-36%)

### Root Cause
**File**: `src/pipeline/stages.py`, lines 520-523
```python
subprocess.run([
    colmap, self.config.reconstruct.matcher + "_matcher",
    "--database_path", str(db_path),
], check=True, capture_output=True, timeout=300)
```

The sequential matcher is called with **zero additional parameters**. COLMAP defaults:
- `--SequentialMatching.overlap 10` (matches each frame only against the next 10 frames)
- `--SequentialMatching.loop_detection 0` (disabled)
- `--SequentialMatching.quadratic_overlap 1`

For a walkthrough video with slow camera motion, overlap=10 is pathetically low. Frame N may have significant visual overlap with frame N+30 or beyond, but COLMAP never checks.

Additionally, the feature extractor (lines 513-518) passes no `--SiftExtraction.max_num_features` override. COLMAP 4.1 defaults to 8192 features, which is fine, but `--ImageReader.single_camera 1` forces all frames to share one intrinsic model. This is correct for video but limits per-frame refinement.

### Impact
- Only 13-36% of frames register into the sparse model
- The 3D reconstruction is sparse, covering only a fraction of the scene
- Downstream training produces a PLY with poor coverage

### Fix Required
```python
subprocess.run([
    colmap, "sequential_matcher",
    "--database_path", str(db_path),
    "--SequentialMatching.overlap", "30",
    "--SequentialMatching.loop_detection", "1",
    "--SequentialMatching.loop_detection_period", "10",
    "--SequentialMatching.loop_detection_num_images", "50",
], check=True, capture_output=True, timeout=300)
```

Also add to `ReconstructConfig`:
- `sequential_overlap: int = 30`
- `loop_detection: bool = True`

---

## Finding 2: Mesh Stage Silently Falls to Convex Hull (818 vertices)

### Root Cause
**File**: `src/pipeline/stages.py`, lines 963-1108 (`_mesh_single`)

The method tries 6 strategies in order:
0. gsplat depth -> TSDF (preferred)
1. Hunyuan3D
2. TSDF from point cloud
3. Point cloud marching cubes
4. Open3D Poisson
5. Convex hull

**Every strategy** is wrapped in `try/except Exception` that logs a warning and falls through. The convex hull at strategy 5 always succeeds and produces ~818 vertices (the convex hull of the point cloud).

The gsplat path (strategy 0) at line 964-1014 does:
```python
try:
    import torch
    if torch.cuda.is_available():
        from pipeline.mesh_extractor import MeshExtractor, TSDFConfig
        extractor = MeshExtractor(config=TSDFConfig(
            target_faces=self.config.mesh.max_vertices // 2,
        ))
        mesh, color_images, cameras = extractor.extract_from_gsplat(
            ply_path,
            num_views=64,
            render_size=1024,
            target_faces=self.config.mesh.max_vertices // 2,
        )
```

This **should** work. torch and gsplat both import correctly. But if `extract_from_gsplat` raises ANY exception (OOM, invalid PLY format, TSDF volume too large, marching cubes level-set failure), the code silently falls through ALL the way to convex hull.

The most likely failure mode: the `TSDFConfig` at line 969 uses **default bounds** `[-1, -1, -1]` to `[1, 1, 1]` because it's constructed with only `target_faces` specified. However, `extract_from_gsplat` (mesh_extractor.py line 660-680) computes its own adaptive bounds from the gaussian positions, so this default doesn't matter for that path.

The actual failure is likely a CUDA OOM or a marching cubes failure. With 64 views at 1024x1024, the TSDF volume can be huge. The adaptive voxel sizing (line 673) targets ~300 voxels per axis max, but for a large scene this can still be 27M+ voxels with color channels = ~400MB+.

**Critical**: There is NO logging of which strategy succeeded. Claude Code sees `"method": "convex_hull"` in the result but the orchestrator (Claude Code itself) doesn't check this and proceeds.

### Impact
- 818 vertex convex hull instead of 30K+ textured mesh
- No texture data (convex hull has no UVs or colors)
- All downstream stages (texture bake, USD assembly) get garbage input

### Fix Required
1. Add explicit OOM handling in gsplat path with reduced resolution fallback
2. After all strategies, check vertex count against `config.mesh.min_vertices` (currently 100, should be higher)
3. Return `StageResult(success=False)` if only convex_hull succeeds AND vertex_count < min_vertices threshold
4. Log WHICH strategy succeeded/failed with timing
5. Add a `--mesh-method` override to force a specific strategy

---

## Finding 3: No Depth Map or Multi-View Preview Generation

### Root Cause
**File**: `src/pipeline/cli.py`, lines 84-224 and `src/pipeline/stages.py`

The pipeline stages are:
1. ingest -> 2. remove_people -> 3. select_frames -> 4. reconstruct -> 5. train -> 6. segment -> 7. extract_objects -> 8. mesh_objects -> 9. texture_bake -> 10. assemble_usd -> 11. validate

**There is no stage that generates depth map previews or viewpoint renders.**

The `MultiViewRenderer` class in `multiview_renderer.py` exists and works correctly (CPU-based gaussian splatting with SH evaluation, proper alpha compositing). Its `_save_views` method (line 843-858) saves RGBA PNGs but **not depth maps**:

```python
def _save_views(self, views: list[ViewResult], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for view in views:
        name = view.camera.name
        path = output_dir / f"{name}.png"
        img = PILImage.fromarray(view.image, mode="RGBA")
        img.save(path)
        saved.append(path)
    return saved
```

Each `ViewResult` contains `.depth` (float32 HxW depth map) but it is never saved.

The `MeshExtractor.extract_from_gsplat` does render depth+color from 64 views, but those are only used for TSDF integration and never saved as preview images.

### Impact
- No depth map colormaps for debugging or visualization
- No viewpoint-matched RGB renders for quality assessment
- Claude Code cannot verify reconstruction quality visually
- Only 1 preview image is generated (likely a single thumbnail, not from this pipeline)

### Fix Required
1. Add a `render_previews` stage to `PipelineStages` that:
   - Loads the trained PLY
   - Uses `MultiViewRenderer` with 8+ views (canonical_4 + elevated)
   - Saves RGBA renders as PNG
   - Saves depth maps as colorized PNG (matplotlib/viridis colormap)
   - Returns paths in `StageResult.artifacts`
2. Add depth saving to `MultiViewRenderer._save_views`:
   ```python
   depth_path = output_dir / f"{name}_depth.png"
   depth_norm = (view.depth - view.depth[view.depth > 0].min()) / (view.depth.max() - view.depth[view.depth > 0].min() + 1e-8)
   depth_colored = plt.cm.viridis(depth_norm)[:, :, :3]
   PILImage.fromarray((depth_colored * 255).astype(np.uint8)).save(depth_path)
   ```
3. Wire the new stage into `cli.py` between train and segment (or after mesh_objects)

---

## Finding 4: SAM3 Segmentation Falls Back to Full-Scene

### Root Cause
**File**: `src/pipeline/sam3_segmentor.py`, lines 38-46

```python
_SAM3_BPE_PATH = os.environ.get(
    "SAM3_BPE_PATH",
    "/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
)
```

The BPE vocab is hardcoded to `/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz`. In this container, the actual path is:
- `/home/devuser/workspace/gaussians/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz`
- `/home/devuser/.local/lib/python3.14/site-packages/assets/bpe_simple_vocab_16e6.txt.gz`

The file at the hardcoded path does NOT exist.

**File**: `src/pipeline/stages.py`, lines 666-737 (`segment`)

When SAM3 fails (which it will, due to the BPE path issue or any model loading error), the fallback chain at line 724-737 is:
```python
except Exception as exc:
    if not decompose_cfg.sam3_fallback_to_sam2:
        return StageResult(success=False, ...)
    logger.warning("SAM3 failed (%s), falling back to full-scene", exc)

# Fallback: treat entire scene as one object
return StageResult(
    success=True, stage="segment",
    metrics={"object_count": 1, "method": "full_scene"},
    artifacts={"objects": json.dumps([{"label": "full_scene", "count": -1}])},
)
```

Note: `sam3_fallback_to_sam2` is `True` by default (config.py line 63), but the fallback doesn't actually USE SAM2. It just returns "full_scene" as a single object. The variable name is misleading.

**Also in `DecomposeConfig`** (config.py lines 64-65):
```python
sam3_bpe_path: str = "/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
sam3_checkpoint_path: str = ""  # auto-detect from HF cache
```
The `sam3_bpe_path` config field exists but is NEVER USED by `sam3_segmentor.py`. The segmentor reads from `os.environ` / hardcoded default, ignoring the config.

### Impact
- SAM3 text-prompted segmentation always fails silently
- Pipeline treats entire scene as one object ("full_scene")
- No per-object decomposition occurs
- extract_objects just copies the full PLY without segmentation

### Fix Required
1. Fix BPE path in `sam3_segmentor.py`:
   ```python
   _SAM3_BPE_PATH = os.environ.get(
       "SAM3_BPE_PATH",
       "/home/devuser/workspace/gaussians/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
   )
   ```
2. Or better: read from `PipelineConfig.decompose.sam3_bpe_path` (pass config into SAM3Segmentor)
3. Fix the misleading fallback: if `sam3_fallback_to_sam2` is True, actually try SAM2 automatic mask generation instead of returning full_scene
4. When full_scene fallback occurs, log it as a WARNING with the original exception

---

## Finding 5: LichtFeld Binary Path Wrong in Default Config

### Root Cause
**File**: `src/pipeline/config.py`, line 43

```python
lichtfeld_binary: str = "/opt/gaussian-toolkit/build/LichtFeld-Studio"
```

This path does not exist. The actual binary is at:
- `/home/devuser/workspace/gaussians/LichtFeld-Studio/build/LichtFeld-Studio`
- Symlinked from `/usr/local/bin/lichtfeld-studio`

**File**: `src/pipeline/stages.py`, lines 568-576

The train stage has a fallback chain:
```python
lfs_binary = self.config.training.lichtfeld_binary
if not Path(lfs_binary).exists():
    for candidate in [
        "/opt/gaussian-toolkit/build/LichtFeld-Studio",
        "/usr/local/bin/lichtfeld-studio",
        str(Path.home() / "workspace/gaussians/LichtFeld-Studio/build/LichtFeld-Studio"),
    ]:
        if Path(candidate).exists():
            lfs_binary = candidate
            break
```

The fallback chain DOES find it at `/usr/local/bin/lichtfeld-studio` (second candidate). So training works, but only after checking two non-existent paths first.

### Impact
- Minor: training works via fallback, but the default config is misleading
- If someone supplies a config JSON with the default path, it will silently fall back

### Fix Required
Update `config.py` line 43:
```python
lichtfeld_binary: str = "/usr/local/bin/lichtfeld-studio"
```

---

## Finding 6: $PYTHONPATH Literal String in sys.path

### Root Cause
The shell environment has `PYTHONPATH=/usr/lib/python3.14/site-packages:$PYTHONPATH` where `$PYTHONPATH` was not expanded, resulting in a literal `$PYTHONPATH` directory in `sys.path`:

```
/home/devuser/workspace/gaussians/LichtFeld-Studio/$PYTHONPATH
```

This is a shell configuration bug (likely in `.bashrc` or `.zshrc` where `PYTHONPATH` was set using single quotes instead of double quotes).

### Impact
- Cosmetic: Python tries to look up modules in a nonexistent directory on every import
- Could mask real import failures with confusing error messages

### Fix Required
Fix the shell config to properly expand:
```bash
export PYTHONPATH="/usr/lib/python3.14/site-packages:${PYTHONPATH:-}"
```

---

## Finding 7: open3d Not Installed (Strategy 4 Fallback Dead)

### Root Cause
`open3d` is not installed in the Python 3.14 environment. The `_mesh_with_open3d` function at `stages.py` line 157-165 will always fail with `ModuleNotFoundError`.

### Impact
- Strategy 4 (Open3D Poisson reconstruction) in `_mesh_single` is dead code
- Reduces the fallback chain from 6 strategies to 5
- Makes convex hull more likely to be the final fallback

### Fix Required
Either install open3d: `pip install open3d` (if available for Python 3.14) or remove the dead strategy and add a better intermediate fallback.

---

## Finding 8: COLMAP Reconstruction Returns Misleading Metrics

### Root Cause
**File**: `src/pipeline/stages.py`, lines 493-502

```python
cameras = len(list(sparse_dir.glob("cameras.*"))) if sparse_dir.exists() else 0
```

This counts the number of **camera model files** (typically 1 for single_camera mode), NOT the number of registered images. The registration rate (fraction of input images that COLMAP successfully placed) is never computed or returned.

To get the actual registration rate, you need to parse `images.bin` or `images.txt` in the sparse model and compare against the input frame count.

### Impact
- Claude Code has no way to detect low registration rates
- 13% registration goes unnoticed; pipeline proceeds with garbage sparse model
- No quality gate on reconstruction quality

### Fix Required
Add registration rate computation:
```python
images_bin = sparse_dir / "images.bin"
if images_bin.exists():
    # Parse binary format to count registered images
    registered = count_images_in_bin(images_bin)
    total = len(list(frames_path.glob("*.jpg")))
    rate = registered / total if total > 0 else 0
    if rate < 0.5:
        return StageResult(success=False, stage="reconstruct",
            error=f"Registration rate too low: {rate:.0%} ({registered}/{total})")
```

---

## Finding 9: Segment Stage Does Not Use All Frames

### Root Cause
**File**: `src/pipeline/stages.py`, lines 676-683

```python
image = cv2.imread(str(frame_paths[0]))
```

SAM3 segmentation only processes the FIRST frame. For video-based segmentation with consistent object tracking, it should process multiple keyframes and propagate masks.

### Impact
- Segmentation quality depends entirely on the first frame
- If the first frame has occlusions or poor viewpoint, objects are missed
- Video propagation features of SAM3 are completely unused

### Fix Required
Use multiple keyframes (e.g., 3-5 evenly spaced frames) and aggregate/vote on object masks.

---

## Finding 10: Python 3.11 Has No usd-core Installed

### Root Cause
`python3.11` exists at `/usr/bin/python3.11` but `usd-core` is not installed:
```
WARNING: Package(s) not found: usd-core
```

The `_find_usd_python` function in `stages.py` (lines 194-218) checks three candidates:
1. `/opt/venv-usd/bin/python3` -- does not exist
2. `~/.lichtfeld/.../.venv-usd/bin/python3` -- does not exist
3. `/usr/bin/python3.11` -- exists but no usd-core

All three fail, so `assemble_usd` always falls back to the minimal USDA stub at line 1253-1256.

### Impact
- USD scene assembly never uses the full assembler script
- Output is always a minimal USDA stub with just mesh path references
- No proper USD stage with cameras, materials, or transforms

### Fix Required
```bash
python3.11 -m pip install usd-core
```

---

## Consolidated Fix List

### Critical (must fix for acceptable output)

| # | Module | Line(s) | Fix | Expected Improvement |
|---|--------|---------|-----|---------------------|
| 1 | stages.py | 520-523 | Add `--SequentialMatching.overlap 30`, `--loop_detection 1` to COLMAP | Registration 13% -> 60%+ |
| 2 | stages.py | 963-1108 | After mesh_objects, check vertex_count against min threshold (5000). Return success=False if convex_hull AND < 5000 verts. | Forces retry or explicit failure instead of 818-vert garbage |
| 3 | stages.py (new) | -- | Add `render_previews()` stage using MultiViewRenderer + depth colormap save | 8+ RGB + 8+ depth previews |
| 4 | sam3_segmentor.py | 39-41 | Fix BPE path to `/home/devuser/workspace/gaussians/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz` | Per-object segmentation works |
| 5 | config.py | 43 | Change default to `/usr/local/bin/lichtfeld-studio` | Correct binary on first try |

### High (should fix for reliable operation)

| # | Module | Line(s) | Fix | Expected Improvement |
|---|--------|---------|-----|---------------------|
| 6 | stages.py | 493-502 | Compute actual registration rate from images.bin, fail if < 50% | Early detection of reconstruction failure |
| 7 | stages.py | 730-737 | Actually try SAM2 auto-mask when SAM3 fails, before full_scene fallback | Better segmentation coverage |
| 8 | multiview_renderer.py | 843-858 | Save depth colormap alongside RGBA PNG in `_save_views` | Depth visualization |
| 9 | stages.py | 676-683 | Segment multiple keyframes, not just frame[0] | More robust object detection |
| 10 | system | -- | `python3.11 -m pip install usd-core` | Proper USD assembly |

### Low (nice to have)

| # | Module | Line(s) | Fix | Expected Improvement |
|---|--------|---------|-----|---------------------|
| 11 | stages.py | 157-165 | Remove _mesh_with_open3d or install open3d | Clean dead code |
| 12 | system | .bashrc/.zshrc | Fix `$PYTHONPATH` literal expansion | Clean sys.path |
| 13 | stages.py | 952-958 | Log which meshing strategy was used with timing | Debug visibility |
| 14 | config.py | 63-64 | Wire sam3_bpe_path config into SAM3Segmentor | Config-driven BPE path |

---

## Target Output Verification

After applying the Critical fixes, the pipeline should produce:

| Output | Current | Target | Gated By |
|--------|---------|--------|----------|
| COLMAP registration | 13-36% | >50% | Fix #1 (overlap+loop detection) |
| Mesh vertices | 818 (convex hull) | 30K+ (TSDF) | Fix #2 (threshold gate) + Fix #1 (better reconstruction) |
| Viewpoint renders | 0-1 | 8+ RGBA PNGs | Fix #3 (render_previews stage) |
| Depth maps | 0 | 8+ colorized PNGs | Fix #3 + Fix #8 |
| Per-object segments | 1 (full_scene) | 5+ objects | Fix #4 (SAM3 BPE path) |
| USD scene | minimal stub | full assembly | Fix #10 (usd-core install) |
