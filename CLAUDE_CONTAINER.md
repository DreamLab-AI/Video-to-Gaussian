# Video-to-Scene Pipeline -- Claude Code Orchestration

You are running inside the gaussian-toolkit Docker container on a dual RTX 6000 Ada system.
**You are the orchestrator.** There is no state machine. You run each pipeline stage manually,
inspect results between steps, and decide what to do next.

**CRITICAL: You MUST complete ALL stages through to USD assembly and validation.
Do NOT stop after training. The full pipeline is:
ingest -> select_frames -> reconstruct -> train -> segment -> extract_objects -> mesh_objects -> assemble_usd -> validate**

## Available Tools

- LichtFeld Studio: `/opt/gaussian-toolkit/build/LichtFeld-Studio`
- COLMAP: `/usr/local/bin/colmap`
- Blender: `/usr/local/bin/blender` (DISPLAY=:1, VNC on :5901)
- ComfyUI API: `http://localhost:8188`
- Python pipeline stages: `from pipeline.stages import PipelineStages`
- Web API: `http://localhost:7860`

## When a job arrives

Check for new jobs:

```bash
curl -s http://localhost:7860/jobs | python3 -m json.tool
```

For each queued job, run the pipeline stage by stage.

---

## Step 1: Ingest -- extract frames from video

Extract frames at 4fps to oversample, then select the best subset.

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.ingest('/data/output/JOB_ID/input.mp4', fps=4.0)
print(result)
"
```

Update the web UI:
```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "ingest", "progress": 0.05, "message": "Extracting frames at 4fps"}'
```

**Inspect**: `ls /data/output/JOB_ID/frames/ | wc -l`

---

## Step 2: Remove people (if needed)

Look at the frames. If people are visible:

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.remove_people('/data/output/JOB_ID/frames/')
print(result)
"
```

If no people, skip to step 3 using the frames directory directly.

---

## Step 3: Select best frames (IMPORTANT for COLMAP registration)

Select 60-80 diverse, high-quality frames from the oversampled set.
This is critical -- sending all frames to COLMAP causes low registration rates.

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.select_frames('/data/output/JOB_ID/frames/', target=80)
print(result)
"
```

**Check**: The selected frame count should be 60-80. If less than 40, re-run with lower blur_threshold.

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "select_frames", "progress": 0.1, "message": "Selected N frames from M extracted"}'
```

---

## Step 4: COLMAP reconstruction

Use the **sequential** matcher for video input (NOT exhaustive -- sequential is faster
and produces better registration rates for video where frames are temporally ordered).

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.reconstruct('/data/output/JOB_ID/frames_selected/', matcher='sequential')
print(result)
"
```

**Check**: Look for `sparse/0/cameras.bin` and `images/` in the colmap dir.
**CRITICAL**: Check the registration rate. At least 70% of input frames should register.
If registration is below 50%, re-run with `matcher='exhaustive'` or reduce frame count.

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "reconstruct", "progress": 0.25, "message": "COLMAP: N/M frames registered"}'
```

---

## Step 5: Train gaussian splatting

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.train('/data/output/JOB_ID/colmap/undistorted/', iterations=30000)
print(result)
"
```

Update progress during training:
```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "train", "progress": 0.5, "message": "30k iter, loss 0.02"}'
```

**Quality check**: The PLY should be > 10 MB for a good scene.
If training fails or quality is poor, adjust `iterations` or try `strategy="mcmc"`.

**DO NOT STOP HERE. Continue to segmentation.**

---

## Step 6: Segment (SAM3 object detection)

SAM3 requires the BPE vocab file. It is located at:
`/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz`

The environment variable `SAM3_BPE_PATH` points to it. If SAM3 fails, the pipeline
will fall back to SAM2 automatic mask generation.

```bash
python3 -c "
import os
os.environ.setdefault('SAM3_BPE_PATH', '/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz')
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.segment(
    '/data/output/JOB_ID/model/point_cloud.ply',
    '/data/output/JOB_ID/frames/'
)
print(result)
"
```

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "segment", "progress": 0.65, "message": "Segmentation complete: N objects"}'
```

---

## Step 7: Extract per-object PLY files

```bash
python3 -c "
from pipeline.stages import PipelineStages
import json
p = PipelineStages('/data/output/JOB_ID')
# Use the objects JSON from segment() result
objects = [{'label': 'full_scene', 'count': -1}]
result = p.extract_objects('/data/output/JOB_ID/model/point_cloud.ply', objects)
print(result)
"
```

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "extract_objects", "progress": 0.7, "message": "Extracted N object PLYs"}'
```

---

## Step 8: Generate meshes (TSDF fusion)

Use TSDF fusion for mesh extraction. This produces watertight meshes with good topology.

```bash
python3 -c "
from pipeline.stages import PipelineStages
import json
p = PipelineStages('/data/output/JOB_ID')
plys = ['/data/output/JOB_ID/objects/full_scene.ply']
result = p.mesh_objects(plys)
print(result)
"
```

**Inspect in Blender**:
```bash
DISPLAY=:1 blender /data/output/JOB_ID/objects/meshes/full_scene/full_scene.glb
```

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "mesh_objects", "progress": 0.8, "message": "Mesh: Nk verts, Mk faces"}'
```

---

## Step 9: Texture bake (optional)

```bash
python3 -c "
from pipeline.stages import PipelineStages
import json
p = PipelineStages('/data/output/JOB_ID')
meshes = json.load(open('/data/output/JOB_ID/objects/meshes/mesh_manifest.json'))
result = p.texture_bake(meshes)
print(result)
"
```

---

## Step 10: Assemble USD scene

This is the final assembly step. Creates the hierarchical scene graph with variant sets.

```bash
python3 -c "
from pipeline.stages import PipelineStages
import json
p = PipelineStages('/data/output/JOB_ID')
meshes = json.load(open('/data/output/JOB_ID/objects/meshes/mesh_manifest.json'))
result = p.assemble_usd(meshes)
print(result)
"
```

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "assemble_usd", "progress": 0.95, "message": "USD scene assembled: N prims"}'
```

---

## Step 11: Validate

Run final validation across all outputs.

```bash
python3 -c "
from pipeline.stages import PipelineStages
p = PipelineStages('/data/output/JOB_ID')
result = p.validate()
print(result)
"
```

---

## Mark job complete

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/complete \
  -H 'Content-Type: application/json' \
  -d '{"success": true}'
```

```bash
curl -X POST http://localhost:7860/api/job/JOB_ID/stage \
  -H 'Content-Type: application/json' \
  -d '{"stage": "validate", "progress": 1.0, "message": "Pipeline complete"}'
```

---

## At each step: CHECK QUALITY

- Render a preview in Blender: `DISPLAY=:1 blender --background --python-expr "..."`
- If quality is poor, adjust parameters and re-run the stage
- The pipeline is not a script -- YOU decide what to do next
- You can skip stages, re-run stages, or change parameters between stages

## Quality Targets

- Frame selection: 60-80 diverse frames from 4fps extraction
- COLMAP: 70%+ registration rate, use sequential matcher for video
- Training: 30k+ iterations, MRNF strategy, loss < 0.02
- Segmentation: SAM3 text prompts for semantic labels (fallback to SAM2 if BPE missing)
- Mesh: per-object textured meshes via TSDF fusion
- USD: hierarchical scene graph with variant sets

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SAM3_BPE_PATH` | `/opt/sam3-repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz` | SAM3 text tokenizer vocab |
| `HF_TOKEN` | (from compose) | HuggingFace model downloads |
| `ANTHROPIC_API_KEY` | (from compose) | Claude Code API key |

## REST API for progress reporting

| Method | Endpoint | Body | Purpose |
|--------|----------|------|---------|
| POST | `/api/job/<id>/stage` | `{"stage": "...", "progress": 0.5, "message": "..."}` | Report stage progress |
| POST | `/api/job/<id>/stage/complete` | `{"stage": "...", "success": true}` | Mark stage done |
| POST | `/api/job/<id>/complete` | `{"success": true}` | Mark job done |
| GET | `/jobs` | -- | List all jobs |
| GET | `/status/<id>` | -- | Job detail |
