# Workflow: Video to Trained Gaussian Splat

## One-Command Pipeline

```bash
video2splat /path/to/drone_footage.mp4 /output/my_scene 0.5 30000 mcmc
```

Parameters: `<video> <output_dir> [fps] [max_iterations] [strategy]`

Strategies: `mcmc` (default, best quality), `mrnf`, `igs+`

## Step-by-Step Manual Pipeline

### 1. Extract Frames

```bash
python3 -c "
import sys; sys.path.insert(0, '$HOME/.lichtfeld/plugins/splat_ready')
from core.frame_extractor import extract_frames
extract_frames('/path/to/video.mp4', '/output/my_scene', 0.5, print)
"
```

### 2. Run COLMAP Reconstruction

```bash
python3 -c "
import sys; sys.path.insert(0, '$HOME/.lichtfeld/plugins/splat_ready')
from core.colmap_processor import process_colmap
process_colmap('/output/my_scene/frames/video', '/output/my_scene',
               '/usr/local/bin/colmap', {'max_image_size': 2000}, print)
"
```

### 3. Train with LichtFeld

```bash
lichtfeld-studio --headless \
    --data-path /output/my_scene/colmap/undistorted \
    --output-path /output/my_scene/model \
    --iter 30000 \
    --strategy mcmc
```

### 4. Export

```bash
lichtfeld-studio convert /output/my_scene/model/point_cloud.ply /output/my_scene/model.spz
lichtfeld-studio convert /output/my_scene/model/point_cloud.ply /output/my_scene/viewer.html
```

## Agent-Controlled Training (MCP)

```bash
# Start GUI mode
lichtfeld-studio &

# Load dataset
lfs-mcp call scene.load_dataset '{"path":"/output/my_scene/colmap/undistorted"}'

# Start training
lfs-mcp call training.start

# Monitor (poll until done)
lfs-mcp call training.get_state

# Capture renders
lfs-mcp call render.capture '{"width":1920,"height":1080}'

# Export
lfs-mcp call scene.export_spz '{"path":"/output/model.spz"}'
lfs-mcp call scene.export_html '{"path":"/output/viewer.html"}'
```

## Quality Tips

| Parameter | Low Quality / Fast | Balanced | High Quality |
|-----------|-------------------|----------|--------------|
| FPS | 0.2 | 0.5 | 1.0-2.0 |
| Frames | 50-100 | 150-300 | 500+ |
| COLMAP max_image_size | 1000 | 2000 | 4000 |
| Training iterations | 10000 | 30000 | 60000+ |
| Strategy | mcmc | mcmc | mcmc |
