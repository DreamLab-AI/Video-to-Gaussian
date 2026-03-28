# SplatReady Plugin Integration

[SplatReady](https://github.com/jacobvanbeets/SplatReady) is a LichtFeld Studio plugin that converts video files into COLMAP-compatible datasets for 3D Gaussian Splatting training.

## Pipeline Stages

### Stage 1: Frame Extraction

- **Engine**: PyAV (Python bindings for FFmpeg)
- **Input**: Video file (.mp4, .mov, .avi) or folder of videos
- **Output**: JPEG frames at configurable FPS
- **Features**: GPS EXIF embedding from DJI SRT files, VFR-safe timestamp extraction
- **Modes**: FPS-based or target frame count

### Stage 2: 3D Reconstruction

- **Primary**: COLMAP (headless, CUDA-accelerated)
- **Alternatives**: Agisoft Metashape, RealityScan
- **COLMAP Pipeline**:
  1. `feature_extractor` — SIFT feature detection
  2. `exhaustive_matcher` — Pairwise feature matching
  3. `mapper` — Sparse reconstruction
  4. `model_aligner` — Coordinate alignment
  5. `image_undistorter` — Lens distortion removal
  6. `model_converter` — Export to text format

### Stage 3: Import (GUI mode)

One-click import into LichtFeld Studio via `lf.load_file()`.

## Output Structure

```
output_dir/
  frames/
    VideoName/
      VideoName_frame_0001.jpg    # With GPS EXIF if SRT available
      VideoName_frame_0002.jpg
      ...
  colmap/
    undistorted/
      images/                      # Undistorted images
      sparse/0/
        cameras.txt                # Camera intrinsics
        images.txt                 # Camera extrinsics
        points3D.txt               # Sparse point cloud
```

## Configuration

Config file at `~/.lichtfeld/plugins/splat_ready/pipeline_config.json`:

```json
{
  "video_path": "",
  "base_output_folder": "",
  "extraction_mode": 0,
  "frame_rate": 1.0,
  "desired_frames": 100,
  "skip_extraction": false,
  "manual_frames_folder": "",
  "reconstruction_method": "colmap",
  "colmap_exe_path": "/usr/local/bin/colmap",
  "use_fisheye": false,
  "max_image_size": 2000,
  "min_scale": 0.5,
  "skip_reconstruction": false
}
```

## CLI Usage (Headless)

### Full Pipeline

```bash
video2splat /path/to/video.mp4 /path/to/output 1.0 30000 mcmc
```

### Runner Script Directly

```bash
cat > /tmp/config.json << 'EOF'
{
  "video_path": "/path/to/video.mp4",
  "base_output_folder": "/path/to/output",
  "frame_rate": 1.0,
  "reconstruction_method": "colmap",
  "colmap_exe_path": "/usr/local/bin/colmap",
  "max_image_size": 2000
}
EOF
python3 ~/.lichtfeld/plugins/splat_ready/core/runner.py /tmp/config.json
```

### Individual Stages

```python
# Frame extraction only
from core.frame_extractor import extract_frames
frames_dir = extract_frames("/path/to/video.mp4", "/output", 1.0, print)

# COLMAP only
from core.colmap_processor import process_colmap
result = process_colmap(frames_dir, "/output", "/usr/local/bin/colmap",
                        {"max_image_size": 2000, "min_scale": 0.5}, print)
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| av (PyAV) | 17.0.0 | Video frame extraction |
| Pillow | latest | Image handling |
| piexif | 1.1.3 | GPS EXIF embedding |
| COLMAP | 4.1.0 | Structure-from-Motion |
| FFmpeg | system | Video codec support (via PyAV) |

## Tips

- **Drone footage**: Use 0.5-1.0 FPS for walk-around captures, 0.2-0.5 for drone orbits
- **Frame count mode**: Set desired frames to 100-300 for good coverage without redundancy
- **Fisheye**: Enable for GoPro, DJI Action cameras, or wide-angle lenses
- **Max image size**: 2000px balances quality vs COLMAP processing time. Use 4000+ for production
- **GPS data**: DJI drones produce `.SRT` files alongside `.MP4` — keep them in the same directory
