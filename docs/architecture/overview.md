# Gaussian Toolkit Architecture

## System Overview

Gaussian Toolkit integrates three core components into a unified 3D Gaussian Splatting pipeline:

```
                     ┌──────────────────────────────────────────┐
                     │            Gaussian Toolkit               │
                     ├──────────────────────────────────────────┤
  Video Input ──────►│  SplatReady     COLMAP      LichtFeld    │──────► Trained Model
  Image Folder ─────►│  (frames)  ──► (SfM)  ──► (training)    │──────► PLY/SPZ/USD/HTML
  COLMAP Dataset ───►│                                          │──────► Renders
                     │          MCP Server (70+ tools)          │
                     │          JSON-RPC 2.0 / HTTP POST        │
                     │          Port 45677                      │
                     └──────────────────────────────────────────┘
                                    ▲
                                    │ lfs-mcp CLI / MCP bridge
                                    ▼
                            Claude Code / Agents
```

## Component Stack

| Component | Version | License | Purpose |
|-----------|---------|---------|---------|
| LichtFeld Studio | 0.4.2+ | GPL-3.0 | 3DGS training, visualisation, editing, export |
| COLMAP | 4.1.0 | BSD | Structure-from-Motion reconstruction |
| SplatReady | 1.0.0 | Plugin | Video-to-COLMAP pipeline automation |
| METIS | 5.2.1 | Apache-2.0 | Graph partitioning (COLMAP dependency) |
| vcpkg | latest | MIT | C++ dependency management (91 packages) |

## Data Flow

### Full Pipeline (video2splat)

```
Video File (.mp4/.mov)
    │
    ▼ [Stage 1: SplatReady - PyAV frame extraction]
JPEG Frames + GPS EXIF
    │
    ▼ [Stage 2: COLMAP - 6-step SfM pipeline]
    │   feature_extractor → exhaustive_matcher → mapper
    │   → model_aligner → image_undistorter → model_converter
    │
    ▼
COLMAP Undistorted Dataset
    │   images/ + sparse/0/{cameras,images,points3D}.txt
    │
    ▼ [Stage 3: LichtFeld - CUDA-accelerated training]
Trained 3D Gaussian Splat Model
    │
    ▼ [Export]
PLY / SOG / SPZ / USD / HTML
```

### MCP-Controlled Pipeline

```
Claude Agent
    │
    ├─► lfs-mcp call scene.load_dataset {"path": "..."}
    ├─► lfs-mcp call training.start
    ├─► lfs-mcp call training.get_state  (poll loop)
    ├─► lfs-mcp call render.capture {"width": 1920}
    ├─► lfs-mcp call selection.by_description {"description": "floaters"}
    ├─► lfs-mcp call gaussians.write {"delete_selected": true}
    └─► lfs-mcp call scene.export_spz {"path": "output.spz"}
```

## GPU Architecture

- **CUDA Toolkit**: 13.1
- **Target architectures**: sm_86 (RTX A6000/3090), sm_75 (RTX 6000/2080)
- **C++ Standard**: C++23
- **CUDA Standard**: C++20
- **Build system**: CMake + Ninja + vcpkg

## MCP Server Architecture

```
scripts/lichtfeld_mcp_bridge.py    ◄── stdio MCP client (Claude Desktop/Codex)
        │
        │ HTTP POST to http://127.0.0.1:45677/mcp
        ▼
src/mcp/mcp_http_server.cpp        ◄── cpp-httplib HTTP listener
        │
        ▼
src/mcp/mcp_server.cpp             ◄── JSON-RPC 2.0 dispatcher
        │
        ├──► ToolRegistry (singleton)       70+ tools
        └──► ResourceRegistry (singleton)   8+ resources
```

### Tool Runtime Model

Tools are registered in two backends depending on the application mode:

- **Headless**: `TrainingContext` singleton manages scene/trainer directly
- **GUI**: `Visualizer` provides the backend with live viewport interaction

Each tool carries metadata:
- `category` (training, scene, render, selection, etc.)
- `kind` (command vs query)
- `runtime` (shared, headless, gui)
- `thread_affinity` (any, training_context, main_thread)
- `destructive`, `long_running`, `user_visible` flags

## Directory Layout

```
gaussian-toolkit/
├── docs/                          # This documentation
│   ├── architecture/              # System design
│   ├── build/                     # Build instructions
│   ├── integration/               # MCP, skills, Docker
│   ├── workflows/                 # Usage workflows
│   └── troubleshooting/           # Common issues
├── scripts/
│   └── tools/                     # CLI wrappers (lfs-mcp, video2splat)
├── src/                           # LichtFeld Studio source (upstream)
│   ├── mcp/                       # Built-in MCP server
│   ├── app/                       # Application entry + GUI tools
│   └── ...
├── external/                      # Git submodules
└── build/                         # Build output (not committed)
```
