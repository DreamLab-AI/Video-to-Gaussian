# Agentic Quality Control

## Decision Trees

### Input Quality Assessment

```mermaid
graph TD
    A[Input Video] --> B{Blur Score<br/>Laplacian Variance}
    B -->|< 100| C[Enable Deblurring 3DGS]
    B -->|>= 100| D{Exposure Range}
    C --> D
    D -->|> 1.5 EV| E[Enable PPISP]
    D -->|<= 1.5 EV| F{Frame Count}
    E --> F
    F -->|< 50| G[WARN: Insufficient Coverage]
    F -->|50-500| H[Proceed]
    F -->|> 500| I[Subsample to 300-500]
    G --> H
    I --> H
```

### Training Quality Gate

```mermaid
graph TD
    A[Training Complete] --> B{Mean PSNR<br/>vs Training Views}
    B -->|< 22| C[FAIL: Retrain<br/>Adjust strategy/LR]
    B -->|22-25| D{Loss Trend}
    B -->|> 25| E[PASS]
    D -->|Plateau| F[Try pose optimisation]
    D -->|Decreasing| G[Continue training +10k iter]
    D -->|Oscillating| H[Reduce learning rate]
    F --> I{PSNR improved?}
    G --> I
    H --> I
    I -->|Yes| E
    I -->|No, after 3 attempts| J[ACCEPT: Best available]
```

### Mesh Quality Validation

```mermaid
graph TD
    A[Extracted Mesh] --> B{Vertex Count}
    B -->|< 1000| C[WARN: Too sparse]
    B -->|1k-1M| D{Watertight?}
    B -->|> 1M| E[Decimate to target]
    C --> D
    E --> D
    D -->|Yes| F{Surface Normal Consistency}
    D -->|No| G[Hole filling + remesh]
    G --> F
    F -->|Mean deviation < 15°| H[PASS]
    F -->|>= 15°| I[Re-extract with higher resolution]
```

### Round-Trip Validation

```
Original Gaussians → render views → PSNR_ref
Extracted Mesh → mesh2splat → Gaussians' → render views → PSNR_mesh

Quality Score = PSNR_mesh / PSNR_ref

Score > 0.95: Excellent mesh fidelity
Score 0.85-0.95: Acceptable
Score < 0.85: Re-extract with different method
```

## Metrics

| Metric | Tool | Threshold | Stage |
|--------|------|-----------|-------|
| Laplacian variance | OpenCV | 100 | Input quality |
| Exposure range | Histogram analysis | 1.5 EV | Input quality |
| Training loss | LichtFeld training.get_state | Convergence | Reconstruction |
| PSNR | skimage.metrics | 25 dB | Post-training |
| SSIM | skimage.metrics | 0.85 | Post-training |
| Mesh vertex count | Open3D / trimesh | 1k-1M | Mesh extraction |
| Watertightness | trimesh.is_watertight | True | Mesh extraction |
| Round-trip PSNR | mesh2splat | 0.85x ref | Mesh validation |
| Inpainting coherence | LPIPS | < 0.3 | Background recovery |
