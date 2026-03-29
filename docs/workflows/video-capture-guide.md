# Video Capture Guide

How to shoot video that produces the best results from the gaussian-toolkit pipeline.

## Camera Movement

- **Slow, steady orbits** around the scene. COLMAP needs parallax between frames to estimate depth.
- **Walk speed**: 0.5 m/s max. Slower is better.
- **No fast pans or whip turns** — causes motion blur and breaks feature matching.
- **Every surface** should be visible from at least 3-5 different viewpoints.
- **Full 360 coverage** if possible — orbit the room or space completely.
- **Multiple heights**: shoot at eye level, then slightly above and below if the scene allows.
- **Hit the corners** — COLMAP fails on areas only seen from one angle.
- **Avoid pure rotation** (spinning in place). The camera must translate, not just rotate, for depth estimation to work.

## Camera Settings

| Setting | Recommendation | Why |
|---------|---------------|-----|
| Resolution | 1080p | 4K works but COLMAP runs 4x slower |
| Frame rate | 30fps | Pipeline extracts at 1-2fps anyway |
| Shutter speed | 1/120s or faster | Minimises motion blur |
| ISO | As low as possible | Noise degrades feature matching |
| Exposure | Lock (manual or AE-lock) | Auto-exposure drift requires PPISP correction |
| Focus | Lock (manual or AF-lock) | Focus hunting creates defocus blur |
| White balance | Lock | Auto WB shifts confuse colour operations |
| Stabilisation | On (optical or electronic) | Reduces blur, keeps framing steady |

## Lighting

- **Consistent** across the space. Avoid mixed lighting (daylight + fluorescent).
- **No harsh shadows** that move with camera position — gaussians bake shadows as geometry.
- **Diffuse light is best** — overcast outdoor or well-lit indoor with soft sources.
- **Avoid direct sun** causing extreme dynamic range that clips highlights.
- **Uniform brightness**: the fewer exposure stops between brightest and darkest areas, the better.

## What Works Well

- Gallery and museum rooms with distinct objects (paintings, sculptures, furniture)
- Living rooms and offices with varied furniture
- Outdoor courtyards with architecture and vegetation
- Shop interiors with products on shelves
- Workshops and studios with tools and equipment

## What Kills the Pipeline

| Problem | Why | Workaround |
|---------|-----|------------|
| Featureless surfaces (white walls, glossy floors) | COLMAP cannot find matching features | Place textured objects or use textured tape markers |
| Moving objects (people walking through) | Creates ghost gaussians from inconsistent views | Clear the space or shoot when empty |
| Reflective surfaces (mirrors, glass) | Gaussians bake the reflection as geometry | Mask reflective areas in post, or cover mirrors |
| Transparent objects (glass tables, windows) | Partially visible through, confuses depth | Accept as limitation — will reconstruct as opaque |
| Repetitive patterns (tiled floors, brick walls) | COLMAP mismatches features across pattern repeats | Include unique landmarks visible from tiled areas |
| Extreme dynamic range | Blown highlights or crushed shadows lose features | Use HDR if camera supports it, or shoot in flat/log profile |
| Fast camera movement | Motion blur prevents feature detection | Keep movement slow and deliberate |

## Duration and Frame Count

| Video Length | Frames at 2fps | Quality | COLMAP Time |
|-------------|-----------------|---------|-------------|
| 15-30s | 30-60 | Minimum viable | 5 min |
| 30-60s | 60-120 | Good | 15-20 min |
| 60-90s | 120-180 | Excellent | 30-45 min |
| 2-5 min | 240-600 | Overkill (diminishing returns) | 1-3 hours |

**Sweet spot: 30-90 seconds** for a single room. This gives 60-180 frames at 2fps extraction — enough views for dense reconstruction without overwhelming COLMAP.

## Phone-Specific Tips

### iPhone
- Use 1080p 30fps (not 4K, not 60fps)
- Lock exposure by long-pressing the screen
- Use ProVideo or Filmic Pro for manual controls if available
- Cinematic mode adds fake depth blur — **turn it off**

### Android
- Use Open Camera app for manual controls
- Lock focus and exposure before starting
- Disable HDR+ processing (it stitches multiple exposures, causing ghosting)

### Drone (DJI)
- Orbit mode at slow speed (5-10 degrees/second)
- Fixed gimbal angle (slightly below horizontal)
- Enable SRT telemetry recording — SplatReady can embed GPS EXIF
- 1080p is sufficient; 4K increases storage but not quality proportionally
- Fly at consistent altitude

## Multi-Room Captures

For spaces with multiple rooms connected by doorways:
1. Capture each room as a separate video (30-60s each)
2. Include the doorway/transition area in both videos
3. Process each room separately through the pipeline
4. Assemble room USD scenes into a building-level USD using composition arcs

Attempting a single continuous video through multiple rooms produces a very long COLMAP run and often fails at the narrow doorway transitions where parallax is minimal.

## Validation Checklist

Before running the pipeline, review your video:

- [ ] Camera moves steadily without sudden jerks
- [ ] All surfaces of interest are visible from multiple angles
- [ ] No people or pets moving through the scene
- [ ] Exposure is consistent (no brightness flicker)
- [ ] Focus is consistent (no hunting)
- [ ] Video is 30-90 seconds long
- [ ] Resolution is 1080p
- [ ] Lighting is even across the space
