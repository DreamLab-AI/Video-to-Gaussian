# ADR-002: Docker Python Version Management for CUDA + USD Pipeline

**Status**: Proposed
**Date**: 2026-03-30
**Context**: Multi-Python (3.11/3.12) Docker container causing import failures in ML pipeline
**Research method**: Web search for production best practices + codebase analysis

---

## Problem Statement

The consolidated Dockerfile (`Dockerfile.consolidated`) installs both Python 3.11
(for usd-core) and Python 3.12 (system default, for torch/gsplat/ComfyUI). The
`update-alternatives` mechanism sets python3 -> python3.12 (priority 2) over
python3.11 (priority 1), but:

1. The USD venv at `/opt/venv-usd` uses python3.11 and is on `PATH` before
   system python (line 262: `ENV PATH="/opt/gaussian-toolkit/build:/opt/venv-usd/bin:${PATH}"`)
2. When Claude Code or subprocess calls `python3`, it resolves to the venv's
   python3.11 instead of system python3.12
3. gsplat, torch, sam3 are installed for python3.12 only
4. Result: pipeline stages fall back to convex hull (818 verts vs 47K)

The PATH ordering in the Dockerfile is the root cause:
```
/opt/venv-usd/bin  <-- python3.11 shadows system python3
/usr/local/bin
/usr/bin           <-- python3.12 via update-alternatives, never reached
```

## Critical Finding: usd-core Now Supports Python 3.12

**usd-core 26.3** (released Feb 2026) ships `cp312` wheels for Linux:
- `usd_core-26.3-cp312-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`

This eliminates the need for Python 3.11 entirely. The original constraint
(usd-core required 3.11) no longer applies.

Source: [usd-core on PyPI](https://pypi.org/project/usd-core/)

---

## Recommendation: Single Python Version (Option A -- Preferred)

### Eliminate Python 3.11 entirely

Since usd-core 26.3 supports Python 3.12, unify everything on a single
interpreter. This is the approach used by NVIDIA's own NGC PyTorch containers
and is the consensus best practice for production ML Docker images.

### Dockerfile Changes

```dockerfile
# REMOVE: deadsnakes PPA, python3.11, python3.11-venv, python3.11-dev
# REMOVE: update-alternatives for python3
# REMOVE: /opt/venv-usd creation with python3.11

# KEEP: python3.12 as the sole interpreter
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install ALL Python deps into system python3.12
RUN pip3 install --break-system-packages \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install --break-system-packages \
        gsplat trimesh xatlas pymeshfix fast-simplification scikit-image \
        pillow numpy scipy opencv-python-headless \
        segment-anything psutil plyfile \
        usd-core>=26.3 \
        # ... rest of deps

# REMOVE /opt/venv-usd/bin from PATH
ENV PATH="/opt/gaussian-toolkit/build:${PATH}"
```

### Pipeline Code Changes

In `src/pipeline/stages.py`, the `_find_usd_python()` function (line 194) can
be simplified:

```python
def _find_usd_python() -> Path | None:
    """System python3 now has usd-core installed directly."""
    import shutil
    py = shutil.which("python3")
    if py is None:
        return None
    p = Path(py)
    try:
        result = subprocess.run(
            [str(p), "-c", "from pxr import Usd; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and "ok" in result.stdout:
            return p
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None
```

The `PYTHONPATH=""` env override in entrypoint.sh `usd-assemble` wrapper
(line 54-57) also becomes unnecessary.

### Advantages

- Zero ambiguity about which python3 runs
- No PATH shadowing bugs
- Simpler Dockerfile (fewer layers, smaller image)
- All packages share the same site-packages
- subprocess calls from Claude Code always resolve correctly
- `python3 -c "import gsplat"` always works

### Risk

- If usd-core 26.3 has regressions on 3.12, the USD assembly stage breaks
- Mitigation: test `from pxr import Usd, UsdGeom; print(Usd.GetVersion())` in
  CI before deploying

---

## Alternative: Keep Dual Python with Fixed PATH (Option B -- Fallback)

If usd-core 26.3+cp312 proves unreliable, keep dual Python but fix the PATH
and resolution bugs.

### Key Changes

1. **Remove /opt/venv-usd/bin from global PATH**

```dockerfile
# WRONG (current):
ENV PATH="/opt/gaussian-toolkit/build:/opt/venv-usd/bin:${PATH}"

# FIXED:
ENV PATH="/opt/gaussian-toolkit/build:${PATH}"
```

2. **Use explicit absolute paths for USD operations only**

The entrypoint already has the right pattern with the `usd-assemble` wrapper:
```bash
PYTHONPATH="" exec /opt/venv-usd/bin/python3 /opt/scripts/assemble_usd_scene.py "$@"
```

Extend this pattern: every USD invocation uses `/opt/venv-usd/bin/python3`
explicitly. Everything else uses bare `python3` which resolves to 3.12.

3. **Add a guard script at container startup**

```bash
# In entrypoint.sh, add validation:
echo "=== Python Environment Validation ==="
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PYVER" != "3.12" ]; then
    echo "FATAL: python3 resolves to $PYVER, expected 3.12"
    echo "PATH=$PATH"
    echo "which python3: $(which python3)"
    exit 1
fi
python3 -c "import torch; print(f'torch {torch.__version__} CUDA={torch.cuda.is_available()}')"
python3 -c "from gsplat import rasterization; print('gsplat OK')"
echo "=== Python OK ==="
```

4. **Pipeline stages should fail hard on wrong Python**

In `src/pipeline/stages.py`, the gsplat mesh extraction (line 963) catches
all exceptions and falls back silently. Add an explicit version check:

```python
import sys
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(
        f"Pipeline requires Python 3.12, running {sys.version}. "
        f"Check PATH: /opt/venv-usd/bin may be shadowing system python3."
    )
```

### Advantages

- Preserves known-working usd-core 3.11 setup
- Minimal code changes

### Disadvantages

- Two Python interpreters to maintain
- Easy to regress if anyone modifies PATH
- More complex mental model

---

## Alternative: Separate venvs with Wrapper Scripts (Option C -- Not Recommended)

Create two venvs (one for ML/torch, one for USD) and use wrapper scripts.

```dockerfile
RUN python3.12 -m venv /opt/venv-ml
RUN /opt/venv-ml/bin/pip install torch gsplat ...

RUN python3.11 -m venv /opt/venv-usd
RUN /opt/venv-usd/bin/pip install usd-core ...
```

With wrappers:
```bash
#!/bin/bash
# /usr/local/bin/pipeline-python
exec /opt/venv-ml/bin/python3 "$@"
```

This is **not recommended** because:
- Docker containers are already isolated; venvs inside Docker add complexity
  without benefit (confirmed by multiple production ML guides)
- Every subprocess call needs to know which wrapper to use
- Claude Code would need to be taught about the wrappers
- The Dockerfile ENV trick (`ENV VIRTUAL_ENV=/opt/venv-ml` + `ENV PATH=...`)
  only works for one venv at a time

---

## Answers to Specific Questions

### Q1: Correct way to handle multiple Python versions where CUDA needs 3.12 and USD needs 3.11?

**Now moot.** usd-core 26.3 supports Python 3.12. Unify on 3.12 (Option A).

If forced to keep dual versions: never put the secondary venv on global PATH.
Use explicit absolute paths (`/opt/venv-usd/bin/python3`) for the minority
use case (USD). The majority use case (torch/gsplat) gets the default `python3`.

### Q2: Single venv for everything, or separate venvs with wrapper scripts?

**Single interpreter, no venvs.** In Docker, the container IS the isolation.
venvs inside Docker are a "misapplied best practice" per recent analysis.
Install everything into system site-packages with `--break-system-packages`.

Exception: if two packages have genuinely conflicting transitive dependencies
that cannot coexist, use one venv for the minority case (the USD pattern
already in entrypoint.sh is correct).

### Q3: How do production ML Docker images handle torch + specialized packages?

NVIDIA NGC containers (`nvcr.io/nvidia/pytorch:*`) install everything into a
single Conda environment inside the container. The key practices:

- Pin exact torch version matching the CUDA toolkit in the base image
- Use `--index-url https://download.pytorch.org/whl/cu128` for CUDA-specific wheels
- Never mix PyPI torch (CPU) with NVIDIA CUDA runtime
- Set `--shm-size=64g` in docker-compose for NCCL/DataLoader

### Q4: Better approach than update-alternatives for Python version management in Docker?

**Yes.** In Docker, update-alternatives is unnecessary complexity. Instead:

- Install only the Python version you need
- If you must have two, make one the unambiguous default and reference the other
  by absolute path (`/usr/bin/python3.11`)
- Set `ENV PATH` to control resolution order (the Docker-native approach)
- The "activate venv in Docker" trick: `ENV VIRTUAL_ENV=/opt/venv` +
  `ENV PATH="/opt/venv/bin:${PATH}"` -- but only for ONE venv

### Q5: Should pipeline stages detect wrong Python and fail hard?

**Yes, absolutely.** Silent fallback is the worst outcome -- you get 818 vertices
instead of 47K and don't know why until you inspect the output.

Add at minimum:
```python
# At top of mesh_extractor.py and stages.py
import sys
assert sys.version_info >= (3, 12), (
    f"Expected Python >=3.12, got {sys.version}. "
    f"gsplat/torch require 3.12. Check PATH."
)
```

And in the entrypoint, validate before starting services (see Option B above).

### Q6: How should Dockerfile guarantee `python3 -c "import gsplat"` always works?

Three layers of defense:

1. **Build-time**: Add a `RUN` step that fails the build if import fails:
   ```dockerfile
   RUN python3 -c "import gsplat; from gsplat import rasterization; print('gsplat OK')" \
       && python3 -c "import torch; assert torch.cuda.is_available() or True; print('torch OK')"
   ```
   (Note: CUDA not available during build, so test import only, not `.is_available()`)

2. **Boot-time**: Entrypoint validation (see Q5 above)

3. **Run-time**: Hard assert in pipeline code, not try/except fallback

---

## Implementation Plan

### Phase 1: Validate usd-core 26.3 on Python 3.12 (1 hour)
```bash
docker run --rm nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -c '
  apt-get update && apt-get install -y python3.12 python3-pip &&
  pip3 install --break-system-packages usd-core>=26.3 &&
  python3 -c "from pxr import Usd, UsdGeom; print(Usd.GetVersion())"
'
```

### Phase 2: Update Dockerfile.consolidated (if Phase 1 passes)
- Remove python3.11, deadsnakes PPA, update-alternatives
- Remove /opt/venv-usd
- Install usd-core into system python3.12
- Add build-time import validation RUN step
- Fix PATH to remove /opt/venv-usd/bin

### Phase 3: Update entrypoint.sh
- Remove usd-assemble wrapper (or simplify to just call python3 directly)
- Add boot-time Python validation

### Phase 4: Update pipeline code
- Simplify `_find_usd_python()` in stages.py
- Add version assertions to mesh_extractor.py
- Remove silent gsplat fallback (or make it loud)

### Phase 5: If Phase 1 fails, implement Option B
- Remove /opt/venv-usd/bin from PATH
- Keep explicit path references for USD
- Add entrypoint validation
- Add hard asserts in pipeline code

---

## Sources

- [RunPod: Optimizing Docker Setup for PyTorch Training with CUDA 12.8](https://www.runpod.io/articles/guides/docker-setup-pytorch-cuda-12-8-python-3-11)
- [Red Hat: A developer's guide to PyTorch, containers, and NVIDIA](https://next.redhat.com/2025/08/26/a-developers-guide-to-pytorch-containers-and-nvidia-solving-the-puzzle/)
- [NVIDIA Docs: Containers For Deep Learning Frameworks](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html)
- [Snyk: Mastering Python virtual environments](https://snyk.io/blog/mastering-python-virtual-environments/)
- [Python Speed: Elegantly activating a virtualenv in a Dockerfile](https://pythonspeed.com/articles/activate-virtualenv-dockerfile/)
- [Will Barillon: Python venv in Docker: A Misapplied Best Practice](https://wbarillon.medium.com/python-venv-in-docker-a-misapplied-best-practice-a1bd7465106e)
- [usd-core 26.3 on PyPI](https://pypi.org/project/usd-core/) -- cp312 wheels confirmed
- [OpenUSD Issue #3116: Python 3.12 support](https://github.com/PixarAnimationStudios/OpenUSD/issues/3116)
- [Markaicode: Reproducible PyTorch Training with Docker](https://markaicode.com/docker-gpu-pytorch-training/)
- [ddelange/pycuda: Compact multi-stage CUDA Docker images](https://github.com/ddelange/pycuda)
