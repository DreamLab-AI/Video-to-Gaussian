# Build Troubleshooting

## METIS Linker Errors

**Symptom**: COLMAP build fails with `undefined reference to gk_*` symbols.

**Cause**: scotch-metis compatibility library does not provide the full METIS 5 API. Ceres-solver's FindSuiteSparse links against METIS::METIS which needs the real METIS + GKlib.

**Fix**: Build METIS 5.2.1 + GKlib from source as static libraries. See [Build Instructions](../build/linux.md#build-metis-from-source).

## vcpkg x264 Fails (nasm not found)

**Symptom**: `Could not find nasm` during vcpkg x264 build.

**Fix**: `sudo pacman -S nasm`

## vcpkg libb2 Fails (autotools)

**Symptom**: libb2 build fails looking for autoconf/automake.

**Fix**: `sudo pacman -S autoconf autoconf-archive automake libtool`

## LichtFeld Python Stubgen Error

**Symptom**: Build fails at 559/559 with numpy ImportError in stubgen.

**Cause**: vcpkg bundles Python 3.12 but the system numpy is for Python 3.14. The stub generator tries to import system numpy via vcpkg's Python.

**Impact**: None. The binary links at step 557/559. The stub generation failure only affects Python type hints for IDE autocompletion.

**Fix**: Use `|| true` in build scripts. Verify the binary exists with `test -f build/LichtFeld-Studio`.

## pacman Signature Errors

**Symptom**: `signature from "CachyOS" is invalid` during package install.

**Fix**:
```bash
sudo pacman -Sy cachyos-keyring archlinux-keyring
```

## CUDA Architecture Mismatch

**Symptom**: Runtime errors or slow performance.

**Fix**: Set the correct compute capability for your GPU:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Returns e.g. "8.6" → use -DCMAKE_CUDA_ARCHITECTURES="86"
```

Common values:
| GPU | Compute Capability |
|-----|-------------------|
| RTX 4090/4080 | 89 |
| RTX A6000/3090/3080 | 86 |
| RTX A5000/3070/3060 | 86 |
| RTX 2080 Ti/Quadro RTX 6000 | 75 |
| V100 | 70 |
| A100 | 80 |
