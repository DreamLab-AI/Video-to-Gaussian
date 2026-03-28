# Build Instructions — Linux

## Prerequisites

### System Requirements

- **OS**: Arch Linux, CachyOS, or similar (tested on CachyOS v3)
- **GPU**: NVIDIA with compute capability >= 7.5 (RTX 2080+)
- **CUDA**: 12.0+ (tested with 13.1)
- **Compiler**: GCC 14+ (tested with 15.2.1)
- **RAM**: 16GB+ recommended for compilation
- **Disk**: ~10GB for vcpkg packages + build artifacts

### Install System Dependencies

```bash
# Core build tools
sudo pacman -S cmake ninja nasm git curl

# COLMAP dependencies
sudo pacman -S eigen ceres-solver suitesparse cgal \
    google-glog glew sqlite openimageio

# Autotools (required by vcpkg libb2, ffmpeg)
sudo pacman -S autoconf autoconf-archive automake libtool

# Python dependencies for SplatReady
pip install --break-system-packages pillow piexif av
```

### Build METIS from Source

COLMAP requires METIS via Ceres/SuiteSparse. The scotch-metis compatibility library does not provide the full API.

```bash
cd /tmp
curl -sL https://github.com/KarypisLab/METIS/archive/refs/tags/v5.2.1.tar.gz | tar xz
cd METIS-5.2.1
git clone --depth 1 https://github.com/KarypisLab/GKlib.git

# Build GKlib (static)
mkdir -p GKlib/build && cd GKlib/build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-fPIC" \
    -DBUILD_SHARED_LIBS=OFF
make -j$(nproc) GKlib && sudo make install

# Build METIS (static)
cd /tmp/METIS-5.2.1
mkdir -p build/xinclude
echo "#define IDXTYPEWIDTH 32" > build/xinclude/metis.h
echo "#define REALTYPEWIDTH 32" >> build/xinclude/metis.h
cat include/metis.h >> build/xinclude/metis.h
cp include/CMakeLists.txt build/xinclude
cd build
cmake /tmp/METIS-5.2.1 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DGKLIB_PATH=/tmp/METIS-5.2.1/GKlib \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-fPIC"
make -j$(nproc) metis
sudo cp libmetis/libmetis.a /usr/local/lib/
sudo cp /tmp/METIS-5.2.1/build/xinclude/metis.h /usr/local/include/

# Create cmake config
sudo mkdir -p /usr/local/lib/cmake/metis
sudo tee /usr/local/lib/cmake/metis/metis-config.cmake << 'EOF'
if(NOT TARGET METIS::METIS)
  add_library(METIS::METIS STATIC IMPORTED)
  set_target_properties(METIS::METIS PROPERTIES
    IMPORTED_LOCATION "/usr/local/lib/libmetis.a"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include"
    INTERFACE_LINK_LIBRARIES "/usr/local/lib/libGKlib.a;m")
endif()
set(METIS_FOUND TRUE)
EOF
sudo ldconfig
```

## Build COLMAP

```bash
git clone --depth 1 https://github.com/colmap/colmap.git /tmp/colmap-build
cd /tmp/colmap-build

# Adjust CUDA architectures to match your GPU
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="86;75" \
    -DGUI_ENABLED=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_PREFIX_PATH=/usr/local \
    -G Ninja

cmake --build build -- -j$(nproc)
sudo cmake --install build
colmap --version
```

## Build LichtFeld Studio

### Setup vcpkg

```bash
git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg
/opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_ROOT=/opt/vcpkg
```

### Clone and Build

```bash
git clone --recursive https://github.com/jjohare/gaussian-toolkit.git
cd gaussian-toolkit

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -G Ninja

# vcpkg will download and build ~91 dependencies (first build takes 15-20 minutes)
cmake --build build -- -j$(nproc)
```

> **Note**: The build may report a Python stubgen error at 559/559 due to a numpy version mismatch with vcpkg's bundled Python. The binary links successfully at step 557/559. This is harmless.

### Verify

```bash
./build/LichtFeld-Studio --version
./build/LichtFeld-Studio --warmup   # Pre-compile PTX kernels
```

### Install

```bash
sudo ln -sf $(pwd)/build/LichtFeld-Studio /usr/local/bin/lichtfeld-studio
```

## Install SplatReady Plugin

```bash
git clone https://github.com/jacobvanbeets/SplatReady.git \
    ~/.lichtfeld/plugins/splat_ready

# Pre-configure with COLMAP path
cat > ~/.lichtfeld/plugins/splat_ready/pipeline_config.json << 'EOF'
{
  "reconstruction_method": "colmap",
  "colmap_exe_path": "/usr/local/bin/colmap",
  "frame_rate": 1.0,
  "max_image_size": 2000,
  "min_scale": 0.5
}
EOF
```

## Install CLI Tools

The `scripts/tools/` directory contains CLI wrappers:

```bash
sudo cp scripts/tools/lfs-mcp.sh /usr/local/bin/lfs-mcp
sudo cp scripts/tools/video2splat.sh /usr/local/bin/video2splat
sudo chmod +x /usr/local/bin/lfs-mcp /usr/local/bin/video2splat
```

## Environment Variables

Add to your shell profile:

```bash
export LICHTFELD_EXECUTABLE=/path/to/build/LichtFeld-Studio
export LICHTFELD_MCP_ENDPOINT=http://127.0.0.1:45677/mcp
export VCPKG_ROOT=/opt/vcpkg
export LD_LIBRARY_PATH="/path/to/build:$LD_LIBRARY_PATH"
```

## Docker Build

See [Docker Integration](../integration/docker.md) for containerised builds.
