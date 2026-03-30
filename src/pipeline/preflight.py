# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pre-flight dependency verification. Fails HARD if critical deps missing."""

from __future__ import annotations

import logging
import shutil
import sys

logger = logging.getLogger(__name__)

REQUIRED_PYTHON = (3, 12)


def check_all() -> dict:
    """Verify all pipeline dependencies. Raises RuntimeError on critical failure."""
    results = {}

    # Python version
    if sys.version_info[:2] < REQUIRED_PYTHON:
        raise RuntimeError(
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required, "
            f"got {sys.version}"
        )
    results["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"

    # torch + CUDA
    try:
        import torch

        results["torch"] = torch.__version__
        if not torch.cuda.is_available():
            raise RuntimeError("torch installed but CUDA unavailable")
        results["cuda"] = True
        results["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        raise RuntimeError("CRITICAL: torch not installed. Pipeline cannot run.")

    # gsplat
    try:
        import gsplat

        results["gsplat"] = gsplat.__version__
    except ImportError:
        raise RuntimeError(
            "CRITICAL: gsplat not installed. Mesh extraction will produce garbage."
        )

    # trimesh + xatlas
    try:
        import trimesh
        import xatlas

        results["trimesh"] = trimesh.__version__
        results["xatlas"] = True
    except ImportError as e:
        raise RuntimeError(f"CRITICAL: {e}")

    # plyfile
    try:
        from plyfile import PlyData  # noqa: F401

        results["plyfile"] = True
    except ImportError:
        raise RuntimeError("CRITICAL: plyfile not installed")

    # COLMAP binary
    colmap = shutil.which("colmap")
    if not colmap:
        raise RuntimeError("CRITICAL: colmap not found in PATH")
    results["colmap"] = colmap

    # Blender binary (optional)
    blender = shutil.which("blender")
    results["blender"] = blender or "not found (optional)"

    # usd-core
    try:
        from pxr import Usd  # noqa: F401

        results["usd_core"] = True
    except ImportError:
        logger.warning(
            "usd-core not available -- USD assembly will use minimal format"
        )
        results["usd_core"] = False

    logger.info("Preflight check PASSED: %s", results)
    return results
