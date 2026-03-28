#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Assemble a USD scene from gallery COLMAP data and trained Gaussian PLY.

Reads COLMAP text-format reconstruction files, creates a hierarchical USD
stage with camera prims, object references, and scene metadata, then writes
the result to a .usda file.

Usage:
    python scripts/assemble_gallery_usd.py \\
        --colmap-dir colmap/exported \\
        --output test-data/gallery_output/scene/gallery_scene.usda \\
        [--ply-path path/to/point_cloud.ply]

If no COLMAP directory exists yet, synthetic data is generated for testing.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from pipeline.colmap_parser import parse_cameras_txt, parse_images_txt
from pipeline.coordinate_transform import CoordinateTransformer
from pipeline.usd_assembler import ObjectDescriptor, UsdSceneAssembler


def _write_synthetic_colmap(colmap_dir: Path) -> None:
    """Generate synthetic COLMAP files for a gallery-like scene."""
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # 4 cameras at cardinal directions around a central gallery
    (colmap_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "1 PINHOLE 1920 1080 1500.0 1500.0 960.0 540.0\n"
    )

    import math

    radius = 5.0
    lines = [
        "# Image list with two lines of data per image:\n",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
    ]
    for i, angle_deg in enumerate([0, 90, 180, 270], start=1):
        angle = math.radians(angle_deg)
        # Camera position on a circle, looking at origin
        cx = radius * math.cos(angle)
        cz = radius * math.sin(angle)
        cy = 1.5  # eye height

        # For a simple orbit camera looking at origin, the rotation is
        # a Y-axis rotation plus the identity. We approximate with
        # a quaternion about Y (COLMAP convention: camera-to-world).
        half = angle / 2.0 + math.pi / 2.0  # face inward
        qw = math.cos(half)
        qy = math.sin(half)

        # COLMAP stores t = -R @ C, for identity-ish R just negate position
        tx, ty, tz = -cx, -cy, -cz
        lines.append(
            f"{i} {qw:.6f} 0.0 {qy:.6f} 0.0 {tx:.6f} {ty:.6f} {tz:.6f} 1 gallery_{i:03d}.jpg\n"
        )
        lines.append("0.0 0.0 -1\n")  # dummy points2D line (required by parser)

    (colmap_dir / "images.txt").write_text("".join(lines))

    (colmap_dir / "points3D.txt").write_text(
        "# 3D point list (empty for synthetic data)\n"
    )


def build_gallery_scene(
    colmap_dir: Path,
    output_path: Path,
    ply_path: str | None = None,
) -> Path:
    """Build the gallery USD scene.

    Args:
        colmap_dir: Directory containing COLMAP cameras.txt, images.txt.
        output_path: Where to write the .usda file.
        ply_path: Optional path to a trained Gaussian .ply for reference.

    Returns:
        The output path.
    """
    transformer = CoordinateTransformer.from_colmap_dir(colmap_dir)
    print(f"Loaded {len(transformer.cameras)} camera(s), {len(transformer.images)} image(s)")

    assembler = UsdSceneAssembler(up_axis="Y", meters_per_unit=1.0)
    assembler.set_colmap_cameras(transformer)
    assembler.set_metadata("project", "gallery")
    assembler.set_metadata("source", "colmap")
    assembler.set_metadata("colmap_dir", str(colmap_dir))

    # Add gallery room as a placeholder object at origin
    assembler.add_object(ObjectDescriptor(
        name="gallery_room",
        centroid=(0.0, 0.0, 0.0),
        diffuse_color=(0.9, 0.9, 0.85),
        metadata={"category": "environment", "description": "Gallery room envelope"},
    ))

    # If a PLY path is provided, reference it as a gaussian object
    if ply_path:
        assembler.add_object(ObjectDescriptor(
            name="gaussian_splat",
            gaussian_usd_path=ply_path,
            centroid=(0.0, 0.0, 0.0),
            metadata={"category": "gaussian", "source_ply": ply_path},
        ))

    # Add some gallery wall art placeholders at cardinal positions
    wall_pieces = [
        ("painting_north", (0.0, 1.5, -4.5), (0.7, 0.3, 0.2)),
        ("painting_south", (0.0, 1.5, 4.5), (0.2, 0.5, 0.7)),
        ("painting_east", (4.5, 1.5, 0.0), (0.3, 0.7, 0.3)),
        ("painting_west", (-4.5, 1.5, 0.0), (0.6, 0.6, 0.2)),
    ]
    for name, centroid, color in wall_pieces:
        assembler.add_object(ObjectDescriptor(
            name=name,
            centroid=centroid,
            diffuse_color=color,
            scale=(2.0, 1.5, 0.05),
            metadata={"category": "artwork"},
        ))

    stage = assembler.write(output_path)
    print(f"Wrote gallery scene to {output_path}")

    # Print hierarchy summary
    from pxr import Usd, UsdGeom

    prim_count = sum(1 for _ in stage.Traverse())
    cam_count = sum(1 for p in stage.Traverse() if p.GetTypeName() == "Camera")
    print(f"  Prims: {prim_count}, Cameras: {cam_count}")
    print(f"  Up axis: {UsdGeom.GetStageUpAxis(stage)}")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble gallery USD scene")
    parser.add_argument(
        "--colmap-dir",
        type=Path,
        default=_PROJECT_ROOT / "colmap" / "exported",
        help="Path to COLMAP sparse reconstruction directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "test-data" / "gallery_output" / "scene" / "gallery_scene.usda",
        help="Output .usda file path",
    )
    parser.add_argument(
        "--ply-path",
        type=str,
        default=None,
        help="Optional path to trained Gaussian .ply file",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic COLMAP data instead of reading from disk",
    )
    args = parser.parse_args()

    colmap_dir = args.colmap_dir

    # If the COLMAP directory doesn't exist or --synthetic, generate test data
    if args.synthetic or not (colmap_dir / "cameras.txt").exists():
        if not args.synthetic:
            print(f"COLMAP dir {colmap_dir} not found, generating synthetic data...")
        else:
            print("Generating synthetic COLMAP data...")
        synth_dir = colmap_dir if colmap_dir.parent.exists() else Path(tempfile.mkdtemp(prefix="gallery_colmap_"))
        synth_dir.mkdir(parents=True, exist_ok=True)
        _write_synthetic_colmap(synth_dir)
        colmap_dir = synth_dir

    build_gallery_scene(colmap_dir, args.output, args.ply_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
