#!/usr/bin/env python3
"""
Compare SAM3D vs Tripo 3D reconstruction quality.

Usage:
    python scripts/test_3d_reconstruction.py [--image path/to/image.png] [--output-dir outputs/3d_compare]

Runs both backends on the same input image, then prints a comparison table
of vertex count, face count, bounding box, texture presence, and timing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline.sam3d_client import SAM3DClient, ReconstructionResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_3d_reconstruction")


def find_sample_image() -> Optional[Path]:
    """Find a sample image from gallery frames or test data."""
    candidates = [
        PROJECT_ROOT / "outputs" / "gallery" / "frames",
        PROJECT_ROOT / "outputs" / "frames",
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "data" / "images",
        PROJECT_ROOT / "tests" / "data",
    ]
    for d in candidates:
        if not d.exists():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            images = sorted(d.glob(ext))
            if images:
                return images[0]
    return None


def compute_mesh_stats(result: ReconstructionResult) -> dict:
    """Compute detailed mesh statistics."""
    stats = {
        "backend": result.backend,
        "vertex_count": result.vertex_count,
        "face_count": result.face_count,
        "has_texture": result.has_texture,
        "duration_seconds": round(result.duration_seconds, 2),
        "filenames": result.filenames,
        "error": result.error,
    }

    mesh = result.mesh
    if mesh is not None:
        bounds = mesh.bounds
        extent = mesh.extents
        stats["bbox_min"] = bounds[0].tolist()
        stats["bbox_max"] = bounds[1].tolist()
        stats["extent"] = extent.tolist()
        stats["surface_area"] = round(float(mesh.area), 6) if hasattr(mesh, "area") else None
        stats["volume"] = round(float(mesh.volume), 6) if mesh.is_volume else None
        stats["is_watertight"] = bool(mesh.is_watertight)
        stats["euler_number"] = int(mesh.euler_number) if hasattr(mesh, "euler_number") else None

        if result.glb_data:
            stats["glb_size_kb"] = round(len(result.glb_data) / 1024, 1)
        if result.ply_data:
            stats["ply_size_kb"] = round(len(result.ply_data) / 1024, 1)
        if result.gaussian_ply_data:
            stats["gaussian_ply_size_kb"] = round(len(result.gaussian_ply_data) / 1024, 1)

    return stats


def format_comparison_table(sam3d_stats: dict, tripo_stats: dict) -> str:
    """Format a side-by-side comparison table."""
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("  3D RECONSTRUCTION COMPARISON: SAM3D vs Tripo")
    lines.append("=" * 72)
    lines.append(f"  {'Metric':<28} {'SAM3D':>18} {'Tripo':>18}")
    lines.append("-" * 72)

    def row(label, key, fmt=None):
        v1 = sam3d_stats.get(key, "N/A")
        v2 = tripo_stats.get(key, "N/A")
        if fmt and v1 != "N/A":
            v1 = fmt(v1)
        if fmt and v2 != "N/A":
            v2 = fmt(v2)
        lines.append(f"  {label:<28} {str(v1):>18} {str(v2):>18}")

    row("Vertices", "vertex_count", lambda v: f"{v:,}")
    row("Faces", "face_count", lambda v: f"{v:,}")
    row("Has Texture", "has_texture")
    row("Is Watertight", "is_watertight")
    row("Surface Area", "surface_area")
    row("Volume", "volume")
    row("Euler Number", "euler_number")
    row("GLB Size (KB)", "glb_size_kb")
    row("PLY Size (KB)", "ply_size_kb")
    row("Gaussian PLY (KB)", "gaussian_ply_size_kb")
    row("Duration (s)", "duration_seconds")
    row("Error", "error")

    lines.append("-" * 72)

    winner_vertices = "SAM3D" if sam3d_stats.get("vertex_count", 0) > tripo_stats.get("vertex_count", 0) else "Tripo"
    winner_speed = "SAM3D" if sam3d_stats.get("duration_seconds", 999) < tripo_stats.get("duration_seconds", 999) else "Tripo"

    lines.append(f"  Higher detail (vertices):    {winner_vertices}")
    lines.append(f"  Faster:                      {winner_speed}")
    lines.append("=" * 72)
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare SAM3D vs Tripo 3D reconstruction")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "3d_compare"))
    parser.add_argument("--api-url", type=str, default="http://192.168.2.48:3001")
    parser.add_argument("--comfyui-url", type=str, default="http://192.168.2.48:8189")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-steps", type=int, default=25, help="SAM3D inference steps (12=fast, 25=quality)")
    parser.add_argument("--texture-size", type=int, default=1024, help="SAM3D texture resolution")
    parser.add_argument("--skip-sam3d", action="store_true", help="Skip SAM3D backend")
    parser.add_argument("--skip-tripo", action="store_true", help="Skip Tripo backend")
    parser.add_argument("--timeout", type=int, default=600, help="API timeout in seconds")
    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
    else:
        image_path = find_sample_image()
        if image_path is None:
            logger.error("No sample image found. Provide one with --image.")
            sys.exit(1)

    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        sys.exit(1)

    logger.info("Input image: %s", image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = SAM3DClient(
        api_url=args.api_url,
        comfyui_url=args.comfyui_url,
        timeout=args.timeout,
    )

    health = client.health_check()
    logger.info("API health: %s", health)

    sam3d_stats = {"backend": "sam3d", "error": "skipped"}
    tripo_stats = {"backend": "tripo", "error": "skipped"}

    if not args.skip_sam3d:
        logger.info("--- Running SAM3D reconstruction ---")
        t0 = time.monotonic()
        try:
            sam3d_result = client.reconstruct_sam3d(
                image_path,
                seed=args.seed,
                quality_steps=args.quality_steps,
                texture_size=args.texture_size,
            )
            sam3d_stats = compute_mesh_stats(sam3d_result)
            saved = client.save_result(sam3d_result, output_dir / "sam3d", prefix="sam3d")
            logger.info("SAM3D files saved: %s", {k: str(v) for k, v in saved.items()})
        except Exception as e:
            logger.error("SAM3D failed: %s", e, exc_info=True)
            sam3d_stats = {"backend": "sam3d", "error": str(e), "duration_seconds": round(time.monotonic() - t0, 2)}

    if not args.skip_tripo:
        logger.info("--- Running Tripo reconstruction ---")
        t0 = time.monotonic()
        try:
            tripo_result = client.reconstruct_tripo(
                image_path,
                seed=args.seed,
            )
            tripo_stats = compute_mesh_stats(tripo_result)
            saved = client.save_result(tripo_result, output_dir / "tripo", prefix="tripo")
            logger.info("Tripo files saved: %s", {k: str(v) for k, v in saved.items()})
        except Exception as e:
            logger.error("Tripo failed: %s", e, exc_info=True)
            tripo_stats = {"backend": "tripo", "error": str(e), "duration_seconds": round(time.monotonic() - t0, 2)}

    comparison = format_comparison_table(sam3d_stats, tripo_stats)
    print(comparison)

    report = {
        "image": str(image_path),
        "sam3d": sam3d_stats,
        "tripo": tripo_stats,
    }
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()
