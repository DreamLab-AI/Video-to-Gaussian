"""Mesh post-processing and cleanup utilities.

Provides decimation, smoothing, hole filling, component removal,
and watertight validation for extracted meshes.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


class MeshCleaner:
    """Clean, simplify, and validate triangle meshes.

    Operations:
    - Remove small disconnected components
    - Laplacian smoothing
    - Decimation to target face count
    - Hole filling
    - Normal recomputation
    - Watertight validation
    """

    def clean(
        self,
        mesh: trimesh.Trimesh,
        target_faces: int = 50000,
        smooth_iterations: int = 3,
        smooth_lambda: float = 0.5,
        min_component_ratio: float = 0.01,
        min_component_faces: int = 100,
        fill_holes: bool = True,
        max_hole_edges: int = 100,
    ) -> trimesh.Trimesh:
        """Run the full cleaning pipeline on a mesh.

        Args:
            mesh: Input mesh to clean.
            target_faces: Target number of faces after decimation.
            smooth_iterations: Number of Laplacian smoothing passes.
            smooth_lambda: Smoothing strength per iteration (0-1).
            min_component_ratio: Remove components with fewer faces
                than this fraction of the largest component.
            min_component_faces: Absolute minimum face count for a
                component to be kept. Components below this are always
                removed regardless of the ratio threshold.
            fill_holes: Whether to attempt hole filling.
            max_hole_edges: Maximum boundary loop size to fill.

        Returns:
            Cleaned trimesh.Trimesh.
        """
        logger.info("Cleaning mesh: %d verts, %d faces",
                     len(mesh.vertices), len(mesh.faces))

        # Step 1: Remove degenerate faces
        mesh = self.remove_degenerate_faces(mesh)

        # Step 2: Remove small disconnected components
        mesh = self.remove_small_components(
            mesh, min_ratio=min_component_ratio,
            min_faces=min_component_faces,
        )

        # Step 3: Fill holes
        if fill_holes:
            mesh = self.fill_holes(mesh, max_hole_edges)

        # Step 4: Laplacian smoothing
        if smooth_iterations > 0:
            mesh = self.laplacian_smooth(mesh, smooth_iterations, smooth_lambda)

        # Step 5: Decimate to target face count
        if len(mesh.faces) > target_faces:
            mesh = self.decimate(mesh, target_faces)

        # Step 6: Recompute normals
        mesh = self.recompute_normals(mesh)

        # Step 7: Watertight check
        is_watertight = self.check_watertight(mesh)
        logger.info("Final mesh: %d verts, %d faces, watertight=%s",
                     len(mesh.vertices), len(mesh.faces), is_watertight)

        return mesh

    def remove_degenerate_faces(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Remove zero-area and duplicate faces."""
        # Filter out degenerate (zero-area) faces
        valid_mask = mesh.nondegenerate_faces()
        if not valid_mask.all():
            mesh.update_faces(valid_mask)
            logger.debug("Removed %d degenerate faces", (~valid_mask).sum())

        # Remove duplicate faces
        unique_mask = mesh.unique_faces()
        if not unique_mask.all():
            mesh.update_faces(unique_mask)
            logger.debug("Removed %d duplicate faces", (~unique_mask).sum())
        mesh.remove_unreferenced_vertices()
        logger.debug("After degenerate removal: %d faces", len(mesh.faces))
        return mesh

    def remove_small_components(
        self,
        mesh: trimesh.Trimesh,
        min_ratio: float = 0.01,
        min_faces: int = 100,
    ) -> trimesh.Trimesh:
        """Remove disconnected components that are too small.

        A component is removed if it has fewer faces than EITHER:
        - ``min_faces`` (absolute floor), or
        - ``min_ratio * largest_component_face_count`` (relative threshold)

        The effective threshold is the maximum of these two values.

        Args:
            mesh: Input mesh.
            min_ratio: Minimum fraction of the largest component's face count.
            min_faces: Absolute minimum number of faces a component must have.

        Returns:
            Mesh with only significant components retained.
        """
        if len(mesh.faces) == 0:
            return mesh

        components = mesh.split(only_watertight=False)
        n_total = len(components)
        if n_total <= 1:
            return mesh

        # Sort by face count descending
        components.sort(key=lambda m: len(m.faces), reverse=True)
        largest_count = len(components[0].faces)
        ratio_threshold = int(largest_count * min_ratio)
        threshold = max(ratio_threshold, min_faces)

        kept = [c for c in components if len(c.faces) >= threshold]
        removed = n_total - len(kept)
        removed_faces = sum(len(c.faces) for c in components if len(c.faces) < threshold)

        logger.info(
            "Component filter: %d total components, kept %d, "
            "removed %d (threshold=%d faces, ratio_threshold=%d, "
            "abs_min=%d, removed_faces=%d)",
            n_total, len(kept), removed, threshold,
            ratio_threshold, min_faces, removed_faces,
        )

        if len(kept) == 0:
            logger.warning("All components below threshold; keeping largest")
            return components[0]

        if len(kept) == 1:
            return kept[0]

        return trimesh.util.concatenate(kept)

    def laplacian_smooth(
        self,
        mesh: trimesh.Trimesh,
        iterations: int = 3,
        lamb: float = 0.5,
    ) -> trimesh.Trimesh:
        """Apply Laplacian smoothing to mesh vertices.

        Uses Taubin smoothing (alternating positive/negative lambda)
        to reduce shrinkage.

        Args:
            mesh: Input mesh.
            iterations: Number of smoothing passes.
            lamb: Smoothing factor (0-1). Higher = more smoothing.

        Returns:
            Smoothed mesh.
        """
        vertices = mesh.vertices.copy()
        faces = mesh.faces

        # Build adjacency: for each vertex, its neighbors
        n_verts = len(vertices)
        adjacency = [set() for _ in range(n_verts)]
        for f in faces:
            adjacency[f[0]].update([f[1], f[2]])
            adjacency[f[1]].update([f[0], f[2]])
            adjacency[f[2]].update([f[0], f[1]])

        mu = -lamb / (1.0 - 0.1 * lamb)  # Taubin's mu parameter

        for iteration in range(iterations):
            # Forward pass (shrink)
            new_verts = vertices.copy()
            for vi in range(n_verts):
                neighbors = list(adjacency[vi])
                if not neighbors:
                    continue
                neighbor_mean = vertices[neighbors].mean(axis=0)
                new_verts[vi] = vertices[vi] + lamb * (neighbor_mean - vertices[vi])
            vertices = new_verts

            # Backward pass (inflate) -- Taubin smoothing
            new_verts = vertices.copy()
            for vi in range(n_verts):
                neighbors = list(adjacency[vi])
                if not neighbors:
                    continue
                neighbor_mean = vertices[neighbors].mean(axis=0)
                new_verts[vi] = vertices[vi] + mu * (neighbor_mean - vertices[vi])
            vertices = new_verts

        result = mesh.copy()
        result.vertices = vertices
        logger.debug("Laplacian smoothing: %d iterations", iterations)
        return result

    def decimate(
        self,
        mesh: trimesh.Trimesh,
        target_faces: int,
    ) -> trimesh.Trimesh:
        """Decimate mesh to target face count, preserving vertex colors.

        Uses quadric error metrics via fast_simplification if available,
        otherwise falls back to vertex clustering. Both paths transfer
        vertex colors from the original mesh via nearest-vertex lookup.

        Args:
            mesh: Input mesh.
            target_faces: Desired number of faces.

        Returns:
            Decimated mesh with vertex colors preserved.
        """
        if len(mesh.faces) <= target_faces:
            return mesh

        logger.info("Decimating from %d to %d faces", len(mesh.faces), target_faces)

        # Capture vertex colors before decimation
        has_colors = (
            mesh.visual is not None
            and hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) == len(mesh.vertices)
        )
        original_colors = None
        if has_colors:
            original_colors = np.array(mesh.visual.vertex_colors)

        # Try fast_simplification directly with target_reduction
        try:
            import fast_simplification
            target_reduction = 1.0 - (target_faces / len(mesh.faces))
            target_reduction = max(0.01, min(0.99, target_reduction))
            verts_out, faces_out = fast_simplification.simplify(
                mesh.vertices.view(np.ndarray).copy(),
                mesh.faces.view(np.ndarray).copy(),
                target_reduction=target_reduction,
            )
            decimated = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=True)
            if len(decimated.faces) > 0:
                logger.info("Quadric decimation result: %d faces", len(decimated.faces))
                if original_colors is not None:
                    decimated = self._transfer_vertex_colors(
                        mesh.vertices, original_colors, decimated,
                    )
                return decimated
        except Exception as e:
            logger.debug("fast_simplification failed: %s", e)

        # Fallback: vertex clustering decimation
        decimated = self._vertex_clustering_decimate(mesh, target_faces)
        if original_colors is not None:
            decimated = self._transfer_vertex_colors(
                mesh.vertices, original_colors, decimated,
            )
        return decimated

    def _vertex_clustering_decimate(
        self,
        mesh: trimesh.Trimesh,
        target_faces: int,
    ) -> trimesh.Trimesh:
        """Simple vertex clustering decimation as fallback.

        Merges nearby vertices into clusters, reducing mesh complexity.
        """
        current_faces = len(mesh.faces)
        ratio = target_faces / max(current_faces, 1)
        # Estimate voxel size from ratio: roughly cube-root relationship
        bounds = mesh.bounds
        diagonal = np.linalg.norm(bounds[1] - bounds[0])
        voxel_size = diagonal * (1.0 - ratio ** 0.33) * 0.1

        if voxel_size <= 0:
            return mesh

        vertices = mesh.vertices
        # Quantize vertices to grid
        quantized = np.round(vertices / voxel_size).astype(np.int64)

        # Map quantized coords to cluster IDs
        unique_coords, inverse = np.unique(quantized, axis=0, return_inverse=True)
        n_clusters = len(unique_coords)

        # Compute cluster centers as mean of member vertices
        cluster_centers = np.zeros((n_clusters, 3), dtype=np.float64)
        cluster_counts = np.zeros(n_clusters, dtype=np.float64)
        np.add.at(cluster_centers, inverse, vertices)
        np.add.at(cluster_counts, inverse, 1.0)
        cluster_centers /= cluster_counts[:, np.newaxis].clip(1)

        # Remap faces
        new_faces = inverse[mesh.faces]
        # Remove degenerate faces (where two or more vertices merged)
        valid = (new_faces[:, 0] != new_faces[:, 1]) & \
                (new_faces[:, 1] != new_faces[:, 2]) & \
                (new_faces[:, 0] != new_faces[:, 2])
        new_faces = new_faces[valid]

        result = trimesh.Trimesh(vertices=cluster_centers, faces=new_faces, process=True)
        logger.info("Vertex clustering decimation: %d -> %d faces", current_faces, len(result.faces))
        return result

    @staticmethod
    def _transfer_vertex_colors(
        src_vertices: np.ndarray,
        src_colors: np.ndarray,
        dst_mesh: trimesh.Trimesh,
    ) -> trimesh.Trimesh:
        """Transfer vertex colors from source vertices to destination mesh.

        Uses nearest-neighbor lookup (KDTree) to map colors from the
        original vertex positions to the decimated/repaired vertex positions.

        Args:
            src_vertices: Original vertex positions (N, 3).
            src_colors: Original vertex colors (N, 3|4) uint8.
            dst_mesh: Target mesh to receive colors.

        Returns:
            dst_mesh with vertex_colors applied.
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(src_vertices)
        _, indices = tree.query(dst_mesh.vertices, k=1)
        transferred = src_colors[indices]

        # Ensure RGBA
        if transferred.ndim == 2 and transferred.shape[1] == 3:
            alpha = np.full((len(transferred), 1), 255, dtype=np.uint8)
            transferred = np.hstack([transferred, alpha])

        dst_mesh.visual.vertex_colors = transferred
        logger.debug("Transferred vertex colors to %d vertices", len(dst_mesh.vertices))
        return dst_mesh

    def fill_holes(
        self,
        mesh: trimesh.Trimesh,
        max_hole_edges: int = 100,
    ) -> trimesh.Trimesh:
        """Fill small holes in the mesh.

        Uses pymeshfix for robust hole filling if available.

        Args:
            mesh: Input mesh.
            max_hole_edges: Maximum boundary loop length to fill.

        Returns:
            Mesh with small holes filled.
        """
        try:
            import pymeshfix

            # Capture vertex colors before repair (pymeshfix drops them)
            has_colors = (
                mesh.visual is not None
                and hasattr(mesh.visual, "vertex_colors")
                and mesh.visual.vertex_colors is not None
                and len(mesh.visual.vertex_colors) == len(mesh.vertices)
            )
            original_verts = mesh.vertices.copy() if has_colors else None
            original_colors = np.array(mesh.visual.vertex_colors) if has_colors else None

            meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            meshfix.repair(verbose=False)
            repaired = trimesh.Trimesh(
                vertices=meshfix.v, faces=meshfix.f, process=True,
            )
            logger.info("pymeshfix repair: %d -> %d faces",
                         len(mesh.faces), len(repaired.faces))

            # Transfer vertex colors to repaired mesh
            if original_colors is not None and len(repaired.vertices) > 0:
                repaired = self._transfer_vertex_colors(
                    original_verts, original_colors, repaired,
                )

            return repaired
        except ImportError:
            logger.debug("pymeshfix not available, using trimesh fill_holes")
        except Exception as e:
            logger.warning("pymeshfix failed: %s, falling back", e)

        # Fallback: trimesh's built-in hole filling
        mesh.fill_holes()
        return mesh

    def recompute_normals(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Recompute face and vertex normals, fixing winding order.

        Returns:
            Mesh with consistent normals.
        """
        mesh.fix_normals()
        return mesh

    def check_watertight(self, mesh: trimesh.Trimesh) -> bool:
        """Check if the mesh is watertight (closed manifold).

        Returns:
            True if the mesh is watertight.
        """
        is_watertight = mesh.is_watertight
        if not is_watertight:
            # Log some diagnostics
            edges = mesh.edges_sorted
            edge_counts = {}
            for e in map(tuple, edges):
                edge_counts[e] = edge_counts.get(e, 0) + 1
            boundary = sum(1 for c in edge_counts.values() if c == 1)
            logger.debug("Mesh is not watertight: %d boundary edges", boundary)
        return is_watertight

    def make_watertight(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Attempt to make mesh watertight by filling holes and fixing normals.

        Returns:
            Repaired mesh (may still not be watertight for complex topology).
        """
        mesh = self.fill_holes(mesh)
        mesh = self.recompute_normals(mesh)
        if not mesh.is_watertight:
            logger.warning("Mesh could not be made fully watertight")
        return mesh
