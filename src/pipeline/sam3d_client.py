"""
SAM3D reconstruction client for remote ComfyUI.

Supports two execution modes:
  1. Native ComfyUI API (port 8189): Upload image, submit prompt, poll history,
     download output files by path. Required for SAM3D workflows that produce
     file outputs (GLB/PLY) on the server filesystem.
  2. Salad API wrapper (port 3001): Single POST /prompt with base64 image input,
     synchronous response. Works for workflows with SaveImage output nodes.

SAM3D uses the native API because its nodes write files to disk and return
paths as STRING outputs, which the Salad API wrapper cannot capture.
"""

from __future__ import annotations

import base64
import json
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
import trimesh

logger = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).parent / "workflows"
SAM3D_WORKFLOW = WORKFLOW_DIR / "sam3d_object_reconstruct.json"
TRIPO_WORKFLOW = WORKFLOW_DIR / "tripo_object_reconstruct.json"


@dataclass
class ReconstructionResult:
    """Container for 3D reconstruction output."""

    mesh: Optional[trimesh.Trimesh] = None
    glb_data: Optional[bytes] = None
    ply_data: Optional[bytes] = None
    gaussian_ply_data: Optional[bytes] = None
    filenames: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    stats: dict = field(default_factory=dict)
    backend: str = "sam3d"
    error: Optional[str] = None
    output_paths: dict = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return len(self.mesh.vertices) if self.mesh is not None else 0

    @property
    def face_count(self) -> int:
        return len(self.mesh.faces) if self.mesh is not None else 0

    @property
    def has_texture(self) -> bool:
        if self.mesh is None:
            return False
        return self.mesh.visual is not None and hasattr(self.mesh.visual, "material")


class SAM3DClient:
    """Client for submitting 3D reconstruction workflows to remote ComfyUI."""

    def __init__(
        self,
        api_url: str = "http://192.168.2.48:3001",
        comfyui_url: str = "http://192.168.2.48:8189",
        timeout: int = 600,
        poll_interval: float = 2.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.comfyui_url = comfyui_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.session = requests.Session()

    def health_check(self) -> dict:
        """Check Salad API health status."""
        resp = self.session.get(f"{self.api_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def comfyui_health_check(self) -> bool:
        """Check if native ComfyUI is reachable."""
        try:
            resp = self.session.get(f"{self.comfyui_url}/system_stats", timeout=10)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _encode_image_base64(self, image_path: Path) -> str:
        """Encode image file to base64 data URI for Salad API."""
        data = image_path.read_bytes()
        suffix = image_path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"

    def _upload_image_to_comfyui(self, image_path: Path) -> str:
        """Upload image to ComfyUI input directory, return server-side filename."""
        image_path = Path(image_path)
        upload_url = f"{self.comfyui_url}/upload/image"
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            resp = requests.post(upload_url, files=files, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        return result.get("name", image_path.name)

    def _load_workflow(self, workflow_path: Path) -> dict:
        """Load a workflow JSON file."""
        with open(workflow_path) as f:
            return json.load(f)

    # ---------------------------------------------------------------
    # Native ComfyUI API methods (for SAM3D file-output workflows)
    # ---------------------------------------------------------------

    def _comfyui_submit(self, prompt: dict) -> str:
        """Submit prompt to native ComfyUI API, return prompt_id."""
        resp = self.session.post(
            f"{self.comfyui_url}/prompt",
            json={"prompt": prompt},
            timeout=30,
        )
        data = resp.json()
        if "error" in data:
            node_errors = data.get("node_errors", {})
            details = "; ".join(
                f"node {nid}: {e['errors']}" for nid, e in node_errors.items()
            ) if node_errors else data["error"].get("message", "unknown")
            raise RuntimeError(f"ComfyUI validation error: {details}")
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"No prompt_id in response: {data}")
        return prompt_id

    def _comfyui_poll(self, prompt_id: str) -> dict:
        """Poll ComfyUI history until prompt completes or times out."""
        deadline = time.monotonic() + self.timeout
        last_log = 0.0
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            resp = self.session.get(
                f"{self.comfyui_url}/history/{prompt_id}",
                timeout=10,
            )
            hist = resp.json()
            if prompt_id not in hist:
                if time.monotonic() - last_log > 30:
                    logger.info("Waiting for prompt %s...", prompt_id[:8])
                    last_log = time.monotonic()
                continue

            entry = hist[prompt_id]
            status = entry.get("status", {}).get("status_str", "unknown")

            if status == "success":
                return entry
            if status == "error":
                messages = entry.get("status", {}).get("messages", [])
                raise RuntimeError(f"ComfyUI execution error: {messages}")

            if time.monotonic() - last_log > 30:
                logger.info("Prompt %s status: %s", prompt_id[:8], status)
                last_log = time.monotonic()

        raise TimeoutError(f"Prompt {prompt_id} timed out after {self.timeout}s")

    def _comfyui_download_file(self, filepath: str) -> bytes:
        """Download a file from ComfyUI server by absolute path.

        ComfyUI doesn't have a direct file download endpoint for arbitrary paths,
        but SAM3D nodes write to the output directory. We try the /view endpoint
        with type=output, or fall back to reading via a helper workflow.
        """
        filename = Path(filepath).name
        subfolder = str(Path(filepath).parent)

        # Try ComfyUI /view endpoint (works for output/ directory files)
        for file_type in ("output", "input", "temp"):
            resp = self.session.get(
                f"{self.comfyui_url}/view",
                params={"filename": filename, "subfolder": subfolder, "type": file_type},
                timeout=30,
            )
            if resp.status_code == 200 and len(resp.content) > 0:
                return resp.content

        # Try with just the filename in common subdirectories
        for sub in ("", "sam3d", "sam3d_output"):
            resp = self.session.get(
                f"{self.comfyui_url}/view",
                params={"filename": filename, "subfolder": sub, "type": "output"},
                timeout=30,
            )
            if resp.status_code == 200 and len(resp.content) > 0:
                return resp.content

        raise FileNotFoundError(
            f"Could not download {filepath} from ComfyUI. "
            f"Tried /view with filename={filename}, subfolder={subfolder}"
        )

    def _extract_paths_from_history(self, history_entry: dict) -> dict[str, str]:
        """Extract file paths from ComfyUI history output nodes.

        PreviewAny nodes output text strings containing file paths.
        SaveImage nodes output image file references.
        """
        outputs = history_entry.get("outputs", {})
        paths = {}

        for node_id, node_output in outputs.items():
            # PreviewAny outputs text containing the file path
            if "text" in node_output:
                text_items = node_output["text"]
                if isinstance(text_items, list):
                    for item in text_items:
                        if isinstance(item, str) and (
                            item.endswith(".glb") or item.endswith(".ply") or item.endswith(".pt")
                        ):
                            ext = Path(item).suffix.lower()
                            key = f"node_{node_id}_{ext.lstrip('.')}"
                            paths[key] = item
                elif isinstance(text_items, str) and (
                    text_items.endswith(".glb") or text_items.endswith(".ply")
                ):
                    ext = Path(text_items).suffix.lower()
                    paths[f"node_{node_id}_{ext.lstrip('.')}"] = text_items

            # SaveImage outputs
            if "images" in node_output:
                for img_info in node_output["images"]:
                    if isinstance(img_info, dict):
                        fname = img_info.get("filename", "")
                        subfolder = img_info.get("subfolder", "")
                        paths[f"node_{node_id}_image"] = f"{subfolder}/{fname}" if subfolder else fname

        return paths

    def _build_sam3d_prompt(
        self,
        image_filename: str,
        seed: int = 42,
        quality_steps: int = 25,
        texture_size: int = 1024,
        texture_mode: str = "opt",
        simplify: float = 0.90,
        postprocess: bool = True,
    ) -> dict:
        """Build the SAM3D ComfyUI API prompt with the given parameters."""
        prompt = self._load_workflow(SAM3D_WORKFLOW)

        prompt["2"]["inputs"]["image"] = image_filename

        prompt["4"]["inputs"]["seed"] = seed
        prompt["4"]["inputs"]["stage1_steps"] = quality_steps
        prompt["4"]["inputs"]["stage2_steps"] = quality_steps

        prompt["6"]["inputs"]["with_postprocess"] = postprocess
        prompt["6"]["inputs"]["simplify"] = simplify

        prompt["7"]["inputs"]["texture_mode"] = texture_mode
        prompt["7"]["inputs"]["texture_size"] = texture_size

        return prompt

    def _build_tripo_prompt(
        self,
        image_filename: str,
        seed: int = 42,
        texture_quality: str = "detailed",
        geometry_quality: str = "detailed",
        face_limit: int = -1,
    ) -> dict:
        """Build the Tripo ComfyUI API prompt with the given parameters."""
        prompt = self._load_workflow(TRIPO_WORKFLOW)

        prompt["1"]["inputs"]["image"] = image_filename

        prompt["2"]["inputs"]["model_seed"] = seed
        prompt["2"]["inputs"]["texture_seed"] = seed
        prompt["2"]["inputs"]["texture_quality"] = texture_quality
        prompt["2"]["inputs"]["geometry_quality"] = geometry_quality
        prompt["2"]["inputs"]["face_limit"] = face_limit

        return prompt

    # ---------------------------------------------------------------
    # Mesh loading helpers
    # ---------------------------------------------------------------

    def _load_glb(self, data: bytes) -> Optional[trimesh.Trimesh]:
        """Load a GLB binary into a trimesh object."""
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            scene = trimesh.load(tmp.name, file_type="glb", force="scene")

        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                return None
            if len(meshes) == 1:
                return meshes[0]
            return trimesh.util.concatenate(meshes)
        if isinstance(scene, trimesh.Trimesh):
            return scene
        return None

    def _load_ply(self, data: bytes) -> Optional[trimesh.Trimesh]:
        """Load a PLY binary into a trimesh object."""
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            mesh = trimesh.load(tmp.name, file_type="ply")
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
        return None

    # ---------------------------------------------------------------
    # Public reconstruction methods
    # ---------------------------------------------------------------

    def reconstruct_sam3d(
        self,
        image_path: str | Path,
        seed: int = 42,
        quality_steps: int = 25,
        texture_size: int = 1024,
        texture_mode: str = "opt",
        simplify: float = 0.90,
        postprocess: bool = True,
    ) -> ReconstructionResult:
        """
        Run SAM3D reconstruction via native ComfyUI API.

        Pipeline: LoadModel -> DepthEstimate -> GenerateSLAT ->
                  GaussianDecode + MeshDecode -> TextureBake

        Uses the native ComfyUI API (not Salad wrapper) because SAM3D nodes
        produce file outputs (GLB/PLY paths) that the Salad API cannot capture.

        Args:
            image_path: Path to input image (PNG/JPG).
            seed: Random seed for generation.
            quality_steps: Inference steps for both SLAT stages (12=fast, 25=quality).
            texture_size: Texture resolution (512-4096).
            texture_mode: "opt" (gradient descent, 30-60s) or "fast" (~5s).
            simplify: Mesh simplification ratio (0.5=gentle, 0.95=aggressive).
            postprocess: Apply mesh simplification and hole filling.

        Returns:
            ReconstructionResult with the reconstructed mesh.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info("SAM3D reconstruction: %s", image_path.name)
        t0 = time.monotonic()

        # Upload image to ComfyUI input directory
        image_filename = self._upload_image_to_comfyui(image_path)
        logger.info("Uploaded image as: %s", image_filename)

        # Build and submit prompt
        prompt = self._build_sam3d_prompt(
            image_filename=image_filename,
            seed=seed,
            quality_steps=quality_steps,
            texture_size=texture_size,
            texture_mode=texture_mode,
            simplify=simplify,
            postprocess=postprocess,
        )

        prompt_id = self._comfyui_submit(prompt)
        logger.info("Submitted prompt %s", prompt_id)

        # Poll for completion
        history_entry = self._comfyui_poll(prompt_id)
        elapsed = time.monotonic() - t0
        logger.info("SAM3D completed in %.1fs", elapsed)

        # Extract output file paths from history
        output_paths = self._extract_paths_from_history(history_entry)
        logger.info("Output paths: %s", output_paths)

        result = ReconstructionResult(
            backend="sam3d",
            duration_seconds=elapsed,
            output_paths=output_paths,
            filenames=list(output_paths.values()),
        )

        # Try to download and parse mesh files
        for key, filepath in output_paths.items():
            lower = filepath.lower()
            try:
                if lower.endswith(".glb"):
                    data = self._comfyui_download_file(filepath)
                    result.glb_data = data
                    if result.mesh is None:
                        result.mesh = self._load_glb(data)
                elif lower.endswith(".ply") and "gaussian" not in key:
                    data = self._comfyui_download_file(filepath)
                    result.ply_data = data
                    if result.mesh is None:
                        result.mesh = self._load_ply(data)
                elif lower.endswith(".ply") and "gaussian" in key:
                    data = self._comfyui_download_file(filepath)
                    result.gaussian_ply_data = data
            except (FileNotFoundError, requests.RequestException) as e:
                logger.warning("Could not download %s: %s", filepath, e)

        if result.mesh is not None:
            logger.info(
                "SAM3D result: %d vertices, %d faces, texture=%s",
                result.vertex_count,
                result.face_count,
                result.has_texture,
            )
        else:
            result.error = "No downloadable mesh data in outputs"
            logger.warning("SAM3D: %s. Paths: %s", result.error, output_paths)

        return result

    def reconstruct_tripo(
        self,
        image_path: str | Path,
        seed: int = 42,
        texture_quality: str = "detailed",
        geometry_quality: str = "detailed",
        face_limit: int = -1,
    ) -> ReconstructionResult:
        """
        Run Tripo reconstruction via native ComfyUI API (cloud API fallback).

        Args:
            image_path: Path to input image (PNG/JPG).
            seed: Random seed for generation.
            texture_quality: "standard" or "detailed".
            geometry_quality: "standard" or "detailed".
            face_limit: Max face count (-1 = no limit).

        Returns:
            ReconstructionResult with the reconstructed mesh.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info("Tripo reconstruction: %s", image_path.name)
        t0 = time.monotonic()

        image_filename = self._upload_image_to_comfyui(image_path)
        logger.info("Uploaded image as: %s", image_filename)

        prompt = self._build_tripo_prompt(
            image_filename=image_filename,
            seed=seed,
            texture_quality=texture_quality,
            geometry_quality=geometry_quality,
            face_limit=face_limit,
        )

        prompt_id = self._comfyui_submit(prompt)
        logger.info("Submitted Tripo prompt %s", prompt_id)

        history_entry = self._comfyui_poll(prompt_id)
        elapsed = time.monotonic() - t0
        logger.info("Tripo completed in %.1fs", elapsed)

        output_paths = self._extract_paths_from_history(history_entry)
        logger.info("Output paths: %s", output_paths)

        result = ReconstructionResult(
            backend="tripo",
            duration_seconds=elapsed,
            output_paths=output_paths,
            filenames=list(output_paths.values()),
        )

        # Tripo outputs model_file as a STRING path
        for key, filepath in output_paths.items():
            lower = filepath.lower()
            try:
                if lower.endswith(".glb") or lower.endswith(".obj") or lower.endswith(".fbx"):
                    data = self._comfyui_download_file(filepath)
                    result.glb_data = data
                    result.mesh = self._load_glb(data)
                elif lower.endswith(".ply"):
                    data = self._comfyui_download_file(filepath)
                    result.ply_data = data
                    result.mesh = self._load_ply(data)
            except (FileNotFoundError, requests.RequestException) as e:
                logger.warning("Could not download %s: %s", filepath, e)

        if result.mesh is not None:
            logger.info(
                "Tripo result: %d vertices, %d faces, texture=%s",
                result.vertex_count,
                result.face_count,
                result.has_texture,
            )
        else:
            result.error = "No downloadable mesh data in outputs"
            logger.warning("Tripo: %s. Paths: %s", result.error, output_paths)

        return result

    def reconstruct(
        self,
        image_path: str | Path,
        backend: str = "sam3d",
        fallback: bool = True,
        **kwargs,
    ) -> ReconstructionResult:
        """
        Run 3D reconstruction with optional fallback.

        Args:
            image_path: Path to input image.
            backend: Primary backend ("sam3d" or "tripo").
            fallback: If True, fall back to the other backend on failure.
            **kwargs: Passed to the chosen backend method.

        Returns:
            ReconstructionResult from whichever backend succeeded.
        """
        backends = {"sam3d": self.reconstruct_sam3d, "tripo": self.reconstruct_tripo}
        fallback_name = "tripo" if backend == "sam3d" else "sam3d"

        primary = backends[backend]
        secondary = backends[fallback_name]

        try:
            result = primary(image_path, **kwargs)
            if result.mesh is not None:
                return result
            if fallback:
                logger.warning(
                    "Primary backend %s returned no mesh, trying %s",
                    backend,
                    fallback_name,
                )
                return secondary(image_path, **kwargs)
            return result
        except Exception as e:
            logger.error("Primary backend %s failed: %s", backend, e)
            if fallback:
                logger.info("Falling back to %s", fallback_name)
                return secondary(image_path, **kwargs)
            return ReconstructionResult(backend=backend, error=str(e))

    def save_result(
        self,
        result: ReconstructionResult,
        output_dir: str | Path,
        prefix: str = "output",
    ) -> dict[str, Path]:
        """Save reconstruction results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = {}

        if result.glb_data:
            path = output_dir / f"{prefix}_textured.glb"
            path.write_bytes(result.glb_data)
            saved["glb"] = path

        if result.ply_data:
            path = output_dir / f"{prefix}_mesh.ply"
            path.write_bytes(result.ply_data)
            saved["ply"] = path

        if result.gaussian_ply_data:
            path = output_dir / f"{prefix}_gaussians.ply"
            path.write_bytes(result.gaussian_ply_data)
            saved["gaussian_ply"] = path

        if result.mesh is not None and "glb" not in saved:
            path = output_dir / f"{prefix}.glb"
            result.mesh.export(str(path), file_type="glb")
            saved["glb"] = path

        return saved
