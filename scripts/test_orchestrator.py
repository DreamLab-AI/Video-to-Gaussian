#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test suite for the pipeline stages, MCP client, quality gates, and config.

Run from repo root:
    python3 scripts/test_orchestrator.py

No running MCP server is required -- HTTP calls are intercepted.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.config import PipelineConfig, IngestConfig, TrainingConfig
from pipeline.stages import PipelineStages, StageResult, STAGE_NAMES
from pipeline.mcp_client import McpClient, McpError, McpConnectionError
from pipeline.quality_gates import (
    FrameStats,
    TrainingMetrics,
    MeshMetrics,
    RoundTripMetrics,
    FinalMetrics,
    QualityVerdict,
    assess_input_quality,
    assess_training_quality,
    assess_mesh_quality,
    assess_roundtrip_quality,
    assess_final_quality,
)


# =========================================================================
# Config tests
# =========================================================================

class TestPipelineConfig(unittest.TestCase):

    def test_defaults_are_valid(self):
        config = PipelineConfig()
        errors = config.validate()
        self.assertEqual(errors, [])

    def test_invalid_fps(self):
        config = PipelineConfig()
        config.ingest.fps = -1
        errors = config.validate()
        self.assertTrue(any("fps" in e for e in errors))

    def test_invalid_iterations(self):
        config = PipelineConfig()
        config.training.max_iterations = 100
        errors = config.validate()
        self.assertTrue(any("max_iterations" in e for e in errors))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            original = PipelineConfig()
            original.ingest.fps = 2.5
            original.training.strategy = "default"
            original.save(path)

            loaded = PipelineConfig.load(path)
            self.assertEqual(loaded.ingest.fps, 2.5)
            self.assertEqual(loaded.training.strategy, "default")

    def test_to_dict_roundtrip(self):
        config = PipelineConfig()
        d = config.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("ingest", d)
        self.assertIn("training", d)


# =========================================================================
# Stage-based pipeline tests
# =========================================================================

class TestPipelineStages(unittest.TestCase):

    def test_stage_names_complete(self):
        """All stage names are defined."""
        self.assertEqual(len(STAGE_NAMES), 11)
        self.assertIn("ingest", STAGE_NAMES)
        self.assertIn("validate", STAGE_NAMES)
        self.assertIn("train", STAGE_NAMES)

    def test_init_creates_instance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = PipelineStages(tmpdir)
            self.assertEqual(str(p.job_dir), tmpdir)
            self.assertIsInstance(p.config, PipelineConfig)

    def test_ingest_missing_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = PipelineStages(tmpdir)
            result = p.ingest("/nonexistent/video.mp4")
            self.assertFalse(result.success)
            self.assertEqual(result.stage, "ingest")
            self.assertIn("not found", result.error)

    def test_reconstruct_missing_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = PipelineStages(tmpdir)
            result = p.reconstruct("/nonexistent/frames")
            self.assertFalse(result.success)
            self.assertEqual(result.stage, "reconstruct")

    def test_train_missing_colmap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = PipelineStages(tmpdir)
            result = p.train("/nonexistent/colmap")
            self.assertFalse(result.success)
            self.assertEqual(result.stage, "train")

    def test_validate_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = PipelineStages(tmpdir)
            result = p.validate()
            self.assertFalse(result.success)
            self.assertEqual(result.stage, "validate")

    def test_status_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = PipelineStages(tmpdir)
            status = p.status()
            self.assertIn("stages_completed", status)
            self.assertEqual(len(status["stages_completed"]), 0)

    def test_stage_result_to_dict(self):
        result = StageResult(success=True, stage="test", metrics={"x": 1})
        d = result.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["stage"], "test")
        self.assertEqual(d["metrics"]["x"], 1)


# =========================================================================
# MCP client tests (no server required)
# =========================================================================

class TestMcpClient(unittest.TestCase):

    def test_build_request_structure(self):
        client = McpClient()
        req = client._build_request("tools/call", {"name": "training.get_state"})
        self.assertEqual(req["jsonrpc"], "2.0")
        self.assertEqual(req["method"], "tools/call")
        self.assertIn("id", req)
        self.assertIn("params", req)

    def test_request_ids_increment(self):
        client = McpClient()
        r1 = client._build_request("ping")
        r2 = client._build_request("ping")
        self.assertEqual(r2["id"], r1["id"] + 1)

    def test_call_tool_builds_correct_payload(self):
        client = McpClient()
        captured = {}

        def mock_post(payload, timeout=None):
            captured.update(payload)
            return {"success": True}

        client._post = mock_post
        client.call_tool("scene.load_dataset", {"path": "/data/colmap"})

        self.assertEqual(captured["method"], "tools/call")
        self.assertEqual(captured["params"]["name"], "scene.load_dataset")
        self.assertEqual(captured["params"]["arguments"]["path"], "/data/colmap")

    def test_load_dataset_arguments(self):
        client = McpClient()
        captured = {}

        def mock_post(payload, timeout=None):
            captured.update(payload)
            return {"success": True, "num_gaussians": 100000}

        client._post = mock_post
        client.load_dataset("/my/dataset", images_folder="img", max_iterations=10000, strategy="default")

        args = captured["params"]["arguments"]
        self.assertEqual(args["path"], "/my/dataset")
        self.assertEqual(args["images_folder"], "img")
        self.assertEqual(args["max_iterations"], 10000)
        self.assertEqual(args["strategy"], "default")

    def test_training_get_state_parsing(self):
        client = McpClient()

        def mock_post(payload, timeout=None):
            return {
                "running": False,
                "iteration": 30000,
                "max_iterations": 30000,
                "loss": 0.0012,
                "psnr": 28.5,
                "ssim": 0.92,
                "elapsed_s": 120.0,
                "num_gaussians": 500000,
            }

        client._post = mock_post
        state = client.training_get_state()
        self.assertFalse(state.running)
        self.assertEqual(state.iteration, 30000)
        self.assertAlmostEqual(state.psnr, 28.5)
        self.assertAlmostEqual(state.ssim, 0.92)

    def test_render_capture_arguments(self):
        client = McpClient()
        captured = {}

        def mock_post(payload, timeout=None):
            captured.update(payload)
            return {"path": "/tmp/render.png", "width": 1920, "height": 1080, "format": "png"}

        client._post = mock_post
        result = client.render_capture(width=1920, height=1080, output_path="/tmp/render.png")

        args = captured["params"]["arguments"]
        self.assertEqual(args["width"], 1920)
        self.assertEqual(args["output_path"], "/tmp/render.png")
        self.assertEqual(result.path, "/tmp/render.png")

    def test_selection_by_description(self):
        client = McpClient()

        def mock_post(payload, timeout=None):
            return {"count": 5000}

        client._post = mock_post
        result = client.selection_by_description("the red car")
        self.assertEqual(result.count, 5000)
        self.assertEqual(result.description, "the red car")

    def test_connection_error_raised(self):
        client = McpClient(
            endpoint="http://127.0.0.1:19999/mcp",
            max_retries=1, retry_delay=0.01,
        )
        self.assertFalse(client.ping())
        with self.assertRaises(McpConnectionError):
            client.call_tool("training.get_state")


# =========================================================================
# Quality gate tests
# =========================================================================

class TestInputQuality(unittest.TestCase):

    def test_good_input(self):
        config = PipelineConfig()
        stats = FrameStats(
            frame_count=100,
            blur_scores=[200.0] * 100,
            exposure_values=[0.5] * 100,
            coverage_score=0.8,
        )
        result = assess_input_quality(stats, config)
        self.assertEqual(result.verdict, QualityVerdict.PASS)

    def test_too_few_frames(self):
        config = PipelineConfig()
        stats = FrameStats(frame_count=5)
        result = assess_input_quality(stats, config)
        self.assertEqual(result.verdict, QualityVerdict.FAIL)

    def test_blurry_frames_warn(self):
        config = PipelineConfig()
        config.ingest.blur_threshold = 100.0
        stats = FrameStats(
            frame_count=50,
            blur_scores=[50.0] * 30 + [200.0] * 20,
            coverage_score=0.5,
        )
        result = assess_input_quality(stats, config)
        self.assertIn(result.verdict, (QualityVerdict.WARN, QualityVerdict.FAIL))


class TestTrainingQuality(unittest.TestCase):

    def test_good_training(self):
        config = PipelineConfig()
        metrics = TrainingMetrics(
            psnr=28.0, ssim=0.9, final_loss=0.001,
            loss_history=[0.01 - 0.001 * i for i in range(600)],
            iterations_completed=30000, max_iterations=30000,
            num_gaussians=500000,
        )
        result = assess_training_quality(metrics, config)
        self.assertTrue(result.passed)

    def test_low_psnr_fails(self):
        config = PipelineConfig()
        config.quality.gate1_min_psnr = 20.0
        metrics = TrainingMetrics(psnr=10.0, ssim=0.5, final_loss=0.1)
        result = assess_training_quality(metrics, config)
        self.assertEqual(result.verdict, QualityVerdict.FAIL)


class TestMeshQuality(unittest.TestCase):

    def test_good_mesh(self):
        config = PipelineConfig()
        metrics = MeshMetrics(
            vertex_count=10000, face_count=20000,
            is_watertight=True, normal_consistency=0.95,
        )
        result = assess_mesh_quality(metrics, config)
        self.assertTrue(result.passed)

    def test_too_few_vertices(self):
        config = PipelineConfig()
        metrics = MeshMetrics(vertex_count=10, normal_consistency=0.1)
        result = assess_mesh_quality(metrics, config)
        self.assertEqual(result.verdict, QualityVerdict.FAIL)


class TestRoundTripQuality(unittest.TestCase):

    def test_good_roundtrip(self):
        config = PipelineConfig()
        metrics = RoundTripMetrics(
            original_psnr=28.0, roundtrip_psnr=25.0,
        )
        result = assess_roundtrip_quality(metrics, config)
        self.assertTrue(result.passed)

    def test_bad_roundtrip(self):
        config = PipelineConfig()
        config.quality.gate2_roundtrip_psnr = 18.0
        metrics = RoundTripMetrics(
            original_psnr=28.0, roundtrip_psnr=8.0,
        )
        result = assess_roundtrip_quality(metrics, config)
        self.assertEqual(result.verdict, QualityVerdict.FAIL)


class TestFinalQuality(unittest.TestCase):

    def test_good_final(self):
        config = PipelineConfig()
        metrics = FinalMetrics(
            render_psnr=25.0, object_count=3,
            total_vertices=50000, usd_file_size_mb=5.0,
            has_materials=True,
        )
        result = assess_final_quality(metrics, config)
        self.assertTrue(result.passed)

    def test_no_objects(self):
        config = PipelineConfig()
        metrics = FinalMetrics(render_psnr=25.0, object_count=0)
        result = assess_final_quality(metrics, config)
        self.assertEqual(result.verdict, QualityVerdict.FAIL)


# =========================================================================
# Run
# =========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
