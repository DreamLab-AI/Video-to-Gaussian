# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Background pipeline executor for web jobs.

Runs PipelineOrchestrator in a daemon thread per job, capturing logs
and updating job state through job_manager. Manages VRAM cleanup between
stages and captures hardware telemetry.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Active runners keyed by job_id
_runners: dict[str, PipelineRunner] = {}
_runners_lock = threading.Lock()


def free_vram() -> None:
    """Release GPU memory between pipeline stages."""
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("torch VRAM cleanup error: %s", exc)

    gc.collect()

    # Tell ComfyUI to free models
    try:
        import requests as req
        req.post(
            "http://localhost:8188/free",
            json={"unload_models": True, "free_memory": True},
            timeout=5,
        )
    except Exception:
        pass


def capture_hardware() -> dict[str, Any]:
    """Snapshot current GPU / RAM / disk utilization."""
    hw: dict[str, Any] = {"timestamp": time.time()}

    # GPU via torch
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.mem_get_info(0)
            total = mem[1] / (1024 ** 2)
            free = mem[0] / (1024 ** 2)
            hw["gpu_memory_used_mb"] = round(total - free, 1)
            hw["gpu_memory_total_mb"] = round(total, 1)
    except Exception:
        pass

    # GPU utilization via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            hw["gpu_utilization_pct"] = float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass

    # RAM via /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                mem[parts[0].rstrip(":")] = int(parts[1])
        total_kb = mem.get("MemTotal", 0)
        avail_kb = mem.get("MemAvailable", 0)
        hw["ram_total_mb"] = round(total_kb / 1024, 1)
        hw["ram_used_mb"] = round((total_kb - avail_kb) / 1024, 1)
    except Exception:
        pass

    # Disk
    try:
        stat = shutil.disk_usage("/data")
        hw["disk_free_gb"] = round(stat.free / (1024 ** 3), 2)
    except Exception:
        try:
            stat = shutil.disk_usage("/")
            hw["disk_free_gb"] = round(stat.free / (1024 ** 3), 2)
        except Exception:
            pass

    return hw


class LogCapture:
    """Thread-safe line-oriented log capture that feeds into job_manager."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._buffer = io.StringIO()
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            self._buffer.write(text)
            # Flush complete lines
            value = self._buffer.getvalue()
            lines = value.split("\n")
            if len(lines) > 1:
                from web.job_manager import append_log
                for line in lines[:-1]:
                    stripped = line.rstrip()
                    if stripped:
                        append_log(self.job_id, stripped)
                self._buffer = io.StringIO()
                self._buffer.write(lines[-1])
        return len(text)

    def flush(self) -> None:
        pass

    def get_lines(self) -> list[str]:
        from web.job_manager import get_job
        job = get_job(self.job_id)
        return job.logs if job else []


class PipelineRunner:
    """Wraps PipelineOrchestrator execution in a background thread."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._log_capture = LogCapture(job_id)

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"pipeline-{self.job_id}",
            daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_event.set()
        logger.info("Cancellation requested for job %s", self.job_id)

    def _run(self) -> None:
        from web.job_manager import (
            get_job, update_job, append_log, set_stage_status,
            JobState, PIPELINE_STAGES,
        )

        job = get_job(self.job_id)
        if job is None:
            logger.error("Job %s not found, aborting runner", self.job_id)
            return

        update_job(self.job_id, state=JobState.RUNNING, started_at=time.time())
        append_log(self.job_id, f"Pipeline started for {job.filename}")

        total_stages = len(PIPELINE_STAGES)

        try:
            # Import orchestrator
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from pipeline.orchestrator import PipelineOrchestrator, PipelineState
            from pipeline.config import PipelineConfig

            config = PipelineConfig()
            config.output_dir = job.output_dir

            orchestrator = PipelineOrchestrator(
                video_path=job.input_video_path,
                output_dir=job.output_dir,
                config=config,
            )

            # Hook into the orchestrator's logging
            pipeline_logger = logging.getLogger("pipeline")
            handler = logging.StreamHandler(self._log_capture)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            pipeline_logger.addHandler(handler)
            pipeline_logger.setLevel(logging.INFO)

            # Run stages manually to inject VRAM cleanup and cancellation checks
            orchestrator._status.started_at = time.time()
            Path(job.output_dir).mkdir(parents=True, exist_ok=True)
            orchestrator._advance(PipelineState.INGEST)

            stage_index = 0

            while orchestrator._state not in (PipelineState.DONE, PipelineState.FAILED):
                if self._cancel_event.is_set():
                    append_log(self.job_id, "Pipeline cancelled by user")
                    update_job(
                        self.job_id,
                        state=JobState.CANCELLED,
                        finished_at=time.time(),
                        error="Cancelled by user",
                    )
                    return

                state = orchestrator._state
                stage_name = state.value

                # Hardware snapshot before stage
                hw_before = capture_hardware()
                set_stage_status(self.job_id, stage_name, "running", hardware=hw_before)
                append_log(self.job_id, f"Starting stage: {stage_name}")

                handler_fn = orchestrator._get_handler(state)
                result = orchestrator._execute_with_retries(handler_fn, stage_name)
                orchestrator._record_result(stage_name, result)

                # Hardware snapshot after stage
                hw_after = capture_hardware()

                # VRAM cleanup between stages
                free_vram()
                append_log(self.job_id, f"VRAM freed after {stage_name}")

                if result.success:
                    stage_index += 1
                    progress = round(stage_index / total_stages, 3)

                    # Check for preview artifacts
                    preview_path = None
                    if result.artifacts:
                        for key, val in result.artifacts.items():
                            if any(val.endswith(ext) for ext in (".png", ".jpg", ".exr")):
                                preview_path = val
                                break

                    set_stage_status(
                        self.job_id, stage_name, "completed",
                        hardware=hw_after, preview_path=preview_path,
                    )
                    update_job(self.job_id, progress=progress)
                    append_log(self.job_id, f"Completed stage: {stage_name} ({progress*100:.0f}%)")

                    from pipeline.orchestrator import _TRANSITIONS
                    next_state = _TRANSITIONS.get(state)
                    if next_state:
                        orchestrator._advance(next_state)
                    else:
                        orchestrator._advance(PipelineState.DONE)
                else:
                    error_msg = result.error or "Unknown error"
                    set_stage_status(
                        self.job_id, stage_name, "failed",
                        hardware=hw_after, error=error_msg,
                    )
                    append_log(self.job_id, f"Stage {stage_name} FAILED: {error_msg}")
                    orchestrator._advance(PipelineState.FAILED)

                orchestrator._write_status()

            # Final state
            if orchestrator._state == PipelineState.DONE:
                archive_path = self._create_result_archive(job)
                update_job(
                    self.job_id,
                    state=JobState.COMPLETED,
                    progress=1.0,
                    finished_at=time.time(),
                    result_archive=archive_path,
                )
                append_log(self.job_id, "Pipeline completed successfully")
            else:
                error_msg = orchestrator._status.error or "Pipeline failed"
                update_job(
                    self.job_id,
                    state=JobState.FAILED,
                    finished_at=time.time(),
                    error=error_msg,
                )
                append_log(self.job_id, f"Pipeline failed: {error_msg}")

            pipeline_logger.removeHandler(handler)

        except Exception as exc:
            error_msg = f"Pipeline exception: {exc}"
            logger.exception("Pipeline runner error for job %s", self.job_id)
            append_log(self.job_id, error_msg)
            update_job(
                self.job_id,
                state=JobState.FAILED,
                finished_at=time.time(),
                error=error_msg,
            )

    def _create_result_archive(self, job: Any) -> str | None:
        """Zip USD + meshes + textures into a downloadable archive."""
        output_dir = Path(job.output_dir)
        if not output_dir.exists():
            return None

        archive_path = output_dir / f"{self.job_id}_result.zip"
        try:
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _dirs, files in os.walk(output_dir):
                    for fname in files:
                        fpath = Path(root) / fname
                        if fpath == archive_path:
                            continue
                        arcname = fpath.relative_to(output_dir)
                        zf.write(fpath, arcname)
            return str(archive_path)
        except Exception as exc:
            logger.error("Failed to create result archive: %s", exc)
            return None


def start_pipeline(job_id: str) -> bool:
    """Launch a pipeline runner for the given job. Returns False if already running."""
    with _runners_lock:
        if job_id in _runners and not _runners[job_id].cancelled:
            return False
        runner = PipelineRunner(job_id)
        _runners[job_id] = runner
        runner.start()
        return True


def cancel_pipeline(job_id: str) -> bool:
    """Request cancellation of a running pipeline."""
    with _runners_lock:
        runner = _runners.get(job_id)
        if runner is None:
            return False
        runner.cancel()
        return True


def is_running(job_id: str) -> bool:
    """Check if a pipeline runner is active for the given job."""
    with _runners_lock:
        runner = _runners.get(job_id)
        if runner is None:
            return False
        return runner._thread is not None and runner._thread.is_alive()
