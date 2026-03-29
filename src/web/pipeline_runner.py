# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Thin job-queue layer for the web pipeline.

Claude Code is the orchestrator. This module only:
1. Saves uploaded videos to the job directory
2. Sets job state to "queued"
3. Provides an API for Claude Code to update job progress

The old PipelineRunner that drove the state machine is removed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Persistent API key location (same as app.py)
_API_KEY_PATH = Path(os.environ.get("LFS_API_KEY_PATH", "/data/.anthropic_key"))


def queue_job(job_id: str) -> bool:
    """Mark a job as queued for Claude Code to pick up.

    Copies the input video into the job output directory so
    Claude Code can find it at a known location.

    Returns False if the job doesn't exist.
    """
    from web.job_manager import get_job, update_job, append_log, JobState

    job = get_job(job_id)
    if job is None:
        logger.error("Job %s not found", job_id)
        return False

    # Ensure output directory exists
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy or symlink the input video into the job directory
    input_path = Path(job.input_video_path)
    if input_path.exists():
        target = output_dir / "input" / input_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(str(input_path), str(target))

        # Also create a convenience symlink as input.mp4
        link = output_dir / "input.mp4"
        if not link.exists():
            try:
                link.symlink_to(target)
            except OSError:
                shutil.copy2(str(target), str(link))

    update_job(job_id, state=JobState.QUEUED)
    append_log(job_id, f"Job queued for Claude Code: {job.filename}")
    append_log(job_id, f"Job directory: {job.output_dir}")

    # Attempt to auto-launch Claude Code if an API key is available
    launched = _launch_claude_code(job_id, job.output_dir)
    if launched:
        append_log(job_id, "Claude Code launched automatically with stored API key")
    else:
        append_log(job_id, "Waiting for Claude Code to pick up this job in the terminal...")

    return True


def _launch_claude_code(job_id: str, output_dir: str) -> bool:
    """Launch Claude Code as a background subprocess to process the job.

    Reads the API key from the persistent volume. If no key is stored,
    returns False (the user must run Claude Code manually via the terminal).
    """
    # Check for API key or OAuth session
    api_key = None
    if _API_KEY_PATH.exists():
        api_key = _API_KEY_PATH.read_text().strip() or None

    prompt = (
        f"Process the video pipeline job at {output_dir}. "
        f"Job ID is {job_id}. "
        f"Follow the instructions in CLAUDE.md step by step. "
        f"You MUST complete ALL stages: ingest → select_frames → reconstruct → "
        f"train → segment → extract_objects → mesh_objects → assemble_usd → validate. "
        f"Do NOT stop after training. Continue through segmentation, mesh extraction, "
        f"and USD assembly. "
        f"Report progress to the web API at http://localhost:7860/api/job/{job_id}/stage "
        f"and mark completion at http://localhost:7860/api/job/{job_id}/complete"
    )

    # Build environment — web interface runs as ubuntu user so
    # Claude Code has direct access to OAuth session in ~/.claude/
    env = {**os.environ, "TERM": "xterm-256color"}
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    # If no API key, Claude Code uses the OAuth session from ~/.claude/

    claude_bin = "/usr/local/bin/claude"
    cmd = [
        claude_bin,
        "--dangerously-skip-permissions",
        "-p", prompt,
        "--allowedTools", "Bash,Read,Write,Edit",
    ]

    logger.info("Launching Claude Code: %s", " ".join(cmd[:4]) + " ...")

    try:
        log_path = Path(output_dir) / "claude_launch.log"
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            cwd="/opt/gaussian-toolkit",
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        logger.info("Launched Claude Code (pid=%d) for job %s, log: %s", proc.pid, job_id, log_path)
        return True
    except FileNotFoundError:
        logger.warning("Claude Code binary not found at %s — cannot auto-launch", claude_bin)
        return False
    except Exception as exc:
        logger.error("Failed to launch Claude Code for job %s: %s", job_id, exc)
        return False


def update_stage(job_id: str, stage: str, progress: float = 0.0, message: str = "") -> bool:
    """Called by Claude Code (via REST API) to report stage progress.

    Args:
        job_id: The job identifier.
        stage: Current stage name (e.g. "train", "segment").
        progress: Overall progress 0.0-1.0.
        message: Optional status message (e.g. "30k iter, loss 0.02").

    Returns False if the job doesn't exist.
    """
    from web.job_manager import get_job, update_job, append_log, set_stage_status, JobState

    job = get_job(job_id)
    if job is None:
        return False

    update_job(job_id, state=JobState.RUNNING, progress=progress, current_stage=stage)
    set_stage_status(job_id, stage, "running")

    if message:
        append_log(job_id, f"[{stage}] {message}")

    return True


def complete_stage(job_id: str, stage: str, success: bool = True, error: str = "") -> bool:
    """Mark a stage as completed or failed.

    Args:
        job_id: The job identifier.
        stage: Stage name that just finished.
        success: Whether the stage succeeded.
        error: Error message if failed.
    """
    from web.job_manager import get_job, append_log, set_stage_status

    job = get_job(job_id)
    if job is None:
        return False

    status = "completed" if success else "failed"
    set_stage_status(job_id, stage, status, error=error if not success else None)

    if success:
        append_log(job_id, f"Stage {stage} completed")
    else:
        append_log(job_id, f"Stage {stage} FAILED: {error}")

    return True


def complete_job(job_id: str, success: bool = True, error: str = "") -> bool:
    """Mark the entire job as completed or failed.

    If successful, creates a downloadable ZIP archive.
    """
    from web.job_manager import get_job, update_job, append_log, JobState

    job = get_job(job_id)
    if job is None:
        return False

    if success:
        archive_path = _create_result_archive(job_id, job.output_dir)
        update_job(
            job_id,
            state=JobState.COMPLETED,
            progress=1.0,
            finished_at=time.time(),
            result_archive=archive_path,
        )
        append_log(job_id, "Pipeline completed successfully")
    else:
        update_job(
            job_id,
            state=JobState.FAILED,
            finished_at=time.time(),
            error=error,
        )
        append_log(job_id, f"Pipeline failed: {error}")

    return True


def cancel_pipeline(job_id: str) -> bool:
    """Cancel a job. Since Claude Code is the orchestrator, this just
    updates the state. Claude Code checks the state and stops."""
    from web.job_manager import get_job, update_job, append_log, JobState

    job = get_job(job_id)
    if job is None:
        return False

    update_job(
        job_id,
        state=JobState.CANCELLED,
        finished_at=time.time(),
        error="Cancelled by user",
    )
    append_log(job_id, "Job cancelled by user")
    return True


def is_running(job_id: str) -> bool:
    """Check if a job is in an active state."""
    from web.job_manager import get_job, JobState

    job = get_job(job_id)
    if job is None:
        return False
    return job.state in (JobState.RUNNING, JobState.QUEUED) or job.state.startswith("stage_")


# ---- Backward compat alias ----
# The old upload handler calls start_pipeline(). Now it just queues.
start_pipeline = queue_job


def _create_result_archive(job_id: str, output_dir: str) -> str | None:
    """Zip USD + meshes + textures into a downloadable archive."""
    output = Path(output_dir)
    if not output.exists():
        return None

    archive_path = output / f"{job_id}_result.zip"
    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(output):
                for fname in files:
                    fpath = Path(root) / fname
                    if fpath == archive_path:
                        continue
                    arcname = fpath.relative_to(output)
                    zf.write(fpath, arcname)
        return str(archive_path)
    except Exception as exc:
        logger.error("Failed to create result archive: %s", exc)
        return None
