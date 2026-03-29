# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Flask web application for the LichtFeld video-to-scene pipeline.

Provides upload, monitoring, log streaming (SSE), preview, and download
endpoints. Runs on port 7860 by default.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
)
from werkzeug.utils import secure_filename

# Ensure src/ is on the path so pipeline imports resolve
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from web.job_manager import (
    INPUT_DIR,
    JobState,
    cleanup_old_jobs,
    create_job,
    delete_job,
    get_job,
    list_jobs,
    update_job,
)
from web.pipeline_runner import cancel_pipeline, is_running, start_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

# 2 GB upload limit
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> str:
    """Serve the single-page upload/monitoring UI."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload() -> tuple[Response, int]:
    """Accept a video file upload, create a job, and start the pipeline."""
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    file = request.files["video"]
    if file.filename is None or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    filename = secure_filename(file.filename)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use timestamp prefix to avoid collisions
    ts = int(time.time() * 1000)
    save_name = f"{ts}_{filename}"
    save_path = INPUT_DIR / save_name

    try:
        file.save(str(save_path))
    except Exception as exc:
        logger.error("File save failed: %s", exc)
        return jsonify({"error": "Failed to save file"}), 500

    file_size = save_path.stat().st_size
    job = create_job(
        filename=filename,
        input_video_path=str(save_path),
        file_size_bytes=file_size,
    )

    started = start_pipeline(job.job_id)
    if not started:
        return jsonify({"error": "Failed to start pipeline"}), 500

    return jsonify({
        "job_id": job.job_id,
        "filename": filename,
        "file_size_bytes": file_size,
        "state": job.state,
    }), 201


@app.route("/status/<job_id>")
def status(job_id: str) -> tuple[Response, int]:
    """Return full job status as JSON."""
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job.to_dict()), 200


@app.route("/stream/<job_id>")
def stream(job_id: str) -> Response:
    """Server-Sent Events endpoint for real-time log and status updates."""
    job = get_job(job_id)
    if job is None:
        abort(404)

    def generate():
        last_log_index = 0
        last_state = ""
        last_progress = -1.0

        while True:
            job = get_job(job_id)
            if job is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                break

            # Send state changes
            if job.state != last_state:
                last_state = job.state
                yield f"data: {json.dumps({'type': 'state', 'state': job.state, 'current_stage': job.current_stage})}\n\n"

            # Send progress changes
            if job.progress != last_progress:
                last_progress = job.progress
                yield f"data: {json.dumps({'type': 'progress', 'progress': job.progress})}\n\n"

            # Send new log lines
            if len(job.logs) > last_log_index:
                new_lines = job.logs[last_log_index:]
                last_log_index = len(job.logs)
                for line in new_lines:
                    yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"

            # Send preview updates
            if job.previews:
                yield f"data: {json.dumps({'type': 'previews', 'previews': job.previews})}\n\n"

            # Terminal states
            if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                yield f"data: {json.dumps({'type': 'done', 'state': job.state, 'error': job.error, 'result_archive': job.result_archive})}\n\n"
                break

            time.sleep(1)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/download/<job_id>")
def download(job_id: str) -> Response:
    """Download the result archive for a completed job."""
    job = get_job(job_id)
    if job is None:
        abort(404)

    if job.state != JobState.COMPLETED:
        abort(400, description="Job not completed")

    if not job.result_archive or not Path(job.result_archive).exists():
        abort(404, description="Result archive not found")

    return send_file(
        job.result_archive,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{job.filename.rsplit('.', 1)[0]}_result.zip",
    )


@app.route("/preview/<job_id>/<stage>")
def preview(job_id: str, stage: str) -> Response:
    """Serve a Blender render / preview image for a specific pipeline stage."""
    job = get_job(job_id)
    if job is None:
        abort(404)

    preview_path = job.previews.get(stage)
    if not preview_path or not Path(preview_path).exists():
        abort(404, description=f"No preview for stage {stage}")

    # Determine mimetype from extension
    ext = Path(preview_path).suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".exr": "image/x-exr"}
    mime = mime_map.get(ext, "application/octet-stream")

    return send_file(preview_path, mimetype=mime)


@app.route("/jobs")
def jobs() -> tuple[Response, int]:
    """List all jobs with summary info."""
    return jsonify(list_jobs()), 200


@app.route("/job/<job_id>", methods=["DELETE"])
def remove_job(job_id: str) -> tuple[Response, int]:
    """Cancel a running job or delete a finished one."""
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    # Cancel if running
    if job.state in (JobState.RUNNING, JobState.QUEUED) or job.state.startswith("stage_"):
        cancel_pipeline(job_id)
        update_job(
            job_id,
            state=JobState.CANCELLED,
            finished_at=time.time(),
            error="Cancelled by user",
        )

    deleted = delete_job(job_id)
    if deleted:
        return jsonify({"status": "deleted", "job_id": job_id}), 200
    return jsonify({"error": "Delete failed"}), 500


@app.route("/health")
def health() -> tuple[Response, int]:
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": time.time()}), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 2 GB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e)}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    """Application factory for WSGI servers."""
    # Run cleanup of old jobs on startup
    try:
        removed = cleanup_old_jobs()
        if removed:
            logger.info("Cleaned up %d old jobs", removed)
    except Exception as exc:
        logger.warning("Job cleanup failed: %s", exc)

    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.environ.get("LFS_WEB_PORT", "7860"))
    application.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("LFS_DEBUG", "").lower() in ("1", "true"),
        threaded=True,
    )
