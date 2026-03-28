#!/bin/bash
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Runs USD assembly using the Python 3.11 venv (usd-core needs Python <=3.12).
# Usage:
#   scripts/tools/usd-assemble.sh [args...]
#
# Examples:
#   scripts/tools/usd-assemble.sh --colmap-dir colmap/exported --output scene.usda
#   scripts/tools/usd-assemble.sh --synthetic --output /tmp/test.usda

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_PYTHON="${PROJECT_ROOT}/.venv-usd/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "ERROR: USD venv not found at ${PROJECT_ROOT}/.venv-usd/" >&2
    echo "Create it with:" >&2
    echo "  python3.11 -m venv ${PROJECT_ROOT}/.venv-usd" >&2
    echo "  ${PROJECT_ROOT}/.venv-usd/bin/pip install usd-core trimesh numpy" >&2
    exit 1
fi

# If called with no arguments, run the gallery assembler
if [[ $# -eq 0 ]]; then
    exec "${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/assemble_gallery_usd.py"
else
    # Check if first arg is a known script name
    case "$1" in
        assemble|gallery)
            shift
            exec "${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/assemble_gallery_usd.py" "$@"
            ;;
        test)
            shift
            exec "${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/test_usd_pipeline.py" "$@"
            ;;
        pipeline)
            shift
            exec "${VENV_PYTHON}" -m pipeline.usd_assembler "$@"
            ;;
        *)
            exec "${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/assemble_gallery_usd.py" "$@"
            ;;
    esac
fi
