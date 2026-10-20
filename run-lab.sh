#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv sync --frozen --python 3.12
if [ ! -f models/face_landmarker.task ]; then
  uv run eye-mouse download-model
fi
exec uv run eye-mouse "$@"
