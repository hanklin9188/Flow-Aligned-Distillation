#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python}" "${root}/server.py" \
  --host "${FAD_HOST:-127.0.0.1}" --port "${1:-8765}" \
  --questions "${root}/questions.json" --static_dir "${root}/static" --mock
