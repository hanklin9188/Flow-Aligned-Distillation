#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
port="${1:-7860}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

exec "${PYTHON_BIN:-python}" "${root}/server.py" \
  --host "${FAD_HOST:-127.0.0.1}" \
  --port "${port}" \
  --questions "${root}/questions.json" \
  --static_dir "${root}/static" \
  --runtime_code "${root}/runtime_code" \
  --deploy_bundle "${FAD_STUDENT_BUNDLE:-${root}/models/student/deploy_bundle.pt}" \
  --base_model "${FAD_BASE_MODEL:-${root}/models/base_llama32_3b}" \
  --teacher "${FAD_TEACHER_MODEL:-${root}/models/merged_teacher}" \
  --policy "${FAD_EXIT_POLICY:-${root}/models/student/oracle_distilled_exit_policy.json}" \
  --warmup_rounds "${FAD_WARMUP_ROUNDS:-1}"
