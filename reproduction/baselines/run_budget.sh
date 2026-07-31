#!/usr/bin/env bash
set -euo pipefail

method="${1:?usage: run_budget.sh flap|tyr|streamline 15|20|25|30}"
budget="${2:?}"
case "${budget}" in 15|20|25|30) ;; *) echo "invalid budget" >&2; exit 2;; esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ratio="0.${budget}"

case "${method}" in
  flap)
    TARGET_COMPRESSION="${ratio}" RUN_TAG="llama32_3b_${budget}pct_s44" SEED=44 \
      bash "${root}/flap/run.sh"
    ;;
  tyr)
    MODEL_NAME_OR_PATH="${TEACHER_CKPT:?set TEACHER_CKPT}" \
      SPARSITY="${ratio}" SPARSITY_TAG="${budget}pct_s44" MODEL_LABEL=llama32_3b \
      RUN_TAG="tyr_llama32_3b_${budget}pct_s44" SEED=44 \
      bash "${root}/tyr/run.sh"
    ;;
  streamline)
    case "${budget}" in 15) remove=5;; 20) remove=6;; 25) remove=8;; 30) remove=10;; esac
    REMOVE_LAYERS="${remove}" RUN_TAG="streamline_llama32_3b_${budget}pct_s44" SEED=44 \
      bash "${root}/llm-streamline/run.sh"
    ;;
  *) echo "method must be flap, tyr or streamline" >&2; exit 2;;
esac
