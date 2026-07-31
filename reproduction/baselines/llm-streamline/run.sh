#!/usr/bin/env bash
set -euo pipefail

# A. Task-fair comparison:
# Compress the same task-tuned teacher with LLM-Streamline, train on the same
# commonsense instruction data, then evaluate teacher and compressed model with
# the local new_idea evaluator.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMLINE_ROOT="${STREAMLINE_ROOT:?set STREAMLINE_ROOT to the official LLM-Streamline checkout}"
WORKSPACE_ROOT="${BASELINE_WORKSPACE_ROOT:?set BASELINE_WORKSPACE_ROOT to a writable workspace}"
CUSTOM_RUNNER="${SCRIPT_DIR}/streamline_fair_runner.py"
EVAL_PY="${SCRIPT_DIR}/../common/eval.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

TEACHER_CKPT="${TEACHER_CKPT:?set TEACHER_CKPT to the merged teacher}"
TRAIN_DATA="${TRAIN_DATA:?set TRAIN_DATA to commonsense_170k.json}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:?set TEST_DATA_ROOT to the seven-task dataset directory}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${WORKSPACE_ROOT}/out/streamline_compare_A_${RUN_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/streamline_compare_A_${RUN_TAG}}"
STREAMLINE_MODEL_DIR="${STREAMLINE_MODEL_DIR:-${OUT_ROOT}/streamline_llmloss}"

DATASETS="${DATASETS:-ARC-Challenge ARC-Easy hellaswag openbookqa piqa social_i_qa winogrande}"
REMOVE_LAYERS="${REMOVE_LAYERS:-8}"
MAX_RECORDS="${MAX_RECORDS:-100000}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
COSINE_NUM_DATA="${COSINE_NUM_DATA:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-5e-5}"
WD="${WD:-1e-3}"
VAL_RATIO="${VAL_RATIO:-0.02}"
NUM_PROC="${NUM_PROC:-1}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_MODE="${EVAL_MODE:-logprob}"
LENGTH_NORM="${LENGTH_NORM:-none}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

TRAIN_STREAMLINE="${TRAIN_STREAMLINE:-true}"
EVAL_TEACHER="${EVAL_TEACHER:-true}"
EVAL_STREAMLINE="${EVAL_STREAMLINE:-true}"
WRITE_STATS="${WRITE_STATS:-true}"

# Optional comma-separated extra CSVs for the final summary, for example:
# EXTRA_CSVS="our_ce=/path/ce.csv,our_ce_fad=/path/fad.csv"
EXTRA_CSVS="${EXTRA_CSVS:-}"

mkdir -p "${OUT_ROOT}" "${RESULTS_DIR}"

echo "[A] teacher=${TEACHER_CKPT}"
echo "[A] train_data=${TRAIN_DATA}"
echo "[A] streamline_model=${STREAMLINE_MODEL_DIR}"
echo "[A] results=${RESULTS_DIR}"

if [ "${TRAIN_STREAMLINE}" = "true" ]; then
  cd "${STREAMLINE_ROOT}"
  "${PYTHON_BIN}" -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${CUSTOM_RUNNER}" train-llmloss \
    --model_name "${TEACHER_CKPT}" \
    --output_dir "${STREAMLINE_MODEL_DIR}" \
    --data_source local_json \
    --data_path "${TRAIN_DATA}" \
    --max_records "${MAX_RECORDS}" \
    --val_ratio "${VAL_RATIO}" \
    --block_size "${BLOCK_SIZE}" \
    --num_proc "${NUM_PROC}" \
    --layer_intervals "${REMOVE_LAYERS}" \
    --cosine_num_data "${COSINE_NUM_DATA}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_step "${GRAD_ACCUM}" \
    --epoches "${EPOCHS}" \
    --lr "${LR}" \
    --min_lr "${MIN_LR}" \
    --wd "${WD}" \
    --seed "${SEED}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --mixed_precision "${MIXED_PRECISION}" \
    2>&1 | tee "${OUT_ROOT}/train_streamline_llmloss.log"
fi

read -r -a DATASET_ARR <<< "${DATASETS}"

run_eval_parallel() {
  local results_csv="$1"
  local log_prefix="$2"
  shift 2

  local eval_gpus="${NUM_GPUS:-${NUM_PROCESSES:-1}}"
  if ! [[ "${eval_gpus}" =~ ^[0-9]+$ ]] || [ "${eval_gpus}" -lt 1 ]; then
    eval_gpus=1
  fi

  local tmp_dir="${RESULTS_DIR}/.${log_prefix}_parts"
  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"

  local -a pids=()
  local idx=0
  local ds safe_ds gpu pid
  for ds in "${DATASET_ARR[@]}"; do
    safe_ds="${ds//[^A-Za-z0-9_.-]/_}"
    gpu=$(( idx % eval_gpus ))
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      "${PYTHON_BIN}" eval.py \
        "$@" \
        --datasets "${ds}" \
        --results_csv "${tmp_dir}/${safe_ds}.csv" \
        2>&1 | tee "${RESULTS_DIR}/${log_prefix}_${safe_ds}.log"
    ) &
    pids+=("$!")
    idx=$((idx + 1))
    if [ "${#pids[@]}" -ge "${eval_gpus}" ]; then
      for pid in "${pids[@]}"; do
        wait "${pid}"
      done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  rm -f "${results_csv}"
  local first=1
  for ds in "${DATASET_ARR[@]}"; do
    safe_ds="${ds//[^A-Za-z0-9_.-]/_}"
    if [ ! -f "${tmp_dir}/${safe_ds}.csv" ]; then
      echo "[A][Error] missing eval shard csv: ${tmp_dir}/${safe_ds}.csv" >&2
      exit 1
    fi
    if [ "${first}" -eq 1 ]; then
      cat "${tmp_dir}/${safe_ds}.csv" > "${results_csv}"
      first=0
    else
      tail -n +2 "${tmp_dir}/${safe_ds}.csv" >> "${results_csv}"
    fi
  done
}

if [ "${EVAL_TEACHER}" = "true" ]; then
  cd "$(dirname "${EVAL_PY}")"
  run_eval_parallel "${RESULTS_DIR}/teacher_native.csv" "eval_teacher_native" \
    --eval_profile pipeline \
    --mode pipeline \
    --base_model "${TEACHER_CKPT}" \
    --teacher_ckpt "${TEACHER_CKPT}" \
    --teacher_loader native \
    --test_data_path "${TEST_DATA_ROOT}" \
    --run_stateft True \
    --run_student_base False \
    --run_student_predictor False \
    --run_student_guidance False \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --eval_mode "${EVAL_MODE}" \
    --length_norm "${LENGTH_NORM}" \
    --device "${DEVICE}" \
    --seed "${SEED}"
fi

if [ "${EVAL_STREAMLINE}" = "true" ]; then
  cd "$(dirname "${EVAL_PY}")"
  run_eval_parallel "${RESULTS_DIR}/streamline_llmloss.csv" "eval_streamline_llmloss" \
    --eval_profile pipeline \
    --mode pipeline \
    --base_model "${TEACHER_CKPT}" \
    --teacher_ckpt "${TEACHER_CKPT}" \
    --teacher_loader native \
    --test_data_path "${TEST_DATA_ROOT}" \
    --run_stateft False \
    --run_student_base True \
    --run_student_predictor False \
    --run_student_guidance False \
    --student_ckpt_dir "${STREAMLINE_MODEL_DIR}" \
    --student_use_control False \
    --student_control_missing disable \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --eval_mode "${EVAL_MODE}" \
    --length_norm "${LENGTH_NORM}" \
    --device "${DEVICE}" \
    --seed "${SEED}"
fi

SUMMARY_ARGS=()
if [ -f "${RESULTS_DIR}/teacher_native.csv" ]; then
  SUMMARY_ARGS+=(--csv "teacher_native=${RESULTS_DIR}/teacher_native.csv")
fi
if [ -f "${RESULTS_DIR}/streamline_llmloss.csv" ]; then
  SUMMARY_ARGS+=(--csv "streamline_llmloss=${RESULTS_DIR}/streamline_llmloss.csv")
fi
if [ -n "${EXTRA_CSVS}" ]; then
  IFS=',' read -r -a EXTRA_ARR <<< "${EXTRA_CSVS}"
  for item in "${EXTRA_ARR[@]}"; do
    SUMMARY_ARGS+=(--csv "${item}")
  done
fi

if [ "${#SUMMARY_ARGS[@]}" -gt 0 ]; then
  cd "${STREAMLINE_ROOT}"
  "${PYTHON_BIN}" "${CUSTOM_RUNNER}" summarize \
    "${SUMMARY_ARGS[@]}" \
    --output_csv "${RESULTS_DIR}/summary_task_fair.csv" \
    --output_json "${RESULTS_DIR}/summary_task_fair.json"
fi

if [ "${WRITE_STATS}" = "true" ]; then
  cd "${STREAMLINE_ROOT}"
  "${PYTHON_BIN}" "${CUSTOM_RUNNER}" model-stats \
    --model_name "${TEACHER_CKPT}" \
    --output_json "${RESULTS_DIR}/teacher_model_stats.json" \
    --torch_dtype "${TORCH_DTYPE}" || true
  "${PYTHON_BIN}" "${CUSTOM_RUNNER}" model-stats \
    --model_name "${STREAMLINE_MODEL_DIR}" \
    --output_json "${RESULTS_DIR}/streamline_model_stats.json" \
    --torch_dtype "${TORCH_DTYPE}" || true
fi

echo "[A] done"
echo "[A] model: ${STREAMLINE_MODEL_DIR}"
echo "[A] summary: ${RESULTS_DIR}/summary_task_fair.csv"
