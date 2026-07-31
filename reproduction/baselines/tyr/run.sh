#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TYR_ROOT="${TYR_ROOT:?set TYR_ROOT to the official Tyr-the-Pruner checkout}"
WORKSPACE_ROOT="${BASELINE_WORKSPACE_ROOT:?set BASELINE_WORKSPACE_ROOT to a writable workspace}"
EVAL_PY="${SCRIPT_DIR}/../common/eval.py"
SUMMARY_RUNNER="${SCRIPT_DIR}/../llm-streamline/streamline_fair_runner.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH is required}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-${MODEL_NAME_OR_PATH}}"
SPARSITY="${SPARSITY:?SPARSITY is required, for example 0.15}"
SPARSITY_TAG="${SPARSITY_TAG:-$(printf '%s' "${SPARSITY}" | tr -d '.' )}"

TRAIN_DATA="${TRAIN_DATA:?set TRAIN_DATA to commonsense_170k.json}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:?set TEST_DATA_ROOT to the seven-task dataset directory}"
DATASETS="${DATASETS:-ARC-Challenge ARC-Easy hellaswag openbookqa piqa social_i_qa winogrande}"

RUN_TAG="${RUN_TAG:-tyr_${MODEL_LABEL}_${SPARSITY_TAG}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${WORKSPACE_ROOT}/out/${RUN_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/${RUN_TAG}}"
SUPERNET_ROOT="${SUPERNET_ROOT:-${OUT_ROOT}/Cached-Supernets}"
MATERIALIZED_MODEL_DIR="${MATERIALIZED_MODEL_DIR:-${OUT_ROOT}/tyr_sparse_model}"
CALIBRATION_CACHE="${CALIBRATION_CACHE:-${OUT_ROOT}/calibration/commonsense_${MODEL_LABEL}_${CALIBRATION_SEQUENCE_LENGTH:-4096}_${CALIBRATION_TOKENS:-4194304}.pt}"

CALIBRATION_SEQUENCE_LENGTH="${CALIBRATION_SEQUENCE_LENGTH:-4096}"
CALIBRATION_TOKENS="${CALIBRATION_TOKENS:-4194304}"
SEARCH_SEQUENCE_LENGTH="${SEARCH_SEQUENCE_LENGTH:-${CALIBRATION_SEQUENCE_LENGTH}}"
SEARCH_TOKENS="${SEARCH_TOKENS:-${CALIBRATION_TOKENS}}"
SEARCH_EVAL_TOKENS="${SEARCH_EVAL_TOKENS:-524288}"
SEARCH_EVAL_SEQUENCE_LENGTH="${SEARCH_EVAL_SEQUENCE_LENGTH:-4096}"
SEARCH_EVAL_DATASETS="${SEARCH_EVAL_DATASETS:-wikitext2}"

ITERATIONS="${ITERATIONS:-4}"
NUM_SPARSITY_LEVELS="${NUM_SPARSITY_LEVELS:-9}"
GENERATIONS="${GENERATIONS:-50 50 50 50}"
OFFSPRING="${OFFSPRING:-128}"
TOKENS_PER_SELECTION="${TOKENS_PER_SELECTION:-2048 16384 131072}"
SURVIVORS_PER_SELECTION="${SURVIVORS_PER_SELECTION:-16 4 2}"
KL_TOPK="${KL_TOPK:-8192}"
FITNESS_FN="${FITNESS_FN:-sparse_kl}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MODE="${EVAL_MODE:-logprob}"
LENGTH_NORM="${LENGTH_NORM:-none}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
SEED="${SEED:-42}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
REBUILD_CALIBRATION_CACHE="${REBUILD_CALIBRATION_CACHE:-false}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]]; then
  NUM_GPUS=1
fi
if [ "${NUM_GPUS}" -lt 1 ]; then
  NUM_GPUS=1
fi

mkdir -p "${OUT_ROOT}" "${RESULTS_DIR}" "${SUPERNET_ROOT}" "$(dirname "${CALIBRATION_CACHE}")"

if [ ! -d "${MODEL_NAME_OR_PATH}" ] && [ ! -f "${MODEL_NAME_OR_PATH}" ]; then
  echo "[TyrFair][Error] model not found: ${MODEL_NAME_OR_PATH}" >&2
  exit 1
fi
if [ ! -f "${TRAIN_DATA}" ]; then
  echo "[TyrFair][Error] train/calibration JSON not found: ${TRAIN_DATA}" >&2
  exit 1
fi
if [ ! -d "${TEST_DATA_ROOT}" ]; then
  echo "[TyrFair][Error] test data root not found: ${TEST_DATA_ROOT}" >&2
  exit 1
fi

read -r -a GENERATION_ARR <<< "${GENERATIONS}"
if [ "${#GENERATION_ARR[@]}" -lt "${ITERATIONS}" ]; then
  echo "[TyrFair][Error] GENERATIONS must provide at least ${ITERATIONS} values." >&2
  exit 1
fi

echo "[TyrFair] model_label=${MODEL_LABEL}"
echo "[TyrFair] model=${MODEL_NAME_OR_PATH}"
echo "[TyrFair] sparsity=${SPARSITY}"
echo "[TyrFair] run_tag=${RUN_TAG}"
echo "[TyrFair] out_root=${OUT_ROOT}"
echo "[TyrFair] results_dir=${RESULTS_DIR}"
echo "[TyrFair] datasets=${DATASETS}"
echo "[TyrFair] num_gpus=${NUM_GPUS}"
echo "[TyrFair] seed=${SEED}"

if [ "${REBUILD_CALIBRATION_CACHE}" = "true" ] || [ ! -f "${CALIBRATION_CACHE}" ]; then
  echo "[TyrFair] building calibration cache=${CALIBRATION_CACHE}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_calibration_cache.py" \
    --data_path "${TRAIN_DATA}" \
    --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}" \
    --output_path "${CALIBRATION_CACHE}" \
    --max_tokens "${CALIBRATION_TOKENS}" \
    --sequence_length "${CALIBRATION_SEQUENCE_LENGTH}" \
    --seed "${SEED}"
else
  echo "[TyrFair] reusing calibration cache=${CALIBRATION_CACHE}"
fi

weights_diff_json="$("${PYTHON_BIN}" - "${MODEL_NAME_OR_PATH}" <<'PY'
import json
import sys
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
hidden = int(cfg.hidden_size)
intermediate = int(cfg.intermediate_size)
print(json.dumps({
    "mha": max((hidden * hidden) // 8, hidden * hidden // 1024),
    "mlp": max((hidden * intermediate) // 8, 32 * hidden),
}))
PY
)"
WEIGHTS_DIFF_MHA="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["mha"])' <<< "${weights_diff_json}")"
WEIGHTS_DIFF_MLP="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["mlp"])' <<< "${weights_diff_json}")"
echo "[TyrFair] initial_weights_diff_mha=${WEIGHTS_DIFF_MHA}"
echo "[TyrFair] initial_weights_diff_mlp=${WEIGHTS_DIFF_MLP}"

cd "${TYR_ROOT}"
prev_dir="None"
prev_config="None"
final_supernet_dir=""
final_config_name="final_configuration.txt"

for ((r = 1; r <= ITERATIONS; r++)); do
  iter_dir="${SUPERNET_ROOT}/${MODEL_LABEL}-s${SPARSITY}-iteration${r}"
  echo "[TyrFair] iteration=${r}/${ITERATIONS} supernet=${iter_dir}"

  torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" "${TYR_ROOT}/prune_to_supernet.py" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --tokenizer_name "${TOKENIZER_NAME_OR_PATH}" \
    --prunable_modules '^(?!.*(?:embedding|emb|head)).+$' \
    --pre_block_modules model.embed_tokens model.rotary_emb \
    --block_modules model.layers \
    --calibration_data "${CALIBRATION_CACHE}" \
    --calibration_sequence_length "${CALIBRATION_SEQUENCE_LENGTH}" \
    --calibration_tokens "${CALIBRATION_TOKENS}" \
    --dtype "${TORCH_DTYPE}" \
    --low_cpu_mem_usage \
    --attn_implementation "${ATTN_IMPL}" \
    --cpu_offload_modules \
    --cpu_offload_activations \
    --verbose \
    --sparsity "${SPARSITY}" \
    --error_accumulation \
    --supernet_dir "${prev_dir}" \
    --supernet_config "${prev_config}" \
    --weights_diff_mlp "${WEIGHTS_DIFF_MLP}" \
    --weights_diff_mha "${WEIGHTS_DIFF_MHA}" \
    --save_dir "${iter_dir}" \
    --num_sparsity_levels "${NUM_SPARSITY_LEVELS}"

  echo "[TyrFair] search iteration=${r}/${ITERATIONS}"
  "${PYTHON_BIN}" "${TYR_ROOT}/search_sparsity_dist.py" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --tokenizer_name "${TOKENIZER_NAME_OR_PATH}" \
    --calibration_data "${CALIBRATION_CACHE}" \
    --calibration_sequence_length "${SEARCH_SEQUENCE_LENGTH}" \
    --calibration_tokens "${SEARCH_TOKENS}" \
    --eval_datasets ${SEARCH_EVAL_DATASETS} \
    --eval_every 5 \
    --eval_tokens "${SEARCH_EVAL_TOKENS}" \
    --eval_sequence_length "${SEARCH_EVAL_SEQUENCE_LENGTH}" \
    --fitness_fn "${FITNESS_FN}" \
    --kl_topk "${KL_TOPK}" \
    --dtype "${TORCH_DTYPE}" \
    --attn_implementation "${ATTN_IMPL}" \
    --offspring "${OFFSPRING}" \
    --generations "${GENERATION_ARR[$((r - 1))]}" \
    --tokens_per_selection ${TOKENS_PER_SELECTION} \
    --survivors_per_selection ${SURVIVORS_PER_SELECTION} \
    --seed "${SEED}" \
    --sparse_weights_path "${iter_dir}" \
    --configuration_name "${final_config_name}" \
    --json_log_name "${iter_dir}/search_log.json"

  prev_dir="${iter_dir}"
  prev_config="${iter_dir}/${final_config_name}"
  final_supernet_dir="${iter_dir}"
  WEIGHTS_DIFF_MHA=$((WEIGHTS_DIFF_MHA / 2))
  WEIGHTS_DIFF_MLP=$((WEIGHTS_DIFF_MLP / 2))
done

echo "[TyrFair] materializing sparse model=${MATERIALIZED_MODEL_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/materialize_sparse_model.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}" \
  --sparse_weights_path "${final_supernet_dir}" \
  --sparse_config_name "${final_config_name}" \
  --output_dir "${MATERIALIZED_MODEL_DIR}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --attn_implementation "${ATTN_IMPL}" \
  --low_cpu_mem_usage \
  --safe_serialization

read -r -a DATASET_ARR <<< "${DATASETS}"
cd "$(dirname "${EVAL_PY}")"
"${PYTHON_BIN}" "${EVAL_PY}" \
  --eval_profile pipeline \
  --mode pipeline \
  --base_model "${MODEL_NAME_OR_PATH}" \
  --teacher_ckpt "${MODEL_NAME_OR_PATH}" \
  --teacher_loader native \
  --test_data_path "${TEST_DATA_ROOT}" \
  --datasets "${DATASET_ARR[@]}" \
  --results_csv "${RESULTS_DIR}/tyr_sparse.csv" \
  --run_stateft False \
  --run_student_base True \
  --run_student_predictor False \
  --run_student_guidance False \
  --student_ckpt_dir "${MATERIALIZED_MODEL_DIR}" \
  --student_use_control False \
  --student_control_missing disable \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --eval_mode "${EVAL_MODE}" \
  --length_norm "${LENGTH_NORM}" \
  --device "${EVAL_DEVICE}" \
  --seed "${SEED}" \
  2>&1 | tee "${RESULTS_DIR}/eval_tyr_sparse.log"

"${PYTHON_BIN}" "${SUMMARY_RUNNER}" summarize \
  --csv "tyr_sparse=${RESULTS_DIR}/tyr_sparse.csv" \
  --output_csv "${RESULTS_DIR}/summary_task_fair.csv" \
  --output_json "${RESULTS_DIR}/summary_task_fair.json"

"${PYTHON_BIN}" "${SUMMARY_RUNNER}" model-stats \
  --model_name "${MODEL_NAME_OR_PATH}" \
  --output_json "${RESULTS_DIR}/teacher_model_stats.json" \
  --torch_dtype "${TORCH_DTYPE}" || true
"${PYTHON_BIN}" "${SUMMARY_RUNNER}" model-stats \
  --model_name "${MATERIALIZED_MODEL_DIR}" \
  --output_json "${RESULTS_DIR}/tyr_sparse_model_stats.json" \
  --torch_dtype "${TORCH_DTYPE}" || true

cat > "${RESULTS_DIR}/run_manifest.json" <<EOF
{
  "method": "Tyr-the-Pruner",
  "model_label": "${MODEL_LABEL}",
  "model_name_or_path": "${MODEL_NAME_OR_PATH}",
  "sparsity": "${SPARSITY}",
  "run_tag": "${RUN_TAG}",
  "out_root": "${OUT_ROOT}",
  "results_dir": "${RESULTS_DIR}",
  "materialized_model_dir": "${MATERIALIZED_MODEL_DIR}",
  "final_supernet_dir": "${final_supernet_dir}",
  "final_config_name": "${final_config_name}",
  "datasets": "${DATASETS}",
  "calibration_data": "${TRAIN_DATA}",
  "calibration_cache": "${CALIBRATION_CACHE}"
}
EOF

echo "[TyrFair] done"
echo "[TyrFair] model: ${MATERIALIZED_MODEL_DIR}"
echo "[TyrFair] summary: ${RESULTS_DIR}/summary_task_fair.csv"
