#!/usr/bin/env bash
#SBATCH --job-name=fad-budgets
#SBATCH --array=0-3
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
matrix=("15 22" "20 19" "25 17" "30 15")
read -r budget prototypes <<< "${matrix[${SLURM_ARRAY_TASK_ID}]}"
suite_id="${FAD_SUITE_ID:-portfolio_${SLURM_ARRAY_JOB_ID}}"
run_name="3b_b${budget}_k${prototypes}_s44_fad"

bash "${repo_root}/reproduction/fad/scripts/run_latest_fad_train.sh" \
  llama32_3b "${budget}" "${prototypes}" 44 fad velocity_pca \
  "${run_name}" "${suite_id}"
