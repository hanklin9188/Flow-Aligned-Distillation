# Reproduction package

This directory contains the exact research pipeline snapshot plus portable wrappers for the four FAD budgets, adaptive exit, and the three external baselines.

## FAD 15/20/25/30%

The selected Llama-3.2-3B structures use `K=22/19/17/15` unique FFN prototypes over 28 decoder layers. Set paths to your own licensed model and data copies:

```bash
export FAD_WORKSPACE_ROOT=/scratch/$USER/fad-work
export FAD_TEACHER_CKPT=/models/llama32-3b-commonsense-merged
export FAD_CANONICAL_ROOT=/artifacts/fad-atlas-and-splits
export FAD_TRAIN_SPLIT=/datasets/commonsense_train.json
export FAD_VAL_SPLIT=/datasets/commonsense_val.json
export NUM_GPUS=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 PYTHON_BIN=python
bash reproduction/fad/scripts/run_policy_prep.sh llama32_3b portfolio-v1
bash reproduction/fad/scripts/run_all_budgets.sh portfolio-v1
```

The training recipe is decision-token CE + reliability-weighted, teacher-whitened point CORE; KD, hidden MSE, geodesic CORE and manifold CORE are disabled. It trains 7,500 steps, validates every 500 steps, and selects the lowest decision CE. See `config/budgets.csv` and `config/experiment_matrix.tsv`.

To evaluate an exported bundle with its saved four-feature controller:

```bash
export FAD_DEPLOY_BUNDLE=/artifacts/25pct/deploy_bundle.pt
export FAD_TEST_DATA_ROOT=/datasets/seven-task
bash reproduction/fad/scripts/run_adaptive_exit_budget.sh 25
```

## External baselines

Clone the official source at the audited commit, install its dependencies, then point the wrapper at the checkout:

| Method | Official source | Audited commit | Portfolio wrapper |
|---|---|---|---|
| FLAP (AAAI 2024) | <https://github.com/CASIA-LMC-Lab/FLAP> | `3bb57db3449dd2fa04a5c2192de80e87e33be2b1` | `baselines/flap/run.sh` |
| Týr-the-Pruner (NeurIPS 2025) | <https://github.com/AMD-AGI/Tyr-the-Pruner> | `cc73bc219457bf5f373a06fd33fd1592c7a2c98c` | `baselines/tyr/run.sh` |
| LLM-Streamline (ICLR 2025 Spotlight) | <https://github.com/RUCKBReasoning/LLM-Streamline> | `adf50baa83a1fabb644a2bd165fb9636838ff00b` | `baselines/llm-streamline/run.sh` |

Common setup:

```bash
export BASELINE_WORKSPACE_ROOT=/scratch/$USER/fad-baselines
export TEACHER_CKPT=/models/llama32-3b-commonsense-merged
export TRAIN_DATA=/datasets/commonsense_170k.json
export TEST_DATA_ROOT=/datasets/seven-task
export PYTHON_BIN=python

export FLAP_ROOT=/src/FLAP
bash reproduction/baselines/run_budget.sh flap 25

export TYR_ROOT=/src/Tyr-the-Pruner
bash reproduction/baselines/run_budget.sh tyr 25

export STREAMLINE_ROOT=/src/LLM-Streamline
bash reproduction/baselines/run_budget.sh streamline 25
```

The portable wrappers set `length_norm=none`, matching the current FAD evaluator artifacts. Archived baseline values in `data/raw/baselines` came from earlier wrappers whose default was `avg`; their JSON did not persist that setting. They are retained as the paper-era results, but a fresh all-method rerun is required before claiming a strictly identical current protocol.

Model weights and benchmark corpora are not redistributed. Users must accept the Llama license and each dataset's terms separately.
