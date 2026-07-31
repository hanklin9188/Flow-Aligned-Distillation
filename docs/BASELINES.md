# Baseline reproduction

The repository does not vendor third-party projects. It provides small adapters around audited official revisions so ownership and upstream licenses remain clear.

| Method | Venue | Paper | Official code | Audited commit |
|---|---|---|---|---|
| FLAP | AAAI 2024 | [arXiv:2312.11983](https://arxiv.org/abs/2312.11983) | [CASIA-LMC-Lab/FLAP](https://github.com/CASIA-LMC-Lab/FLAP) | `3bb57db3449dd2fa04a5c2192de80e87e33be2b1` |
| Týr-the-Pruner | NeurIPS 2025 | [arXiv:2503.09657](https://arxiv.org/abs/2503.09657) | [AMD-AGI/Tyr-the-Pruner](https://github.com/AMD-AGI/Tyr-the-Pruner) | `cc73bc219457bf5f373a06fd33fd1592c7a2c98c` |
| LLM-Streamline | ICLR 2025 Spotlight | [OpenReview](https://openreview.net/forum?id=IC5RJvRoMp) | [RUCKBReasoning/LLM-Streamline](https://github.com/RUCKBReasoning/LLM-Streamline) | `adf50baa83a1fabb644a2bd165fb9636838ff00b` |

## Common contract for a fresh rerun

The public wrappers use the same teacher checkpoint, calibration corpus path, seven test files, budget, and evaluator. Multiple-choice scoring uses `length_norm=none`. `run_budget.sh` dispatches to each method-specific adapter:

```bash
export BASELINE_WORKSPACE_ROOT=/scratch/$USER/fad-baselines
export TEACHER_CKPT=/models/llama32-3b-commonsense-merged
export TRAIN_DATA=/datasets/commonsense_170k.json
export TEST_DATA_ROOT=/datasets/seven-task

export FLAP_ROOT=/src/FLAP
bash reproduction/baselines/run_budget.sh flap 25

export TYR_ROOT=/src/Tyr-the-Pruner
bash reproduction/baselines/run_budget.sh tyr 25

export STREAMLINE_ROOT=/src/LLM-Streamline
bash reproduction/baselines/run_budget.sh streamline 25
```

Before running, verify the checkout with `git -C "$FLAP_ROOT" rev-parse HEAD` (and similarly for the other two projects). Each adapter records the generated sparse/shared structure and runs the same seven-task evaluator in `reproduction/baselines/common/eval.py`.

## Archived versus rerun data

`data/raw/baselines/` contains selected paper-era output JSON. These files are useful for provenance but do not record their former `length_norm` argument. The new wrappers deliberately use `none` to match current FAD output. Results from a new run should be stored separately and should not overwrite archived artifacts.

