# File guide

## Portfolio website

| Path | Purpose |
|---|---|
| `index.html` | Single-page research narrative and accessible semantic structure. |
| `styles.css` | Responsive visual system, theme, cards, charts, and gallery. |
| `app.js` | Budget selector, exit distribution, SVG result chart, theme, and reveal behavior. |
| `assets/figures/` | Web-optimized PNG previews generated from the supplied paper figures. |
| `assets/paper/` | Original supplied PDF/PNG figures plus the manuscript and bibliography snapshot. |

## Data

| Path | Purpose |
|---|---|
| `data/processed/static_budget_results.csv` | One row per method and budget, including all seven tasks and macro accuracy. |
| `data/processed/adaptive_exit_budget_results.csv` | Four selected controllers, exit rates, average depth, speed, and accuracy deltas. |
| `data/processed/runtime_25pct.csv` | Correctly separated aggregate and task-geomean runtime measures. |
| `data/processed/paper_comparison.csv` | 3B and 8B cross-method macro table. |
| `data/processed/mmlu_8b_15pct.csv` | 8B 15% MMLU exact and normalized scoring. |
| `data/processed/multiseed_3b_20pct.csv` | Paper-designated three-seed mean and standard deviation. |
| `data/raw/fad_static/` | Seven per-task evaluator JSON files and PPL summaries at each budget. |
| `data/raw/adaptive_exit/` | Selected controller and runtime artifacts, plus paired and 25% runtime summaries. |
| `data/raw/baselines/` | Paper-era FLAP, Týr, and Streamline evaluator summaries. |

## FAD reproduction

| Path | Purpose |
|---|---|
| `reproduction/fad/config/budgets.csv` | Budget-to-prototype mapping (`22/19/17/15`). |
| `reproduction/fad/config/experiment_matrix.tsv` | Selected recovery hyperparameters and loss switches. |
| `reproduction/fad/scripts/run_policy_prep.sh` | Builds calibration split, functional distances, grouping views, and selected policy. |
| `reproduction/fad/scripts/run_latest_fad_train.sh` | Launches one selected structural recovery run. |
| `reproduction/fad/scripts/run_all_budgets.sh` | Runs 15/20/25/30% sequentially with explicit logs. |
| `reproduction/fad/scripts/run_adaptive_exit_budget.sh` | Evaluates a deploy bundle with its saved controller. |
| `reproduction/fad/slurm/fad_budgets.sbatch` | Four-GPU Slurm entry point. |
| `reproduction/fad/core/newthesis_pipeline_final_llama.py` | Main FAD training/export pipeline snapshot. |
| `reproduction/fad/core/thesis_common_final_llama.py` | Shared data, model, and evaluation helpers. |
| `reproduction/fad/core/ffn_functional_redundancy_ddp.py` | Distributed functional-distance measurement. |
| `reproduction/fad/core/train_oracle_distilled_exit_policy_final_llama.py` | Four-feature exit-controller training. |
| `reproduction/fad/core/runtime_confidence_exit_eval_final_llama.py` | Runtime controller execution and evaluation. |
| `reproduction/fad/core/fad_weight_quantization.py` | Optional FAD weight-quantization utilities. |

The other files in `reproduction/fad/core/` are policy analysis, split construction, validation, strategy selection, and threshold sweeps used by these launchers.

## Baseline reproduction

| Path | Purpose |
|---|---|
| `reproduction/baselines/run_budget.sh` | Validates a method/budget request and dispatches it. |
| `reproduction/baselines/flap/` | FLAP channel-pruning adapter. |
| `reproduction/baselines/tyr/` | Týr calibration-cache and sparse-model materialization adapters. |
| `reproduction/baselines/llm-streamline/` | Layer-sharing adapter. |
| `reproduction/baselines/common/eval.py` | Shared seven-task zero-shot evaluator. |

## 25% web deployment

| Path | Purpose |
|---|---|
| `deployment/web-demo/server.py` | Standard-library HTTP API, model loading, scoring, serialization lock, and mock mode. |
| `deployment/web-demo/static/` | Offline comparison interface for curated multiple-choice questions. |
| `deployment/web-demo/runtime_code/` | Minimal FAD modules needed to deserialize and execute the deploy bundle. |
| `deployment/web-demo/start.sh` | Real CUDA launch using environment-provided model paths. |
| `deployment/web-demo/start-mock.sh` | Weight-free UI preview. |
| `deployment/web-demo/smoke_test.py` | Health, metadata, question, and comparison endpoint checks. |
| `deployment/web-demo/Dockerfile` | NVIDIA-runtime-compatible container recipe. |
| `deployment/web-demo/models/README.md` | Required artifact names and verified SHA-256 values. |

## Quality controls

| Path | Purpose |
|---|---|
| `scripts/verify_data.py` | Recomputes macros/speed ratios, validates links and scans public files for local paths. |
The portfolio is published directly from the `main` branch root through GitHub Pages. This avoids a generated build layer: the public files are exactly the reviewed repository files.
