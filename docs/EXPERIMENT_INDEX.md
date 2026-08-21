# ICLR 2027 experiment and artifact index

This index maps every empirical block in `paper/fad.tex` to its compact public evidence, plotted asset, and implementation. Model checkpoints and large raw evaluator dumps are intentionally absent.

| Paper evidence | Public data | Figure | Analysis / execution code |
|---|---|---|---|
| Six objectives × three seeds | `data/paper/all_seed_results.csv`, `training_seed_summary.csv`, `paired_bootstrap.json` | table only | `experiments/iclr2027/analysis/analyze_p1.py`, `reproduction/fad/core/newthesis_pipeline_final_llama.py` |
| MMLU domain consistency | `data/paper/mmlu_domains.csv`, `mmlu_subject_deltas.csv` | table only | `experiments/iclr2027/analysis/analyze_p1.py`, `reproduction/fad/core/lm_eval_ours_deploy.py` |
| Projection intervention | `data/paper/performance.csv` | `component_projection.pdf` | `scripts/generate_paper_figures.py` |
| Coordinate-basis intervention | `data/paper/performance.csv` | `component_coordinates.pdf` | `scripts/generate_paper_figures.py` |
| Within-subspace metric intervention | `data/paper/performance.csv` | `component_metrics.pdf` | `scripts/generate_paper_figures.py` |
| Held-out subspace energy | `data/paper/final_completion_results.json`, `trajectory_summary.csv` | table only | `experiments/iclr2027/analysis/subspace_covariance_audit.py` |
| PIQA transport fidelity | `data/paper/trajectory_summary.csv` | `transport_teacher_metric.pdf`, `transport_cosine.pdf`, `transport_norm_ratio.pdf` | `scripts/generate_paper_figures.py` |
| Teacher covariance scale | `data/paper/teacher_scale_summary.json`; frozen tensor excluded | `teacher_scale_D_log_heatmap.png`, `teacher_scale_inv_std_heatmap.png` | `experiments/iclr2027/analysis/analyze_teacher_scale.py` |
| Projection rank and calibration | `data/paper/performance.csv`, `calibration_sensitivity.csv` | table only | geometry/training configs in `experiments/iclr2027/config/` |
| Compression stress test (`K=19/17/15`) | `data/paper/compression_objectives.csv` | `compression_commonsense.pdf`, `compression_mmlu0.pdf`, `compression_mmlu5.pdf` | `scripts/generate_paper_figures.py` |
| Nested structural restriction (`K=22/19/17/15`) | `data/paper/nested_budget_curve.json` | `nested_fad_loss.pdf`, `nested_commonsense.pdf` | `scripts/generate_paper_figures.py` |
| Generic WikiText-2/C4 likelihood | `data/paper/perplexity.csv`, `all_seed_results.csv` | table only | pipeline evaluation code and final-completion analysis |
| C4 replay mitigation | `data/paper/performance.csv`, `perplexity.csv`, `final_completion_results.json` | `c4_mmlu_pareto_revised.pdf` | `scripts/generate_paper_figures.py` |
| Sharing-policy comparison | `data/paper/sharing_policy_comparison.csv`; policy hash in public manifest | table only | functional observations and policy code under `reproduction/fad/core/` |
| Replacement-cost geometry | `data/paper/replacement_cost.npy`, `replacement_cost_heatmap.json` | `replacement_cost_heatmap.pdf` | `experiments/iclr2027/analysis/plot_replacement_cost_geometry.py`, `scripts/generate_paper_figures.py` |
| External 3B/8B compression comparison | `data/processed/static_budget_results.csv`, `paper_comparison.csv`, `mmlu_8b_15pct.csv` | table only | adapters under `reproduction/baselines/` |

## Core execution chain

1. `train_teacher_lora.py` fine-tunes and merges the teacher (weights not redistributed).
2. `ffn_functional_redundancy_ddp.py` measures directed FFN replacement costs.
3. `newthesis_pipeline_final_llama.py` builds the geometry, materializes shared FFN banks, trains the recovery objectives, selects a validation checkpoint, and exports a deploy bundle.
4. `lm_eval_ours_deploy.py` performs complete MMLU evaluation.
5. `experiments/iclr2027/analysis/` audits seeds, samples, covariance structure, and figure-specific diagnostics.

The exact primary contract is recorded in `experiments/iclr2027/provenance/primary_k19_manifest.json`. Paths in public configs are repository-relative placeholders for separately licensed assets. Benchmark question text is excluded; the PIQA decontamination artifact retains only overlap counts and source indices.
