# Data card

This directory separates compact, analysis-ready tables from the source artifacts used to audit them.

- `processed/static_budget_results.csv`: all seven task accuracies for the selected Llama-3.2-3B teacher, FAD, FLAP, Týr, and LLM-Streamline checkpoints.
- `processed/adaptive_exit_budget_results.csv`: the 15/20/25/30% adaptive-exit frontier.
- `processed/runtime_25pct.csv`: the 25% runtime comparison. It deliberately stores **both** speedup definitions.
- `processed/paper_comparison.csv`: cross-backbone macro comparison used by the portfolio.
- `processed/mmlu_8b_15pct.csv`: 57-subject MMLU result at the matched 15% budget.
- `processed/multiseed_3b_20pct.csv`: the paper-designated seeds 42/43/44 result.
- `raw/fad_static/`: untouched evaluator JSON and PPL summaries for each selected FAD checkpoint.
- `raw/adaptive_exit/`: untouched controller, runtime and budget summary artifacts. The 29 MB question-level paired file is intentionally reduced to `paired_question_summary.json`; it contains no benchmark question text.
- `raw/baselines/`: untouched `summary_task_fair.json` files from the three reproduced baselines.

## Metric contract

`macro_accuracy` is the unweighted mean over PIQA, Social-IQA, WinoGrande, ARC-Challenge, ARC-Easy, HellaSwag and OpenBookQA. `aggregate_samples_per_s` is total questions divided by total elapsed time. `task_geomean_speedup_vs_teacher` first computes a speed ratio per dataset and then takes the geometric mean. Those two speed metrics are not interchangeable.

The current `fad.tex` prose calls 1.463× a task-wise geometric mean. The source JSON shows that 1.465× is the aggregate throughput ratio, while the task-wise geometric mean is 1.388×. This repository preserves both values and uses their correct names.
