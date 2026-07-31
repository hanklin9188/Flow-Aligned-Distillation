# Results and provenance

## Latest Llama-3.2-3B budget sweep

| Compression | PIQA | SIQA | Wino | ARC-C | ARC-E | Hella | OBQA | Macro |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15% | 87.05 | 81.06 | 86.42 | 76.28 | 87.50 | 94.55 | 84.80 | **85.38** |
| 20% | 86.40 | 79.99 | 86.74 | 78.24 | 88.68 | 94.59 | 85.00 | **85.66** |
| 25% | 85.96 | 79.02 | 86.74 | 77.30 | 88.68 | 94.12 | 83.40 | **85.03** |
| 30% | 85.85 | 79.12 | 82.79 | 74.57 | 86.41 | 93.45 | 80.20 | **83.20** |

Machine-readable rows and selection notes are in [`static_budget_results.csv`](../data/processed/static_budget_results.csv). The corresponding evaluator JSON files are under `data/raw/fad_static/{15,20,25,30}pct/`.

## Adaptive exit

| Compression | Static | Adaptive | Δ | Avg. exit | Layer saving | Speedup vs own full |
|---:|---:|---:|---:|---:|---:|---:|
| 15% | 85.38 | **85.41** | +0.04 pp | 17.32 | 38.16% | 1.486× |
| 20% | 85.66 | **85.53** | −0.14 pp | 18.16 | 35.16% | 1.399× |
| 25% | 85.03 | **84.94** | −0.10 pp | 16.39 | 41.47% | 1.573× |
| 30% | 83.20 | **82.89** | −0.31 pp | 17.16 | 38.70% | 1.488× |

All four controllers satisfy the predeclared −0.5 pp safety bound. At 25%, 93.50% of samples exit at layer 16, 4.18% at 20, 1.43% at 24, and 0.88% reach 28. Source: [`adaptive_exit_budget_results.csv`](../data/processed/adaptive_exit_budget_results.csv) and the controller JSON under `data/raw/adaptive_exit/`.

## 25% matched runtime

| Design | Macro | Aggregate samples/s | Task geomean samples/s | Geomean speedup | Aggregate speedup |
|---|---:|---:|---:|---:|---:|
| Merged teacher | 82.29 | 55.88 | 66.29 | 1.000× | 1.000× |
| FAD static | 85.03 | 52.06 | 60.21 | 0.908× | 0.932× |
| FAD-AE | **84.94** | **81.84** | **92.02** | **1.388×** | **1.465×** |

The three rows above come from the same H100, batch-size-1 job. Baseline throughput rows in the CSV came from separate matched jobs and therefore have only task-geomean ratios. See [`runtime_25pct.csv`](../data/processed/runtime_25pct.csv).

## Student–teacher pairing

The compact paired summary covers 19,149 multiple-choice examples:

| Outcome | Count |
|---|---:|
| Both correct | 15,890 |
| Student only correct | 1,189 |
| Teacher only correct | 836 |
| Both wrong | 1,234 |

The adaptive student's weighted accuracy is 89.19%, versus 87.35% for the teacher in this paired artifact. The source is [`paired_question_summary.json`](../data/raw/adaptive_exit/paired_question_summary.json). Benchmark prompts and per-question records are intentionally excluded from the public portfolio.

## Other paper tables

- **Llama-3.1-8B, 15% compression, seven-task macro:** FAD 87.67, LLM-Streamline 86.86, Týr 83.61, FLAP 79.83.
- **Llama-3.1-8B, 20% compression, seven-task macro:** FAD 87.66, LLM-Streamline 86.87, Týr 81.72, FLAP 76.99.
- **Llama-3.1-8B, 15% MMLU, exact/normalized:** FAD 61.13/63.05, teacher 60.72/62.35, LLM-Streamline 58.01/60.42, FLAP 50.79/51.30, Týr 47.71/48.46.
- **Llama-3.2-3B, 20%, paper-designated seeds 42/43/44:** FAD 82.70±0.30, LLM-Streamline 81.64±0.16, Týr 75.42±1.11, FLAP 68.03±0.33.

The machine-readable tables are [`paper_comparison.csv`](../data/processed/paper_comparison.csv), [`mmlu_8b_15pct.csv`](../data/processed/mmlu_8b_15pct.csv), and [`multiseed_3b_20pct.csv`](../data/processed/multiseed_3b_20pct.csv). The multi-seed table is the paper-designated experiment; the latest seed-44 budget sweep is a separate, newer checkpoint family and must not be substituted into its mean.

