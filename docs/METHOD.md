# Method

## 1. Structural compression

FAD treats FFN reuse as a functional grouping problem. A hard input gate and held-out functional distances identify late layers whose feed-forward transformations can share a prototype. A 28-layer Llama-3.2-3B model becomes a Transformer with `K` unique FFNs while retaining all attention blocks and residual depth.

| Budget | `K` | Shared late-layer structure |
|---:|---:|---|
| 15% | 22 | selected restricted-late policy |
| 20% | 19 | selected restricted-late policy |
| 25% | 17 | private 0–13, shared 14–18, shared 19–26, private 27 |
| 30% | 15 | selected restricted-late policy |

The `K` values are the actual selected prototypes, not rounded parameter percentages. In the 25% deploy checkpoint, removing 11 FFN parameter sets saves approximately 0.83B of about 3.21B parameters (25.8%).

## 2. Flow-aligned recovery

For a teacher layer, let its FFN residual update be the local transport vector. FAD projects those updates into a shared velocity-PCA basis and estimates a layer-specific teacher covariance. Student deviations are whitened by the inverse teacher standard deviation, so a mismatch in a stable direction is penalized more than the same raw mismatch in a noisy direction.

The selected recipe uses:

- decision-token cross entropy, weight `1.0`;
- teacher-whitened point CORE, weight `1.0`;
- reliability weighting derived from teacher correctness/confidence;
- no token KL, hidden MSE, geodesic CORE, or manifold CORE;
- LoRA rank/alpha `128/128` on the shared bank;
- 7,500 training steps, validation every 500 steps, best decision CE selection;
- bank learning rate `3e-5`, adapter learning rate `8e-5`, 750-step warmup, cosine decay, weight decay `0.01`;
- four GPUs and effective global batch size 32 in the selected runs.

The executable implementation is [`newthesis_pipeline_final_llama.py`](../reproduction/fad/core/newthesis_pipeline_final_llama.py); the selected matrix is [`experiment_matrix.tsv`](../reproduction/fad/config/experiment_matrix.tsv).

## 3. Adaptive exit

The runtime evaluator scores candidate exits at layers 16, 20, and 24, with layer 28 as fallback. The controller receives only four deployment-time features:

1. normalized answer entropy;
2. top-answer surprisal;
3. Jensen–Shannon change from the previous exit;
4. information-energy gap.

A standardized linear sigmoid controller uses decision threshold `0.50`; a separate answer-confidence gate uses `0.90`. The teacher and ground-truth label are not needed during deployment. The controller training script is [`train_oracle_distilled_exit_policy_final_llama.py`](../reproduction/fad/core/train_oracle_distilled_exit_policy_final_llama.py), and the runtime is [`runtime_confidence_exit_eval_final_llama.py`](../reproduction/fad/core/runtime_confidence_exit_eval_final_llama.py).

## 4. Evaluation contract

- model: merged Llama-3.2-3B teacher and structurally shared students;
- seed: 44 for the latest four-budget FAD sweep;
- tasks: PIQA, Social-IQA, WinoGrande, ARC-Challenge, ARC-Easy, HellaSwag, OpenBookQA;
- scoring: multiple-choice log probability, `length_norm=none` in current FAD artifacts;
- headline score: unweighted macro mean of the seven task accuracies;
- adaptive safety criterion: within 0.5 percentage points of its corresponding static checkpoint.

See [AUDIT.md](AUDIT.md) for the distinction between paper-era and current-wrapper baseline contracts.

