# Limitations and Claim Boundaries

## Primary controlled study

The current primary claim is based on six objectives and three independently trained seeds at one Llama-3.2-3B `K=19` architecture. Training-seed variability and evaluation-sample bootstrap uncertainty are separate quantities. This evidence does not establish dominance for every backbone, compression level, data distribution, or metric.

FAD is task-aligned. The paper's C4 and WikiText-2 results show substantial generic-language degradation after structural recovery; mild C4 replay improves but does not eliminate that gap. At `K=17` and `K=15`, FAD is not the best MMLU objective, so the main conclusion is anchored at `K=19`.

## Historical baseline comparability

The archived baseline results were produced during the paper-era workflow. Their historical answer-length-normalization behavior is not fully identical to the current FAD artifact protocol. They are retained for context, not represented as a completed matched rerun.

## Selected checkpoints and external comparisons

Only the primary objective table is a complete six-objective, three-seed comparison. Component interventions, transport diagnostics, projection/calibration sensitivity, external 3B/8B baselines, and some replay results are selected or single-run analyses unless explicitly labeled otherwise.

## Runtime scope

The archived 25% runtime decomposition is hardware-, precision-, batch-, implementation-, and workload-specific. Aggregate throughput and task-wise geometric-mean speedup are different summaries. Neither value should be extrapolated to other accelerators, serving stacks, batch regimes, or free-form generation workloads.

## Evaluation scope

The controlled study covers seven in-domain multiple-choice tasks and MMLU 0-/5-shot evaluation. It does not establish free-form instruction quality, long-context behavior, safety, factuality, coding ability, or general chat quality.

## Adaptive exit scope

The controller is evaluated against the retained multiple-choice artifacts and candidate exits 16, 20, 24, and 28. Its behavior should not be assumed to transfer unchanged to another model, task distribution, decoding strategy, or calibration set.

## External assets

A clean checkout cannot reconstruct Meta model weights, full benchmark corpora, merged checkpoints, or the complete deployable student bundle. These must be obtained separately under their own licenses and access terms.

## Research status

The repository is a public research artifact. Publication, review, or acceptance status is deliberately not asserted in public-facing documentation. Repository availability must not be interpreted as a venue decision or endorsement.

## Licensing

No broad open-source grant is implied for original unpublished FAD material unless a file explicitly states one. Third-party software, models, and datasets retain their own licenses.
