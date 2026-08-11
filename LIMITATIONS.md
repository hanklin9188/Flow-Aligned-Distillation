# Limitations and Claim Boundaries

## Historical baseline comparability

The archived baseline results were produced during the paper-era workflow. Their historical answer-length-normalization behavior is not fully identical to the current FAD artifact protocol. They are retained for context, not represented as a completed matched rerun.

## Selected checkpoints

The README snapshot reports selected artifacts rather than claiming that every table entry is a multi-seed estimate. Where multi-seed evidence exists, it should be cited from its dedicated artifact; otherwise a selected run must not be described as a population estimate.

## Runtime scope

The archived 25% runtime decomposition is hardware-, precision-, batch-, implementation-, and workload-specific. Aggregate throughput and task-wise geometric-mean speedup are different summaries. Neither value should be extrapolated to other accelerators, serving stacks, batch regimes, or free-form generation workloads.

## Evaluation scope

The headline accuracy table covers seven zero-shot multiple-choice benchmarks. It does not establish free-form instruction quality, long-context behavior, safety, factuality, coding ability, or general chat quality.

## Adaptive exit scope

The controller is evaluated against the retained multiple-choice artifacts and candidate exits 16, 20, 24, and 28. Its behavior should not be assumed to transfer unchanged to another model, task distribution, decoding strategy, or calibration set.

## External assets

A clean checkout cannot reconstruct Meta model weights, full benchmark corpora, merged checkpoints, or the complete deployable student bundle. These must be obtained separately under their own licenses and access terms.

## Research status

The repository is a public research artifact. Publication, review, or acceptance status is deliberately not asserted in public-facing documentation. Repository availability must not be interpreted as a venue decision or endorsement.

## Licensing

No broad open-source grant is implied for original unpublished FAD material unless a file explicitly states one. Third-party software, models, and datasets retain their own licenses.
