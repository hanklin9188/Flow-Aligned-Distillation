# Reproducibility and claim audit

This document separates observations directly supported by the archived artifacts from claims that require a fresh controlled rerun.

## Confirmed from included artifacts

- The selected FAD checkpoints at 15/20/25/30% have seven-task macros 85.38/85.66/85.03/83.20%.
- Adaptive exit changes those scores to 85.41/85.53/84.94/82.89%; all changes are within 0.5 percentage points.
- The 25% controller executes an average of 16.39 of 28 layers and exits 93.50% of examples at layer 16.
- In the same H100 batch-1 teacher/FAD job, 25% FAD-AE reaches 81.843 aggregate samples/s versus 55.878 for the teacher.
- The aggregate speed ratio is 1.46469×; the geometric mean of per-task speed ratios is 1.38828×.
- The compact pairing counts sum to 19,149 and give a +353 student-only-minus-teacher-only advantage.

`python scripts/verify_data.py` recalculates these relationships instead of only checking copied headline strings.

## Corrected label

The manuscript snapshot describes about `1.463×` as a task-wise geometric mean. `runtime_decomposition_25p.json` exposes both components and shows that this value is the aggregate throughput ratio. The geometric mean is 1.388×. The website and processed CSV use the corrected labels; the manuscript snapshot remains a historical source artifact.

## Baseline protocol caveat

Current FAD per-task JSON records `length_norm=none`. The archived baseline wrapper version defaulted to `avg`, and its result JSON omitted this field. Consequently:

- the archived numbers are valid records of those executions;
- method orderings may be shown as paper-era reproduced results with this caveat;
- they are not sufficient for a strict same-evaluator-contract claim;
- the public adapters now use `none`, and a full rerun should be performed before a final publication claim.

## Selection and scope caveats

- Latest single-seed FAD checkpoints and paper-designated multi-seed experiments are reported separately.
- Multiple-choice accuracy does not establish free-form generation quality.
- Throughput is hardware, batch-size, sequence-length, and implementation dependent.
- Adaptive exit saves executed depth; structural FFN sharing saves stored parameters. These are independent axes and must not be added as one “compression percentage.”
- Model weights are checksummed but absent, so this public repository verifies data relationships and code integrity, not bit-for-bit inference without separately licensed artifacts.

