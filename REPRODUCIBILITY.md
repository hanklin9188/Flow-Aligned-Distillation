# Reproducibility

This document defines what can be verified from a clean checkout, what requires external assets, and what evidence is allowed to support a public claim.

The current ICLR 2027 release adds `scripts/verify_paper_release.py`, the public primary manifest, all paper-referenced figures, and compact tables under `data/paper/`. The older adaptive-exit portfolio evidence remains available but is not mixed into the primary controlled objective study.

## 1. Verification tiers

### Tier A — public integrity checks, no GPU

```bash
python scripts/verify_data.py
python scripts/verify_paper_release.py
```

The verifier re-derives the public FAD macro scores and runtime summaries from retained artifacts, validates adaptive-exit and paired-question accounting, checks website-local links, and rejects common private path markers.

### Tier B — weight-free deployment smoke test

```bash
bash deployment/web-demo/start-mock.sh 8765 &
server_pid=$!
python deployment/web-demo/smoke_test.py --base_url http://127.0.0.1:8765
kill "$server_pid"
```

Mock mode exists to validate the API and interface contract. It is not live model inference and cannot support a quality or performance claim.

### Tier C — GPU research reproduction

Obtain the model weights and datasets under their own terms, configure the paths described in [`reproduction/README.md`](reproduction/README.md), and submit the workload through Slurm:

```bash
sbatch reproduction/fad/slurm/fad_budgets.sbatch
```

Do not run GPU- or CPU-intensive experiments on a cluster login node.

## 2. Artifact authority

A public numeric statement should be traceable through the following chain:

```text
raw evaluator/controller artifact
→ processed CSV
→ scripts/verify_data.py
→ README or docs result table
```

If the raw artifact and a prose statement disagree, the raw artifact plus executable verifier take precedence.

## 3. Protocol controls

Record and freeze at least:

- model identifier and exact revision;
- compression budget and FFN assignment;
- training seed and selected checkpoint;
- dataset split and evaluator revision;
- answer-length normalization mode;
- precision, batch size, hardware, and runtime settings;
- adaptive-exit controller artifact and threshold;
- exact command and git commit.

Historical baseline outputs that lack a field required by the current protocol must be labelled historical rather than silently treated as matched.

## 4. External assets

This repository does not redistribute model weights, full datasets, or the complete deployable bundle. Expected deployment artifact names and checksums are documented in [`deployment/web-demo/models/README.md`](deployment/web-demo/models/README.md).

## 5. Expected outputs

A complete reproduction should preserve:

- per-task evaluator JSON;
- aggregate and task-wise runtime summaries;
- selected model and controller metadata;
- processed CSV tables;
- stdout/stderr logs from scheduled jobs;
- the git revision and environment specification.

## 6. Failure handling

Failed, incomplete, or infrastructure-invalid runs must not be promoted into a headline table. Preserve their logs separately and state why they were excluded.
