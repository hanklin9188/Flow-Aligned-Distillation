<div align="center">

# Flow-Aligned Distillation

### Preserving layer-wise residual trajectories in compressed language models

[![Verify repository](https://github.com/hanklin9188/Flow-Aligned-Distillation/actions/workflows/verify.yml/badge.svg)](https://github.com/hanklin9188/Flow-Aligned-Distillation/actions/workflows/verify.yml)
[![Paper source](https://img.shields.io/badge/ICLR%202027-paper%20source-8a2be2)](paper/fad.tex)
[![No checkpoints](https://img.shields.io/badge/model%20weights-not%20included-555)](#artifact-boundary)

[繁體中文](README_zh-TW.md) · [Paper](paper/README.md) · [Experiment index](docs/EXPERIMENT_INDEX.md) · [Reproduce](reproduction/README.md)

</div>

Flow-Aligned Distillation (FAD) compresses a transformer by sharing FFN banks across layers, then recovers capability by aligning the teacher and student FFN residual updates in a teacher-derived low-rank geometry. This repository is the public, weight-free companion to the ICLR 2027 manuscript.

The release is organized around an auditable chain:

```text
paper claim → compact result table → analysis/plot script → scheduled experiment recipe
```

## Main controlled result

The primary study fixes one Llama-3.2-3B student architecture (`K=19`, 20.93% whole-model parameter reduction), teacher, initialization, data, optimizer, checkpoint rule, and evaluation protocol. Only the auxiliary recovery objective changes. Values are mean ± sample standard deviation over independently trained seeds 42, 43, and 44.

| Recovery objective | Commonsense | MMLU 0-shot | MMLU 5-shot |
|---|---:|---:|---:|
| CE only | 85.10 ± 0.57 | 33.37 ± 5.34 | 34.08 ± 4.47 |
| Logit KL | 84.91 ± 0.08 | 39.88 ± 1.12 | 43.38 ± 0.47 |
| Hidden-state MSE | 85.06 ± 0.07 | 44.26 ± 2.21 | 45.71 ± 2.31 |
| Ambient velocity MSE | **85.17 ± 0.39** | 40.99 ± 1.47 | 43.50 ± 1.42 |
| Projected isotropic velocity | 84.96 ± 0.06 | 45.02 ± 0.72 | 48.60 ± 1.12 |
| **FAD** | 84.84 ± 0.42 | **47.55 ± 3.84** | **52.09 ± 1.96** |

FAD improves over the strongest non-FAD mean by 2.53 points on MMLU 0-shot and 3.49 points on MMLU 5-shot, and wins against projected isotropic matching in all three training seeds. The paper also reports component interventions, MMLU-domain consistency, transport diagnostics, stronger-compression stress tests, generic-language trade-offs, and 3B/8B external compression comparisons.

## Repository map

```text
paper/                         complete current TeX source and all 16 referenced figures
data/paper/                    compact tables behind the paper claims and plots
experiments/iclr2027/          audited configs, provenance, and analysis scripts
reproduction/fad/              teacher, geometry, compression, training, and evaluation code
reproduction/baselines/        FLAP, Týr, and LLM-Streamline adapters
scripts/generate_paper_figures.py
scripts/verify_paper_release.py
docs/EXPERIMENT_INDEX.md       claim-to-artifact map
```

## Fast verification

The integrity checks need no model weights or GPU:

```bash
python scripts/verify_data.py
python scripts/verify_paper_release.py
```

Regenerating the 13 data-driven plots requires NumPy and Matplotlib:

```bash
python scripts/generate_paper_figures.py
```

On managed clusters, submit verification and plotting through the scheduler; do not run them on a login node. The full GPU workflow starts in [reproduction/README.md](reproduction/README.md).

## Provenance corrections

The public manifest distinguishes two ranks that were conflated in an internal draft:

- teacher fine-tuning LoRA: rank/alpha 64/64;
- compressed-student private adapter: rank/alpha 128/128.

The manuscript source and [public primary manifest](experiments/iclr2027/provenance/primary_k19_manifest.json) use these audited values. The manifest also records the exact backbone revision, geometry hash, compression accounting, optimization contract, and evaluation anchor.

## Artifact boundary

Included: source code, experiment configurations, compact result tables, provenance metadata, paper figures, and verification scripts.

Excluded: checkpoints, model weights, geometry tensors, full datasets, credentials, scheduler logs, and large per-question evaluator dumps. Obtain Meta Llama weights and benchmark data under their respective licenses. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [LIMITATIONS.md](LIMITATIONS.md) before citing results.

## Citation and licensing

Citation metadata is in [CITATION.cff](CITATION.cff). Original unpublished FAD material follows [NOTICE.md](NOTICE.md); third-party models, datasets, and repositories retain their own terms as listed in [THIRD_PARTY.md](THIRD_PARTY.md).
