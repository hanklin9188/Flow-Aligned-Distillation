# Paper source

This directory contains the current public artifact version of `fad.tex`, its bibliography and ICLR style files, and every figure referenced by the TeX source.

Build with a standard TeX installation:

```bash
pdflatex fad
bibtex fad
pdflatex fad
pdflatex fad
```

On a Slurm-managed cluster, submit this build through `sbatch` rather than compiling on a login node.

Thirteen figures are regenerated from compact public tables by `scripts/generate_paper_figures.py`. The method overview (`Figure/fad.pdf`) is a designed illustration. The two teacher-scale heatmaps require the excluded frozen geometry tensor; their analysis implementation is retained at `experiments/iclr2027/analysis/analyze_teacher_scale.py`, while the rendered figures are included here.

The public source corrects the primary private-adapter rank to 128 and uses a repository-relative replacement-cost figure path. These changes are backed by the audited public manifest.
