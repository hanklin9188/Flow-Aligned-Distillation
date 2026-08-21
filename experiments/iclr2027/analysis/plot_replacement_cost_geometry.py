#!/usr/bin/env python3
"""Render the publication version of the replacement-cost geometry figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np


PRIMARY_GROUPS = ((16, 18, "Group A: 16--18"), (19, 26, "Group B: 19--26"))
DEPTH_BOUNDARIES = (9.5, 18.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def bidirectional_cost(directed: np.ndarray) -> np.ndarray:
    if directed.ndim != 2 or directed.shape[0] != directed.shape[1]:
        raise ValueError(f"expected a square matrix, got {directed.shape}")
    paired = np.stack((directed, directed.T), axis=0)
    with np.errstate(all="ignore"):
        result = np.nanmax(paired, axis=0)
    np.fill_diagonal(result, 0.0)
    return result


def main() -> None:
    args = parse_args()
    directed = np.load(args.matrix)
    matrix = bidirectional_cost(directed)
    n_layers = matrix.shape[0]
    if n_layers != 28:
        raise ValueError(f"the primary Llama-3.2-3B figure expects 28 layers, got {n_layers}")

    off_diagonal = ~np.eye(n_layers, dtype=bool)
    values = matrix[np.isfinite(matrix) & off_diagonal]
    if values.size == 0:
        raise ValueError("replacement-cost matrix has no finite off-diagonal entries")

    q01, q50, q99 = np.percentile(values, (1, 50, 99))
    vmin, vmax = float(q01), float(q99)
    if not vmax > vmin:
        vmin, vmax = float(values.min()), float(values.max())
    if vmin < 0.0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap_name = "coolwarm"
    else:
        vmin = max(0.0, vmin)
        norm = PowerNorm(gamma=0.55, vmin=vmin, vmax=vmax)
        cmap_name = "magma"

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad("#eeeeee")

    fig, ax = plt.subplots(figsize=(6.65, 5.75), constrained_layout=True)
    image = ax.imshow(matrix, origin="upper", cmap=cmap, norm=norm,
                      interpolation="nearest", aspect="equal")

    for boundary in DEPTH_BOUNDARIES:
        ax.axvline(boundary, color="white", lw=1.15, ls=(0, (4, 3)), alpha=0.95)
        ax.axhline(boundary, color="white", lw=1.15, ls=(0, (4, 3)), alpha=0.95)

    group_colors = ("#20d9d2", "#f6e944")
    legend_handles = []
    for (start, end, label), color in zip(PRIMARY_GROUPS, group_colors):
        width = end - start + 1
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), width, width,
                               fill=False, edgecolor=color, linewidth=2.0))
        legend_handles.append(Patch(facecolor="none", edgecolor=color,
                                    linewidth=2.0, label=label))

    ticks = np.arange(0, n_layers, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(n_layers - 0.5, -0.5)
    ax.set_xlabel(r"Target layer $\ell^{\prime}$")
    ax.set_ylabel(r"Source layer $\ell$")

    # Regime names live outside the heatmap so they do not obscure cost values.
    regime_centers = ((4.5, "Early"), (14.0, "Middle"), (23.0, "Late"))
    for center, name in regime_centers:
        ax.text(center, -1.28, name, ha="center", va="bottom", fontsize=8,
                color="#333333", clip_on=False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035, extend="both")
    cbar.set_label(r"Bidirectional replacement cost $C^{\mathrm{bi}}_{\ell\ell^{\prime}}$")
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, 1.075),
              ncol=2, frameon=False, fontsize=8, borderaxespad=0.0,
              handlelength=1.6, columnspacing=1.4)

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, bbox_inches="tight")
    fig.savefig(args.output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "source_matrix": str(args.matrix),
        "matrix_shape": list(matrix.shape),
        "cost_definition": "max(C_i_to_j, C_j_to_i)",
        "finite_off_diagonal_entries": int(values.size),
        "value_min": float(values.min()),
        "value_median": float(q50),
        "value_max": float(values.max()),
        "color_limits_percentiles": {"p01": float(q01), "p99": float(q99)},
        "colormap": cmap_name,
        "primary_groups": [[start, end] for start, end, _ in PRIMARY_GROUPS],
        "depth_boundaries": list(DEPTH_BOUNDARIES),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
