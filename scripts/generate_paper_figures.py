#!/usr/bin/env python3
"""Regenerate the data-driven figures referenced by paper/fad.tex."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/paper"
OUTPUT = ROOT / "paper/Figure"
COLORS = {
    "ce": "#7F7F7F",
    "kl": "#4C78A8",
    "hidden": "#F58518",
    "ambient": "#54A24B",
    "isotropic": "#B279A2",
    "fad": "#D62728",
}
LABELS = {
    "ce": "CE",
    "kl": "KL",
    "hidden": "Hidden",
    "ambient": "Ambient",
    "isotropic": "Isotropic",
    "fad": "FAD",
}


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def grouped_panel(name: str, variants: list[tuple[str, str, str]], performance: dict[str, dict[str, str]]) -> None:
    metrics = (("common", "Common"), ("mmlu0", "MMLU-0"), ("mmlu5", "MMLU-5"))
    x = np.arange(len(metrics), dtype=float)
    width = 0.78 / len(variants)
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for index, (run_id, label, color) in enumerate(variants):
        values = [100.0 * float(performance[run_id][metric]) for metric, _ in metrics]
        offset = (index - (len(variants) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=label, color=color)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=5.8, frameon=False, loc="upper right")
    save(fig, name)


def component_figures() -> None:
    performance = {row["id"]: row for row in rows("performance.csv")}
    grouped_panel("component_projection", [
        ("opt3b_ambient", "Ambient", COLORS["ambient"]),
        ("opt3b_isotropic", "Projected Iso.", COLORS["isotropic"]),
    ], performance)
    grouped_panel("component_coordinates", [
        ("opt3b_coord_random", "Random", "#7F7F7F"),
        ("opt3b_coord_hidden", "Hidden PCA", "#F2CF5B"),
        ("opt3b_fad", "Teacher-Vel. PCA", COLORS["fad"]),
    ], performance)
    grouped_panel("component_metrics", [
        ("opt3b_isotropic", "Isotropic", COLORS["isotropic"]),
        ("opt3b_metric_global", "Global", COLORS["kl"]),
        ("opt3b_metric_shrink", "Shrinkage", COLORS["hidden"]),
        ("opt3b_fad", "Layer-Cond.", COLORS["fad"]),
    ], performance)


def transport_figures() -> None:
    source = {row["objective"]: row for row in rows("trajectory_summary.csv") if row["K"] == "19" and row["seed"] == "44"}
    objectives = list(COLORS)
    specs = (
        ("transport_teacher_metric", "mean_fad_error", "Teacher-standardized error", None),
        ("transport_cosine", "mean_velocity_cosine", "Velocity cosine", None),
        ("transport_norm_ratio", "mean_velocity_norm_ratio", "Velocity norm ratio", 1.0),
    )
    for filename, key, ylabel, reference in specs:
        fig, ax = plt.subplots(figsize=(3.45, 2.75))
        values = [float(source[objective][key]) for objective in objectives]
        ax.bar(np.arange(len(values)), values, color=[COLORS[objective] for objective in objectives])
        ax.set_xticks(np.arange(len(values)), [LABELS[objective] for objective in objectives], rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        if reference is not None:
            ax.axhline(reference, color="#333333", linewidth=1.0, linestyle="--")
        ax.grid(axis="y", alpha=0.22)
        save(fig, filename)


def compression_figures() -> None:
    source = [row for row in rows("compression_objectives.csv") if row["seed"] == "44" and row["objective"] in COLORS and row["objective"] != "ce"]
    specs = (
        ("compression_commonsense", "common_macro", "Commonsense accuracy (%)", (65, 88)),
        ("compression_mmlu0", "mmlu_0", "MMLU 0-shot (%)", (20, 56)),
        ("compression_mmlu5", "mmlu_5", "MMLU 5-shot (%)", (20, 58)),
    )
    for filename, key, ylabel, ylim in specs:
        fig, ax = plt.subplots(figsize=(3.45, 2.75))
        for objective in ("kl", "hidden", "ambient", "isotropic", "fad"):
            selected = sorted((row for row in source if row["objective"] == objective), key=lambda row: float(row["reduction_pct"]))
            ax.plot([float(row["reduction_pct"]) for row in selected], [100.0 * float(row[key]) for row in selected], marker="o", label=LABELS[objective], color=COLORS[objective])
        ax.set_xlabel("Whole-model reduction (%)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=5.5, frameon=False, loc="upper right")
        save(fig, filename)


def nested_figures() -> None:
    source = sorted(json.loads((DATA / "nested_budget_curve.json").read_text(encoding="utf-8"))["rows"], key=lambda row: row["whole_model_compression_pct"])
    x = [float(row["whole_model_compression_pct"]) for row in source]
    labels = [f"{value:.2f}\nK={int(row['k'])}" for value, row in zip(x, source)]
    for filename, key, ylabel, color, ylim in (
        ("nested_fad_loss", "fad_loss", "Observed FAD loss", COLORS["fad"], (0.095, 0.185)),
        ("nested_commonsense", "macro_accuracy_pct", "Commonsense accuracy (%)", COLORS["kl"], (82.5, 86.0)),
    ):
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        ax.plot(x, [float(row[key]) for row in source], marker="o", color=color)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Whole-model reduction (%)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.22)
        save(fig, filename)


def c4_tradeoff() -> None:
    performance = {row["id"]: row for row in rows("performance.csv")}
    perplexity = {row["id"]: row for row in rows("perplexity.csv")}
    variants = (
        ("opt3b_fad", "FAD anchor", "D", "#111111"),
        ("fad_c4r_mild", "Mild replay (1/10)", "*", COLORS["fad"]),
        ("fad_c4r_strong", "Strong replay (1/4)", "o", COLORS["hidden"]),
        ("fad_kl025", "FAD + KL (.25)", "s", COLORS["kl"]),
        ("fad_kl025_c4r", "FAD + KL + replay", "^", "#9467BD"),
    )
    fig, ax = plt.subplots(figsize=(5.25, 3.55))
    for run, label, marker, color in variants:
        ax.scatter(float(perplexity[run]["c4_nll"]), 100.0 * float(performance[run]["mmlu5"]), marker=marker, s=110 if marker == "*" else 48, color=color, label=label)
    ax.set_xlabel("C4 NLL (lower is better)")
    ax.set_ylabel("MMLU 5-shot (%)")
    ax.legend(loc="lower right", fontsize=7.0, frameon=False)
    ax.grid(alpha=0.22)
    save(fig, "c4_mmlu_pareto_revised")


def replacement_cost() -> None:
    directed = np.load(DATA / "replacement_cost.npy")
    matrix = np.nanmax(np.stack((directed, directed.T)), axis=0)
    np.fill_diagonal(matrix, 0.0)
    values = matrix[np.isfinite(matrix) & ~np.eye(matrix.shape[0], dtype=bool)]
    q01, q99 = np.percentile(values, (1, 99))
    norm = TwoSlopeNorm(vmin=q01, vcenter=0.0, vmax=q99) if q01 < 0.0 < q99 else PowerNorm(gamma=0.55, vmin=max(0.0, q01), vmax=q99)
    fig, ax = plt.subplots(figsize=(6.65, 5.75))
    image = ax.imshow(matrix, cmap="coolwarm" if q01 < 0.0 < q99 else "magma", norm=norm, interpolation="nearest")
    for boundary in (9.5, 18.5):
        ax.axvline(boundary, color="white", linestyle="--")
        ax.axhline(boundary, color="white", linestyle="--")
    handles = []
    for (start, end, label), color in zip(((16, 18, "Group A: 16--18"), (19, 26, "Group B: 19--26")), ("#20d9d2", "#f6e944")):
        width = end - start + 1
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), width, width, fill=False, edgecolor=color, linewidth=2.0))
        handles.append(Patch(facecolor="none", edgecolor=color, linewidth=2.0, label=label))
    fig.colorbar(image, ax=ax, label="Bidirectional replacement cost")
    ax.set_xlabel("Target layer")
    ax.set_ylabel("Source layer")
    ax.legend(handles=handles, frameon=False, fontsize=8)
    save(fig, "replacement_cost_heatmap")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    component_figures()
    transport_figures()
    compression_figures()
    nested_figures()
    c4_tradeoff()
    replacement_cost()
    print("Regenerated 13 data-driven paper figures in paper/Figure")


if __name__ == "__main__":
    main()
