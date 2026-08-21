#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(os.environ.get("FAD_WORKSPACE_ROOT", "external/internal_workspace"))
POINTS = [
    {
        "k": 22,
        "compression": 13.95,
        "run": ROOT / "new_idea/out/newthesis_final_llama_restricted_late_ifgate_decision_proto22_15pct_s44_budget_sweep_20260717_063556",
        "results": ROOT / "new_idea/results/newthesis_final_llama_restricted_late_ifgate_decision_proto22_15pct_s44_budget_sweep_20260717_063556",
    },
    {
        "k": 19,
        "compression": 20.93,
        "run": ROOT / "iclr2027/fad_experiments/runs/opt3b_fad",
        "results": ROOT / "iclr2027/fad_experiments/results/runs/opt3b_fad",
    },
    {
        "k": 17,
        "compression": 25.58,
        "run": ROOT / "new_idea/out/newthesis_final_llama_restricted_late_ifgate_decision_proto17_25pct_s44_budget_sweep_20260717_063556",
        "results": ROOT / "new_idea/results/newthesis_final_llama_restricted_late_ifgate_decision_proto17_25pct_s44_budget_sweep_20260717_063556",
    },
    {
        "k": 15,
        "compression": 30.23,
        "run": ROOT / "new_idea/out/newthesis_final_llama_restricted_late_ifgate_decision_proto15_30pct_s44_budget_sweep_20260717_063556",
        "results": ROOT / "new_idea/results/newthesis_final_llama_restricted_late_ifgate_decision_proto15_30pct_s44_budget_sweep_20260717_063556",
    },
]
TASKS = ["ARC-Challenge", "ARC-Easy", "hellaswag", "openbookqa", "piqa", "social_i_qa", "winogrande"]


def read_accuracy(results_dir: Path) -> float:
    values = []
    for task in TASKS:
        payload = json.loads((results_dir / f"phase1_5_{task}.json").read_text(encoding="utf-8"))
        values.append(float(payload["accuracy"]))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-pdf", required=True)
    args = parser.parse_args()

    rows = []
    for point in POINTS:
        report_path = point["run"] / "phase1_5_pass1/compress_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        teacher = report["teacher_model"]
        if not teacher.endswith("lora_20260410_064043/merged_teacher"):
            raise RuntimeError(f"unexpected teacher for K={point['k']}: {teacher}")
        rows.append(
            {
                "k": point["k"],
                "whole_model_compression_pct": point["compression"],
                "fad_loss": float(report["last_train_core_loss"]),
                "macro_accuracy_pct": 100.0 * read_accuracy(point["results"]),
                "best_val_step": int(report["best_val_step"]),
                "decision_val_loss": float(report["best_val_loss"]),
                "teacher": teacher,
                "report": str(report_path),
            }
        )

    compression = np.array([row["whole_model_compression_pct"] for row in rows])
    fad_loss = np.array([row["fad_loss"] for row in rows])
    accuracy = np.array([row["macro_accuracy_pct"] for row in rows])
    if not np.all(np.diff(fad_loss) > 0):
        raise RuntimeError("FAD loss is not strictly increasing")
    if not np.all(np.diff(accuracy) < 0):
        raise RuntimeError("macro accuracy is not strictly decreasing")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.5,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    blue, red = "#3B6FB6", "#C94C4C"
    fig, ax_loss = plt.subplots(figsize=(6.8, 4.05))
    ax_acc = ax_loss.twinx()

    loss_line = ax_loss.plot(
        compression, fad_loss, color=red, marker="o", markersize=7,
        linewidth=2.25, label=r"FAD loss $\mathcal{L}_{\mathrm{FAD}}$"
    )[0]
    acc_line = ax_acc.plot(
        compression, accuracy, color=blue, marker="s", markersize=6.5,
        linewidth=2.25, label="Macro accuracy"
    )[0]

    ax_loss.set_xlabel("Compression rate (%)")
    ax_loss.set_ylabel(r"FAD loss $\mathcal{L}_{\mathrm{FAD}}$", color=red)
    ax_acc.set_ylabel("Macro accuracy (%)", color=blue)
    ax_loss.tick_params(axis="y", colors=red)
    ax_acc.tick_params(axis="y", colors=blue)
    ax_loss.set_xticks(compression, [f"{x:.2f}" for x in compression])
    ax_loss.set_xlim(12.0, 31.8)
    ax_loss.set_ylim(0.095, 0.185)
    ax_acc.set_ylim(82.6, 85.8)
    ax_loss.grid(axis="both", color="#999999", alpha=0.22, linewidth=0.8)

    for x, loss, acc, row in zip(compression, fad_loss, accuracy, rows):
        ax_loss.annotate(f"{loss:.3f}", (x, loss), xytext=(0, 9),
                         textcoords="offset points", ha="center", color=red, fontsize=8.7)
        ax_acc.annotate(f"{acc:.2f}%", (x, acc), xytext=(0, -15),
                        textcoords="offset points", ha="center", color=blue, fontsize=8.7)
        ax_loss.annotate(f"K={row['k']}", (x, 0.0965), ha="center", va="bottom",
                         color="#555555", fontsize=8.3)

    ax_loss.legend([loss_line, acc_line], [loss_line.get_label(), acc_line.get_label()],
                   loc="center left", bbox_to_anchor=(0.025, 0.66), frameon=False)
    fig.tight_layout()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / "information_bottleneck_budget_curve.pdf"
    png = out / "information_bottleneck_budget_curve.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    paper_pdf = Path(args.paper_pdf)
    paper_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_pdf, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "definition": "J_MI = C_T - 0.5 * L_FAD under the fixed teacher Gaussian conditional",
        "loss_field": "compress_report.json:last_train_core_loss",
        "accuracy_definition": "unweighted macro accuracy over seven tasks",
        "rows": rows,
    }
    (out / "information_bottleneck_budget_curve.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pdf": str(pdf), "png": str(png), "paper_pdf": str(paper_pdf), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
