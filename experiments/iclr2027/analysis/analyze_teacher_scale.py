#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze teacher-induced diagonal scale matrix D^(ell).")
    p.add_argument("--run_root", type=str, required=True)
    p.add_argument("--structure_prior", type=str, default="")
    p.add_argument("--sharing_policy", type=str, default="")
    p.add_argument("--compress_report", type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--label", type=str, default="D")
    p.add_argument("--cold_quantile", type=float, default=0.10)
    p.add_argument("--warm_quantile", type=float, default=0.90)
    p.add_argument("--top_dims", type=int, default=12)
    return p.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fmt(value: Any, digits: int = 5) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().reshape(-1)
    qs = torch.quantile(x, torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99], dtype=torch.float32))
    return {
        "min": float(x.min().item()),
        "p01": float(qs[0].item()),
        "p10": float(qs[1].item()),
        "median": float(qs[2].item()),
        "mean": float(x.mean().item()),
        "p90": float(qs[3].item()),
        "p99": float(qs[4].item()),
        "max": float(x.max().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def plot_heatmap(path: str, matrix: torch.Tensor, title: str, ylabel: str) -> None:
    if plt is None:
        return
    arr = matrix.detach().float().cpu().numpy()
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=150)
    image = ax.imshow(arr, aspect="auto", interpolation="nearest")
    if title:
        ax.set_title(title)
    ax.set_xlabel("subspace dimension")
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_layer_curves(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if plt is None or not rows:
        return
    xs = [int(row["layer"]) for row in rows]
    fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=150)
    ax1.plot(xs, [float(row["D_median"]) for row in rows], marker="o", label="D median", color="#2f6ae5")
    ax1.plot(xs, [float(row["D_p10"]) for row in rows], linestyle="--", label="D p10", color="#157f7b")
    ax1.plot(xs, [float(row["D_p90"]) for row in rows], linestyle="--", label="D p90", color="#e67e22")
    ax1.set_yscale("log")
    ax1.set_xlabel("layer")
    ax1.set_ylabel("D scale, log")
    ax2 = ax1.twinx()
    ax2.plot(xs, [float(row["inv_std_mean"]) for row in rows], marker="s", label="mean D^-1/2", color="#c0392b", alpha=0.8)
    ax2.set_ylabel("mean D^-1/2")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = os.path.abspath(args.run_root)
    prior_path = args.structure_prior or os.path.join(run_root, "phase1_atlas", "final_structure_prior.pt")
    policy_path = args.sharing_policy or os.path.join(run_root, "phase1_atlas", "sharing_policy.json")
    compress_path = args.compress_report or os.path.join(run_root, "phase1_5_pass1", "compress_report.json")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    prior = torch.load(prior_path, map_location="cpu")
    if not isinstance(prior, dict) or not torch.is_tensor(prior.get("layer_metric_diag")):
        raise RuntimeError(f"{prior_path} does not contain tensor layer_metric_diag.")

    D = prior["layer_metric_diag"].detach().float().cpu()
    eps = max(1e-12, float(prior.get("tau_eps", 1e-5)))
    inv_std = torch.rsqrt(D.clamp(min=eps))
    precision = 1.0 / D.clamp(min=eps)
    reliability = prior.get("layer_reliability", torch.ones((D.size(0),), dtype=torch.float32)).detach().float().cpu()
    regimes = prior.get("layer_to_regime", [""] * int(D.size(0)))
    regimes = list(regimes) if isinstance(regimes, list) else [""] * int(D.size(0))

    policy = load_json(policy_path)
    compress = load_json(compress_path)
    layer_to_group: Dict[int, int] = {}
    for group in policy.get("groups", []) if isinstance(policy.get("groups", []), list) else []:
        for layer in group.get("layers", []):
            layer_to_group[int(layer)] = int(group.get("group_id", -1))

    cold_q = min(max(float(args.cold_quantile), 0.0), 1.0)
    warm_q = min(max(float(args.warm_quantile), 0.0), 1.0)
    top_dims = max(1, int(args.top_dims))

    rows: List[Dict[str, Any]] = []
    for layer in range(int(D.size(0))):
        d = D[layer]
        w = inv_std[layer]
        p = precision[layer]
        cold_thr = torch.quantile(d, torch.tensor(cold_q))
        warm_thr = torch.quantile(d, torch.tensor(warm_q))
        cold_dims = torch.argsort(d, descending=False)[:top_dims].tolist()
        warm_dims = torch.argsort(d, descending=True)[:top_dims].tolist()
        row = {
            "layer": layer,
            "regime": regimes[layer] if layer < len(regimes) else "",
            "group_id": layer_to_group.get(layer, -1),
            "reliability": float(reliability[layer].item()) if layer < int(reliability.numel()) else 1.0,
            "D_min": float(d.min().item()),
            "D_p10": float(torch.quantile(d, torch.tensor(0.10)).item()),
            "D_median": float(torch.quantile(d, torch.tensor(0.50)).item()),
            "D_mean": float(d.mean().item()),
            "D_p90": float(torch.quantile(d, torch.tensor(0.90)).item()),
            "D_max": float(d.max().item()),
            "D_condition": float((d.max() / d.min().clamp(min=eps)).item()),
            "inv_std_mean": float(w.mean().item()),
            "inv_std_max": float(w.max().item()),
            "precision_mean": float(p.mean().item()),
            "cold_threshold": float(cold_thr.item()),
            "warm_threshold": float(warm_thr.item()),
            "cold_dims": " ".join(str(x) for x in cold_dims),
            "warm_dims": " ".join(str(x) for x in warm_dims),
        }
        rows.append(row)

    group_rows: List[Dict[str, Any]] = []
    for group in policy.get("groups", []) if isinstance(policy.get("groups", []), list) else []:
        layers = [int(x) for x in group.get("layers", [])]
        if not layers:
            continue
        gd = D[layers].reshape(-1)
        gw = inv_std[layers].reshape(-1)
        group_rows.append(
            {
                "group_id": int(group.get("group_id", -1)),
                "layers": " ".join(str(x) for x in layers),
                "regime": group.get("regime", ""),
                "D_mean": float(gd.mean().item()),
                "D_median": float(gd.median().item()),
                "D_condition": float((gd.max() / gd.min().clamp(min=eps)).item()),
                "inv_std_mean": float(gw.mean().item()),
                "reliability_mean": float(reliability[layers].mean().item()),
                "mean_upstream_similarity": group.get("mean_upstream_similarity", ""),
            }
        )

    summary = {
        "label": args.label,
        "run_root": run_root,
        "structure_prior": prior_path,
        "sharing_policy": policy_path,
        "compress_report": compress_path,
        "tau_eps": eps,
        "D_shape": list(D.shape),
        "D_global": tensor_stats(D),
        "inv_std_global": tensor_stats(inv_std),
        "precision_global": tensor_stats(precision),
        "global_condition": float((D.max() / D.min().clamp(min=eps)).item()),
        "core_use_metric_whitening": compress.get("core_use_metric_whitening", ""),
        "core_use_reliability_weighting": compress.get("core_use_reliability_weighting", ""),
        "lambda_core": compress.get("lambda_core", ""),
        "last_train_core_loss": compress.get("last_train_core_loss", ""),
        "best_val_loss": compress.get("best_val_loss", ""),
        "best_val_step": compress.get("best_val_step", ""),
    }

    write_csv(
        os.path.join(output_dir, "teacher_scale_matrix_layers.csv"),
        rows,
        [
            "layer",
            "regime",
            "group_id",
            "reliability",
            "D_min",
            "D_p10",
            "D_median",
            "D_mean",
            "D_p90",
            "D_max",
            "D_condition",
            "inv_std_mean",
            "inv_std_max",
            "precision_mean",
            "cold_threshold",
            "warm_threshold",
            "cold_dims",
            "warm_dims",
        ],
    )
    write_csv(
        os.path.join(output_dir, "teacher_scale_matrix_groups.csv"),
        group_rows,
        ["group_id", "layers", "regime", "D_mean", "D_median", "D_condition", "inv_std_mean", "reliability_mean", "mean_upstream_similarity"],
    )
    with open(os.path.join(output_dir, "teacher_scale_matrix_report.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "layers": rows, "groups": group_rows}, handle, ensure_ascii=False, indent=2)

    plot_heatmap(os.path.join(output_dir, "teacher_scale_D_log_heatmap.png"), torch.log10(D.clamp(min=eps)), "", "layer")
    plot_heatmap(os.path.join(output_dir, "teacher_scale_inv_std_heatmap.png"), inv_std, "", "layer")
    plot_layer_curves(os.path.join(output_dir, "teacher_scale_layer_curves.png"), rows)

    md = [
        f"# Teacher-Induced Scale Matrix {args.label}",
        "",
        f"- run_root: `{run_root}`",
        f"- structure_prior: `{prior_path}`",
        f"- shape: `{tuple(D.shape)}`",
        f"- tau_eps: `{eps}`",
        "",
        "## Summary",
        "",
        f"- global D mean / median / min / max: `{fmt(summary['D_global']['mean'])}` / `{fmt(summary['D_global']['median'])}` / `{fmt(summary['D_global']['min'])}` / `{fmt(summary['D_global']['max'])}`",
        f"- global D condition max/min: `{fmt(summary['global_condition'], 1)}`",
        f"- global D^(-1/2) mean / median / max: `{fmt(summary['inv_std_global']['mean'])}` / `{fmt(summary['inv_std_global']['median'])}` / `{fmt(summary['inv_std_global']['max'])}`",
        f"- core_use_metric_whitening: `{summary['core_use_metric_whitening']}`",
        f"- core_use_reliability_weighting: `{summary['core_use_reliability_weighting']}`",
        f"- lambda_core: `{summary['lambda_core']}`",
        f"- last_train_core_loss: `{summary['last_train_core_loss']}`",
        "",
        "## Interpretation",
        "",
        "Small D entries are cold teacher-stable directions. The loss multiplies errors in those directions by D^(-1/2), so they receive larger gradients.",
        "Large D entries are warm high-variance directions. They are still tracked, but the Mahalanobis metric is more tolerant there.",
        "",
        "## Shared Groups",
        "",
        "| group | layers | regime | D median | D condition | mean D^-1/2 | reliability | upstream sim |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        md.append(
            f"| {row['group_id']} | {row['layers']} | {row['regime']} | {fmt(row['D_median'])} | "
            f"{fmt(row['D_condition'], 1)} | {fmt(row['inv_std_mean'])} | {fmt(row['reliability_mean'])} | {fmt(row['mean_upstream_similarity'])} |"
        )
    md.extend(
        [
            "",
            "## Files",
            "",
            f"- `teacher_scale_matrix_layers.csv`",
            f"- `teacher_scale_matrix_groups.csv`",
            f"- `teacher_scale_matrix_report.json`",
            f"- `teacher_scale_D_log_heatmap.png`",
            f"- `teacher_scale_inv_std_heatmap.png`",
            f"- `teacher_scale_layer_curves.png`",
        ]
    )
    with open(os.path.join(output_dir, "teacher_scale_matrix_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(md) + "\n")

    print(f"[Scale-D] output_dir={output_dir}", flush=True)
    print(f"[Scale-D] report={os.path.join(output_dir, 'teacher_scale_matrix_report.md')}", flush=True)
    print(
        "[Scale-D] D mean/median="
        f"{fmt(summary['D_global']['mean'])}/{fmt(summary['D_global']['median'])} "
        f"condition={fmt(summary['global_condition'], 1)} inv_std_mean={fmt(summary['inv_std_global']['mean'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
