#!/usr/bin/env python3
"""Aggregate six objectives x three seeds, subject consistency, and paired bootstrap."""
from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("FAD_FINAL_ROOT", "artifacts/final_completion"))
SOURCE = Path(os.environ.get("FAD_EXPERIMENT_ROOT", "artifacts/fad_experiments"))
PRIORITY = Path(os.environ.get("FAD_PRIORITY_ROOT", "artifacts/priority_ablation"))
TASKS = ("ARC-Challenge", "ARC-Easy", "hellaswag", "openbookqa", "piqa", "social_i_qa", "winogrande")
OBJECTIVES = ("ce", "kl", "hidden", "ambient", "isotropic", "fad")
SEEDS = (42, 43, 44)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest(directory):
    paths = sorted(Path(directory).glob("results_*.json"))
    if not paths:
        raise RuntimeError(f"missing MMLU results: {directory}")
    return load(paths[-1]), paths[-1]


def common_dir(objective, seed):
    if seed == 44:
        return SOURCE / "results/runs" / f"opt3b_{objective}"
    if objective in ("ce", "isotropic"):
        return ROOT / "results/runs" / f"k19_{objective}_s{seed}"
    if objective == "fad":
        return SOURCE / "results/runs" / f"opt3b_fad_s{seed}"
    return PRIORITY / "results/runs" / f"a3_k19_{objective}_s{seed}"


def macro(objective, seed):
    directory = common_dir(objective, seed)
    values = [float(load(directory / f"phase1_5_{task}.json")["accuracy"]) for task in TASKS]
    return statistics.mean(values)


def mmlu(objective, seed, shots):
    directory = ROOT / "results/mmlu_logged" / f"k19_{objective}_s{seed}" / f"{shots}shot"
    payload, path = latest(directory)
    return float(payload["results"]["mmlu"]["acc,none"]), payload, path


def ppl(objective, seed):
    data = load(ROOT / "results/ppl" / f"k19_{objective}_s{seed}" / "results.json")
    return float(data["wikitext2"]["nll"]), float(data["c4"]["nll"])


def leaf_subjects(payload):
    groups = set(payload.get("groups", {}))
    result = {}
    for name, metrics in payload["results"].items():
        if not name.startswith("mmlu_") or name == "mmlu" or name in groups:
            continue
        if "acc,none" in metrics:
            result[name.removeprefix("mmlu_")] = float(metrics["acc,none"])
    if len(result) != 57:
        raise RuntimeError(f"expected 57 MMLU leaf subjects, found {len(result)}")
    return result


def sample_metric(row):
    for key in ("acc", "acc,none", "exact_match"):
        if key in row and isinstance(row[key], (int, float, bool)):
            return float(row[key])
    metrics = row.get("metrics", {})
    for key in ("acc", "acc,none", "exact_match"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"cannot find correctness metric in sample keys={sorted(row)}")


def logged_samples(objective, seed, shots):
    directory = ROOT / "results/mmlu_logged" / f"k19_{objective}_s{seed}" / f"{shots}shot"
    result = {}
    paths = sorted(directory.glob("samples_*.jsonl"))
    if len(paths) >= 57:
        for path in paths:
            stem = path.stem
            subject = stem.split("mmlu_", 1)[-1]
            if "_20" in subject:
                subject = subject.rsplit("_20", 1)[0]
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    doc_id = row.get("doc_id", row.get("id"))
                    key = (subject, str(doc_id))
                    result[key] = sample_metric(row)
        return result

    # The pinned compatibility runner serializes lm-eval's logged samples inside
    # results_*.json instead of emitting one samples_*.jsonl file per task.
    payload, _ = latest(directory)
    samples = payload.get("samples", {})
    if not isinstance(samples, dict):
        raise RuntimeError(f"no logged MMLU samples found in {directory}")
    for task, rows in samples.items():
        if not task.startswith("mmlu_") or not isinstance(rows, list):
            continue
        subject = task.removeprefix("mmlu_")
        for row in rows:
            doc_id = row.get("doc_id", row.get("id"))
            key = (subject, str(doc_id))
            result[key] = sample_metric(row)
    if len(result) < 14000:
        raise RuntimeError(f"too few embedded MMLU samples in {directory}: {len(result)}")
    return result


def paired_bootstrap(a, b, iters=10000, seed=44):
    keys = sorted(set(a) & set(b))
    if len(keys) < 14000:
        raise RuntimeError(f"too few paired MMLU examples: {len(keys)}")
    av = np.asarray([a[k] for k in keys], dtype=np.float64)
    bv = np.asarray([b[k] for k in keys], dtype=np.float64)
    delta = av - bv
    rng = np.random.default_rng(seed)
    means = np.empty(iters, dtype=np.float64)
    chunk = 100
    for start in range(0, iters, chunk):
        stop = min(iters, start + chunk)
        indices = rng.integers(0, len(delta), size=(stop-start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    return {"paired_examples": len(keys), "delta": float(delta.mean()),
            "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    rows = []
    payloads = {}
    for objective in OBJECTIVES:
        for seed in SEEDS:
            m0, p0, _ = mmlu(objective, seed, 0)
            m5, p5, _ = mmlu(objective, seed, 5)
            wiki, c4 = ppl(objective, seed)
            row = {"objective": objective, "seed": seed, "commonsense": macro(objective, seed),
                   "mmlu_0": m0, "mmlu_5": m5, "wikitext_nll": wiki, "c4_nll": c4}
            rows.append(row); payloads[(objective, seed, 0)] = p0; payloads[(objective, seed, 5)] = p5
    summary = []
    for objective in OBJECTIVES:
        selected = [r for r in rows if r["objective"] == objective]
        output = {"objective": objective}
        for metric in ("commonsense", "mmlu_0", "mmlu_5", "wikitext_nll", "c4_nll"):
            values = [r[metric] for r in selected]
            output[f"{metric}_mean"] = statistics.mean(values)
            output[f"{metric}_sd"] = statistics.stdev(values)
        summary.append(output)
    strongest0 = max((r for r in summary if r["objective"] != "fad"), key=lambda r: r["mmlu_0_mean"])["objective"]
    strongest5 = max((r for r in summary if r["objective"] != "fad"), key=lambda r: r["mmlu_5_mean"])["objective"]
    fad_summary = next(r for r in summary if r["objective"] == "fad")
    base0 = next(r for r in summary if r["objective"] == strongest0)
    base5 = next(r for r in summary if r["objective"] == strongest5)
    wins0 = sum(next(r for r in rows if r["objective"] == "fad" and r["seed"] == seed)["mmlu_0"] >
                next(r for r in rows if r["objective"] == strongest0 and r["seed"] == seed)["mmlu_0"] for seed in SEEDS)
    wins5 = sum(next(r for r in rows if r["objective"] == "fad" and r["seed"] == seed)["mmlu_5"] >
                next(r for r in rows if r["objective"] == strongest5 and r["seed"] == seed)["mmlu_5"] for seed in SEEDS)

    subject_rows = []
    bootstrap = []
    for shots, strongest in ((0, strongest0), (5, strongest5)):
        per_seed_delta = {subject: [] for subject in leaf_subjects(payloads[("fad", 44, shots)])}
        for seed in SEEDS:
            fad_subject = leaf_subjects(payloads[("fad", seed, shots)])
            baseline_subject = leaf_subjects(payloads[(strongest, seed, shots)])
            for subject in per_seed_delta:
                per_seed_delta[subject].append(fad_subject[subject] - baseline_subject[subject])
            bootstrap.append({"shots": shots, "seed": seed, "baseline": strongest,
                              **paired_bootstrap(logged_samples("fad", seed, shots),
                                                 logged_samples(strongest, seed, shots), seed=44+seed+shots)})
        for subject, values in per_seed_delta.items():
            subject_rows.append({"shots": shots, "baseline": strongest, "subject": subject,
                                 "seed_mean_delta": statistics.mean(values),
                                 "seed_median_delta": statistics.median(values),
                                 "positive_seeds": sum(value > 0 for value in values)})

    result = {
        "status": "complete", "strongest_non_fad_0": strongest0, "strongest_non_fad_5": strongest5,
        "fad_margin_0": fad_summary["mmlu_0_mean"] - base0["mmlu_0_mean"],
        "fad_margin_5": fad_summary["mmlu_5_mean"] - base5["mmlu_5_mean"],
        "fad_seed_wins_0": wins0, "fad_seed_wins_5": wins5,
        "subject_positive_fraction_0": statistics.mean(r["seed_mean_delta"] > 0 for r in subject_rows if r["shots"] == 0),
        "subject_positive_fraction_5": statistics.mean(r["seed_mean_delta"] > 0 for r in subject_rows if r["shots"] == 5),
        "summary": summary, "bootstrap": bootstrap,
    }
    out = ROOT / "results/analysis"; out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "all_seed_results.csv", rows)
    write_csv(out / "training_seed_summary.csv", summary)
    write_csv(out / "mmlu_subject_deltas.csv", subject_rows)
    (out / "paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")
    (out / "P1_ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    moderate_pass = (fad_summary["mmlu_0_mean"] > base0["mmlu_0_mean"] and
                     fad_summary["mmlu_5_mean"] > base5["mmlu_5_mean"] and wins0 >= 2 and wins5 >= 2)
    result["classification"] = "PASS" if moderate_pass else "FAIL_OR_STORY_REVISION"
    (out / "P1_ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if moderate_pass:
        (out / "P1_GATE_PASS").write_text("PASS\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("summary", "bootstrap")}, indent=2))
    if not moderate_pass:
        raise SystemExit("P1 moderate-pass gate failed; downstream P2 jobs must not start")


if __name__ == "__main__":
    main()
