#!/usr/bin/env python3
"""Recompute the public FAD claims from their archived source artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-8
TASKS = {
    "piqa": "piqa",
    "social_i_qa": "social_i_qa",
    "winogrande": "winogrande",
    "arc-challenge": "arc_challenge",
    "arc-easy": "arc_easy",
    "hellaswag": "hellaswag",
    "openbookqa": "openbookqa",
}


def close(actual: float, expected: float, label: str, tol: float = TOL) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_static() -> None:
    rows = {
        int(row["compression_pct"]): row
        for row in read_csv("data/processed/static_budget_results.csv")
        if row["method"] == "FAD"
    }
    if set(rows) != {15, 20, 25, 30}:
        raise AssertionError(f"unexpected FAD budgets: {sorted(rows)}")

    for budget, row in rows.items():
        task_values: dict[str, float] = {}
        raw_dir = ROOT / f"data/raw/fad_static/{budget}pct"
        for path in raw_dir.glob("phase1_5_*.json"):
            artifact = json.loads(path.read_text(encoding="utf-8"))
            dataset = str(artifact["dataset"])
            key = TASKS.get(dataset.lower())
            if key is None:
                raise AssertionError(f"unknown dataset {dataset!r} in {path}")
            if artifact.get("length_norm") != "none":
                raise AssertionError(f"current FAD artifact is not length_norm=none: {path}")
            task_values[key] = float(artifact["accuracy"])
        if set(task_values) != set(TASKS.values()):
            raise AssertionError(f"budget {budget} has task keys {sorted(task_values)}")
        for key, value in task_values.items():
            close(value, float(row[key]), f"budget {budget} {key}")
        close(sum(task_values.values()) / 7, float(row["macro_accuracy"]), f"budget {budget} macro")


def verify_adaptive() -> None:
    rows = {
        int(row["compression_pct"]): row
        for row in read_csv("data/processed/adaptive_exit_budget_results.csv")
    }
    for budget in (15, 20, 25, 30):
        row = rows[budget]
        artifact = json.loads(
            (ROOT / f"data/raw/adaptive_exit/{budget}pct/adaptive_exit_final_summary.json").read_text(
                encoding="utf-8"
            )
        )
        runtime = artifact["runtime_test"]
        controller = artifact["controller"]
        close(float(runtime["static_phase1_5_macro_accuracy"]), float(row["static_macro_accuracy"]), f"AE {budget} static")
        close(float(runtime["adaptive_exit_macro_accuracy"]), float(row["adaptive_macro_accuracy"]), f"AE {budget} accuracy")
        close(float(runtime["weighted_avg_exit_layer"]), float(row["average_exit_layer"]), f"AE {budget} avg exit", 5e-5)
        close(100 * float(runtime["weighted_estimated_layer_savings"]), float(row["layer_saving_pct"]), f"AE {budget} saving", 5e-4)
        close(float(runtime["aggregate_wall_clock_speedup_vs_full"]), float(row["speedup_vs_own_full"]), f"AE {budget} speed", 5e-5)
        close(float(controller["decision_threshold"]), float(row["controller_threshold"]), f"AE {budget} threshold")
        hist = {16: 0, 20: 0, 24: 0, 28: 0}
        for task in runtime["per_dataset"].values():
            for layer, count in task["exit_layer_hist"].items():
                hist[int(layer)] += int(count)
        if sum(hist.values()) != int(runtime["samples"]):
            raise AssertionError(f"AE {budget} exit histogram does not sum to samples")
        for layer in hist:
            expected_pct = 100 * hist[layer] / sum(hist.values())
            close(expected_pct, float(row[f"exit_layer_{layer}_pct"]), f"AE {budget} exit {layer}", 0.006)


def verify_runtime() -> None:
    artifact = json.loads((ROOT / "data/raw/adaptive_exit/runtime_decomposition_25p.json").read_text(encoding="utf-8"))
    designs = artifact["designs"]
    per_task = artifact["per_dataset"]
    geomean = math.exp(sum(math.log(float(x["ae_speedup_vs_teacher"])) for x in per_task) / len(per_task))
    aggregate = float(designs["fad_ae_25p"]["aggregate_samples_per_s"]) / float(
        designs["merged_teacher"]["aggregate_samples_per_s"]
    )
    close(geomean, float(designs["fad_ae_25p"]["task_geomean_speedup_vs_teacher"]), "25% task geomean")
    close(aggregate, float(designs["fad_ae_25p"]["aggregate_speedup_vs_teacher"]), "25% aggregate ratio")
    if abs(geomean - aggregate) < 0.05:
        raise AssertionError("runtime metrics unexpectedly collapsed into one value")

    rows = {row["design"]: row for row in read_csv("data/processed/runtime_25pct.csv")}
    close(geomean, float(rows["FAD-AE"]["task_geomean_speedup_vs_teacher"]), "runtime CSV geomean")
    close(aggregate, float(rows["FAD-AE"]["aggregate_speedup_vs_teacher"]), "runtime CSV aggregate")


def verify_pairing() -> None:
    artifact = json.loads((ROOT / "data/raw/adaptive_exit/paired_question_summary.json").read_text(encoding="utf-8"))
    counts = {key: int(value["count"]) for key, value in artifact["correctness_categories"].items()}
    if sum(counts.values()) != int(artifact["total_questions"]):
        raise AssertionError("paired correctness categories do not sum to total")
    if counts["both_correct"] + counts["ours_only_correct"] != int(artifact["ours_correct"]):
        raise AssertionError("paired student correct count is inconsistent")
    if counts["both_correct"] + counts["teacher_only_correct"] != int(artifact["teacher_correct"]):
        raise AssertionError("paired teacher correct count is inconsistent")


def verify_local_links() -> None:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    links = re.findall(r"(?:href|src)=[\"']([^\"']+)", html)
    for link in links:
        if link.startswith(("#", "http://", "https://", "mailto:", "data:")):
            continue
        target = ROOT / link.split("#", 1)[0].split("?", 1)[0]
        if not target.exists():
            raise AssertionError(f"broken local site link: {link}")


def verify_public_paths() -> None:
    # Build markers in pieces so this verifier does not flag its own source.
    forbidden = ("/" + "work/", "/" + "home/", "hank" + "9188@")
    suffixes = {".py", ".sh", ".json", ".csv", ".tsv", ".md", ".html", ".js", ".css", ".tex", ".bib", ".yml", ".yaml"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes or ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in forbidden:
            if marker in text:
                raise AssertionError(f"private/local path marker {marker!r} in {path.relative_to(ROOT)}")


def main() -> int:
    checks = [verify_static, verify_adaptive, verify_runtime, verify_pairing, verify_local_links, verify_public_paths]
    for check in checks:
        check()
        print(f"[ok] {check.__name__}")
    print("All FAD portfolio checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        raise SystemExit(1)
