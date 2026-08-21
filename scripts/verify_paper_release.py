#!/usr/bin/env python3
"""Verify the public ICLR 2027 paper artifact without model weights or a GPU."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
DATA = ROOT / "data/paper"
MANIFEST = ROOT / "experiments/iclr2027/provenance/primary_k19_manifest.json"
OBJECTIVES = ("ce", "kl", "hidden", "ambient", "isotropic", "fad")
TEXT_SUFFIXES = {".md", ".py", ".sh", ".sbatch", ".tex", ".bib", ".json", ".csv", ".tsv", ".yml", ".yaml"}
PRIVATE_MARKERS = ("/" + "work/" + "hank9188", "/" + "home/" + "hank9188")
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, got {actual}")


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    return [ROOT / line for line in output.splitlines() if line]


def verify_figures() -> int:
    tex = (PAPER / "fad.tex").read_text(encoding="utf-8")
    references = sorted(set(re.findall(r"Figure/[A-Za-z0-9_. -]+\.(?:pdf|png)", tex)))
    missing = [reference for reference in references if not (PAPER / reference).is_file()]
    if missing:
        raise AssertionError(f"missing paper figures: {missing}")
    if "Private adapter rank & $\\rho=128$" not in tex or "and $\\rho=128$" not in tex:
        raise AssertionError("paper source does not consistently state private adapter rank 128")
    return len(references)


def verify_multiseed() -> dict[str, float | int]:
    with (DATA / "all_seed_results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 18:
        raise AssertionError(f"expected 18 objective/seed rows, got {len(rows)}")
    grouped = {objective: [] for objective in OBJECTIVES}
    for row in rows:
        grouped[row["objective"]].append(row)
    for objective, objective_rows in grouped.items():
        if sorted(int(row["seed"]) for row in objective_rows) != [42, 43, 44]:
            raise AssertionError(f"unexpected seeds for {objective}")

    with (DATA / "training_seed_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = {row["objective"]: row for row in csv.DictReader(handle)}
    for objective, objective_rows in grouped.items():
        for metric in ("commonsense", "mmlu_0", "mmlu_5", "wikitext_nll", "c4_nll"):
            values = [float(row[metric]) for row in objective_rows]
            close(float(summary[objective][f"{metric}_mean"]), statistics.mean(values))
            close(float(summary[objective][f"{metric}_sd"]), statistics.stdev(values))

    fad = grouped["fad"]
    isotropic = grouped["isotropic"]
    fad_by_seed = {int(row["seed"]): row for row in fad}
    iso_by_seed = {int(row["seed"]): row for row in isotropic}
    wins_0 = sum(float(fad_by_seed[s]["mmlu_0"]) > float(iso_by_seed[s]["mmlu_0"]) for s in (42, 43, 44))
    wins_5 = sum(float(fad_by_seed[s]["mmlu_5"]) > float(iso_by_seed[s]["mmlu_5"]) for s in (42, 43, 44))
    margin_0 = float(summary["fad"]["mmlu_0_mean"]) - float(summary["isotropic"]["mmlu_0_mean"])
    margin_5 = float(summary["fad"]["mmlu_5_mean"]) - float(summary["isotropic"]["mmlu_5_mean"])
    close(margin_0, 0.02530503726914496)
    close(margin_5, 0.03491905236670939)
    if (wins_0, wins_5) != (3, 3):
        raise AssertionError(f"unexpected FAD seed wins: {(wins_0, wins_5)}")
    return {"rows": len(rows), "fad_margin_0": margin_0, "fad_margin_5": margin_5, "wins_0": wins_0, "wins_5": wins_5}


def verify_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["teacher"]["lora_rank"] != 64 or manifest["teacher"]["lora_alpha"] != 64:
        raise AssertionError("teacher LoRA provenance must be 64/64")
    if manifest["compression"]["private_adapter_rank"] != 128:
        raise AssertionError("student private adapter provenance must be rank 128")
    close(manifest["compression"]["whole_model_reduction_pct"], 20.92909092942806)


def verify_public_hygiene(files: list[Path]) -> None:
    weights = [path.relative_to(ROOT).as_posix() for path in files if path.suffix.lower() in WEIGHT_SUFFIXES]
    if weights:
        raise AssertionError(f"tracked checkpoint/weight files are forbidden: {weights}")
    leaks = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in PRIVATE_MARKERS):
            leaks.append(path.relative_to(ROOT).as_posix())
    if leaks:
        raise AssertionError(f"private cluster paths found: {leaks}")


def main() -> None:
    files = tracked_files()
    result = verify_multiseed()
    verify_manifest()
    figure_count = verify_figures()
    verify_public_hygiene(files)
    print(json.dumps({"status": "PASS", "paper_figure_references": figure_count, **result}, indent=2))


if __name__ == "__main__":
    main()
