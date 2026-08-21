#!/usr/bin/env python3
"""Validate completed lm-eval JSON artifacts and report embedded sample layout."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("FAD_FINAL_ROOT", "artifacts/final_completion"))


def main() -> None:
    files = sorted((ROOT / "results/mmlu_logged").glob("k19_*_s*/?shot/results_*.json"))
    rows = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = payload.get("samples")
        sample_tasks = len(samples) if isinstance(samples, dict) else 0
        sample_rows = (
            sum(len(value) for value in samples.values() if isinstance(value, list))
            if isinstance(samples, dict)
            else 0
        )
        leaf_subjects = sum(
            name.startswith("mmlu_")
            and name != "mmlu"
            and isinstance(metrics, dict)
            and "acc,none" in metrics
            for name, metrics in payload.get("results", {}).items()
        )
        rows.append(
            {
                "path": str(path),
                "has_mmlu_aggregate": "mmlu" in payload.get("results", {}),
                "leaf_subjects": leaf_subjects,
                "sample_tasks": sample_tasks,
                "sample_rows": sample_rows,
                "top_level_keys": sorted(payload),
            }
        )

    report = {
        "result_files": len(files),
        "valid_aggregate_files": sum(
            row["has_mmlu_aggregate"] and row["leaf_subjects"] >= 57 for row in rows
        ),
        "files_with_embedded_samples": sum(row["sample_rows"] > 0 for row in rows),
        "total_embedded_sample_rows": sum(row["sample_rows"] for row in rows),
        "rows": rows,
    }
    out = ROOT / "results/analysis/MMLU_ARTIFACT_AUDIT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if report["valid_aggregate_files"] != 36:
        raise SystemExit("not all 36 MMLU result files contain complete aggregate results")


if __name__ == "__main__":
    main()
