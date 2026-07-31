#!/usr/bin/env python3
"""End-to-end HTTP smoke test for mock or live GPU demo mode."""

from __future__ import annotations

import argparse
import json
import urllib.request


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health = get_json(f"{base}/api/health")
    assert health["ready"] is True, health
    catalog = get_json(f"{base}/api/questions")
    assert catalog["ok"] is True
    question_count = int(catalog["question_count"])
    assert question_count > 0
    assert int(catalog["paper_benchmark"]["questions"]) == 19149
    assert len(catalog["categories"]) >= 1
    assert len(catalog["datasets"]) >= 1
    sample = get_json(f"{base}/api/sample?count=5&dataset=all")
    assert sample["ok"] is True
    assert sample["sampling"] == "uniform_without_replacement"
    assert sample["pool_size"] == question_count
    assert len(sample["questions"]) == 5
    assert len({question["id"] for question in sample["questions"]}) == 5
    assert all("gold_answer" not in question for question in sample["questions"])
    assert all("historical" not in question for question in sample["questions"])
    question = sample["questions"][0]
    result = post_json(
        f"{base}/api/infer",
        {
            "question_id": question["id"],
            "models": ["student", "teacher"],
            "benchmark": True,
        },
    )
    assert result["ok"] is True
    assert set(result["results"]) == {"student", "teacher"}
    assert float(result["results"]["student"]["latency_ms"]) > 0
    assert int(result["results"]["teacher"]["exit_layer"]) == 28
    assert result["timing_protocol"] == "steady_state_interleaved_median"
    assert result["results"]["student"]["timing"]["repeats"] == 5
    assert result["results"]["teacher"]["timing"]["statistic"] == "median"
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": health["mode"],
                "gpu": health["gpu"],
                "demo_questions": question_count,
                "paper_benchmark_questions": int(catalog["paper_benchmark"]["questions"]),
                "tested_question": question["id"],
                "student": result["results"]["student"],
                "teacher": result["results"]["teacher"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
