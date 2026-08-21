#!/usr/bin/env python3
"""Held-out retained/off-subspace error and diagonal/full-covariance audit."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE = Path(os.environ.get(
    "FAD_PIPELINE",
    REPO_ROOT / "reproduction/fad/core/newthesis_pipeline_final_llama.py",
))
ORIGINAL_PIQA = Path(os.environ.get("FAD_PIQA_JSON", "external/piqa/test.json"))
CLEAN_PIQA = Path(os.environ.get(
    "FAD_CLEAN_PIQA_JSON",
    "external/piqa/heldout_clean.json",
))
OVERLAP_AUDIT = REPO_ROOT / "experiments/iclr2027/provenance/data_overlap_audit.json"


def load_module():
    spec = importlib.util.spec_from_file_location("fad_final_subspace_pipeline", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def records(dataset, count):
    if dataset == "piqa":
        if CLEAN_PIQA.is_file():
            rows = json.loads(CLEAN_PIQA.read_text(encoding="utf-8"))
        else:
            rows = json.loads(ORIGINAL_PIQA.read_text(encoding="utf-8"))
            audit = json.loads(OVERLAP_AUDIT.read_text(encoding="utf-8"))
            excluded = set(int(index) for index in audit["piqa_overlap_indices"])
            rows = [row for index, row in enumerate(rows) if index not in excluded]
        random.Random(1944).shuffle(rows)
        return [("record", row) for row in rows[:count]]
    if dataset == "wikitext2":
        stream = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        stream = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    output = []
    for row in stream:
        text = str(row.get("text", "")).strip()
        if len(text.split()) >= 48:
            output.append(("text", text))
        if len(output) >= count:
            break
    return output


def ranks(values):
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def correlation(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--dataset", choices=("piqa", "wikitext2", "c4"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    torch.manual_seed(44)
    pipe = load_module(); device = torch.device("cuda:0"); dtype = torch.bfloat16
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    student, _ = pipe._build_shared_model_for_eval(
        base_model=bundle["base_model"], atlas_payload=bundle["atlas"],
        shared_payload=bundle["shared_student"], quant_bank_int4=None,
        use_quant_bank_int4=False, device=device, dtype=dtype, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype,
                                                    trust_remote_code=True).to(device).eval()
    tokenizer = pipe.load_tokenizer(args.teacher, trust_remote_code=True)
    geometry = Path(args.geometry)
    prior = torch.load(geometry / "final_structure_prior.pt", map_location="cpu", weights_only=False)
    policy = json.loads((geometry / "sharing_policy.json").read_text(encoding="utf-8"))
    regimes = [str(row.get("regime", "llama_late")) for row in policy["layers"]]
    global_basis = prior["basis"].float() if "basis" in prior else bundle["atlas"]["basis"].float()
    regime_basis = prior.get("regime_basis", {})
    diagonal = prior["layer_metric_diag"].float().clamp_min(1e-5)
    layers = len(regimes)
    parallel, perpendicular, leakage = [], [], []
    teacher_codes = [[] for _ in range(layers)]
    residual_codes = [[] for _ in range(layers)]

    def capture(model, encoded, token_index):
        outputs = {}; handles = []
        for lid, layer in enumerate(pipe._resolve_layers(model)):
            def hook(_m, _i, out, lid=lid):
                outputs[lid] = (out[0] if isinstance(out, tuple) else out).detach()
            handles.append(layer.mlp.register_forward_hook(hook))
        try:
            model(**encoded, output_hidden_states=False, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack([outputs[i][0, token_index].float().cpu() for i in range(layers)])

    with torch.inference_mode():
        for kind, item in records(args.dataset, args.samples):
            prompt = pipe._build_prompt(item, "piqa") if kind == "record" else item
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            valid = int(encoded["attention_mask"][0].sum().item())
            if valid < 3:
                continue
            token_index = max(0, valid - 2)
            sv, tv = capture(student, encoded, token_index), capture(teacher, encoded, token_index)
            p_row, q_row, r_row = [], [], []
            for lid in range(layers):
                basis = regime_basis.get(regimes[lid], global_basis).float()
                residual = sv[lid] - tv[lid]
                z_error = residual @ basis
                z_teacher = tv[lid] @ basis
                e_parallel = float(z_error.square().sum())
                e_total = float(residual.square().sum())
                e_perp = max(0.0, e_total - e_parallel)
                p_row.append(e_parallel); q_row.append(e_perp)
                r_row.append(e_perp / max(1e-12, e_parallel + e_perp))
                teacher_codes[lid].append(z_teacher.numpy())
                residual_codes[lid].append(z_error.numpy())
            parallel.append(p_row); perpendicular.append(q_row); leakage.append(r_row)

    parallel = np.asarray(parallel); perpendicular = np.asarray(perpendicular); leakage = np.asarray(leakage)
    if len(parallel) < max(32, args.samples // 2):
        raise RuntimeError(f"too few usable samples: {len(parallel)}")
    layer_rows = []
    for lid in range(layers):
        x = np.asarray(teacher_codes[lid], dtype=np.float64)
        e = np.asarray(residual_codes[lid], dtype=np.float64)
        x = x - x.mean(0, keepdims=True)
        covariance = (x.T @ x) / max(1, len(x) - 1)
        diag = np.clip(np.diag(covariance), 1e-8, None)
        denom = np.sqrt(np.outer(diag, diag))
        corr = covariance / np.clip(denom, 1e-12, None)
        off = np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])
        alpha = 0.1
        regularized = (1-alpha) * covariance + alpha * np.diag(diag)
        regularized += np.eye(len(diag)) * max(1e-8, float(diag.mean()) * 1e-5)
        eigenvalues = np.linalg.eigvalsh(regularized)
        precision_error = np.linalg.solve(regularized, e.T).T
        d_full = np.sum(e * precision_error, axis=1)
        d_diag = np.sum(e * e / diag[None, :], axis=1)
        layer_rows.append({
            "layer": lid + 1, "regime": regimes[lid],
            "E_parallel_mean": float(parallel[:, lid].mean()),
            "E_perp_mean": float(perpendicular[:, lid].mean()),
            "R_perp_mean": float(leakage[:, lid].mean()),
            "R_perp_median": float(np.median(leakage[:, lid])),
            "mean_abs_offdiag_correlation": float(off.mean()),
            "p95_abs_offdiag_correlation": float(np.quantile(off, .95)),
            "regularized_spectral_min": float(eigenvalues.min()),
            "regularized_spectral_max": float(eigenvalues.max()),
            "regularized_condition_number": float(eigenvalues.max()/max(1e-12, eigenvalues.min())),
            "diag_full_pearson": correlation(d_diag, d_full),
            "diag_full_spearman": correlation(ranks(d_diag), ranks(d_full)),
        })
    summary = {
        "method": args.method, "dataset": args.dataset, "samples": int(len(parallel)),
        "E_parallel_mean": float(parallel.mean()),
        "E_perp_mean": float(perpendicular.mean()),
        "R_perp_mean": float(leakage.mean()),
        "R_perp_median": float(np.median(leakage)),
        "mean_abs_offdiag_correlation": float(np.mean([r["mean_abs_offdiag_correlation"] for r in layer_rows])),
        "p95_abs_offdiag_correlation_layer_mean": float(np.mean([r["p95_abs_offdiag_correlation"] for r in layer_rows])),
        "diag_full_pearson_layer_mean": float(np.nanmean([r["diag_full_pearson"] for r in layer_rows])),
        "diag_full_spearman_layer_mean": float(np.nanmean([r["diag_full_spearman"] for r in layer_rows])),
        "covariance_regularization": "0.1 shrinkage to diagonal + 1e-5 mean-diagonal ridge",
        "per_layer": layer_rows,
    }
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "error_decomposition.npz", E_parallel=parallel,
                        E_perp=perpendicular, R_perp=leakage)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_layer"}, indent=2))


if __name__ == "__main__":
    main()
