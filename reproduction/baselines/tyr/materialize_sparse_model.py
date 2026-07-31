#!/usr/bin/env python3
"""Apply a Tyr sparse configuration to a base model and save a HF checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_utils import load_sparse_weights  # noqa: E402


def dtype_from_name(name: str) -> Any:
    normalized = str(name).strip().lower()
    if normalized in {"auto", ""}:
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--tokenizer_name_or_path", default="")
    parser.add_argument("--sparse_weights_path", required=True)
    parser.add_argument("--sparse_config_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--safe_serialization", action="store_true")
    args = parser.parse_args()

    dtype = dtype_from_name(args.torch_dtype)
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    load_sparse_weights(model, args.sparse_weights_path, args.sparse_config_name)

    tokenizer_source = args.tokenizer_name_or_path or args.model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    except Exception:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_source, use_fast=False)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=bool(args.safe_serialization))
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "method": "Tyr-the-Pruner",
        "model_name_or_path": args.model_name_or_path,
        "sparse_weights_path": args.sparse_weights_path,
        "sparse_config_name": args.sparse_config_name,
        "output_dir": args.output_dir,
        "torch_dtype": args.torch_dtype,
    }
    with open(os.path.join(args.output_dir, "tyr_materialize_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
