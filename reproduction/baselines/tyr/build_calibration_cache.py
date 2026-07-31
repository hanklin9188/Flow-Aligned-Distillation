#!/usr/bin/env python3
"""Build a Tyr-compatible calibration tensor cache from a local JSON dataset."""
from __future__ import annotations

import argparse
import json
import os
import random
from itertools import chain
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoTokenizer, LlamaTokenizer


def format_instruction_example(ex: Dict[str, Any]) -> str:
    instruction = str(ex.get("instruction", "") or "").strip()
    input_text = str(ex.get("input", "") or "").strip()
    output = str(ex.get("output", "") or "").strip()
    answer = str(ex.get("answer", "") or "").strip()
    response = output or answer
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{response}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def load_texts(path: str, max_records: int, seed: int) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        for key in ("train", "data", "examples"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list or dict containing a list: {path}")

    records = list(raw)
    random.Random(seed).shuffle(records)
    if max_records > 0:
        records = records[:max_records]

    texts: List[str] = []
    for ex in records:
        if not isinstance(ex, dict):
            texts.append(str(ex))
        elif "text" in ex:
            texts.append(str(ex["text"]))
        else:
            texts.append(format_instruction_example(ex))
    return texts


def chunked_tokens(token_lists: Iterable[List[int]], sequence_length: int, max_tokens: int) -> List[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    buffer: List[int] = []
    tokens_kept = 0
    sep = []
    for ids in token_lists:
        if sep:
            buffer.extend(sep)
        buffer.extend(ids)
        while len(buffer) >= sequence_length and tokens_kept + sequence_length <= max_tokens:
            piece = buffer[:sequence_length]
            buffer = buffer[sequence_length:]
            chunks.append(torch.tensor([piece], dtype=torch.long))
            tokens_kept += sequence_length
        if tokens_kept + sequence_length > max_tokens:
            break
        sep = []
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--tokenizer_name_or_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_tokens", type=int, default=4_194_304)
    parser.add_argument("--sequence_length", type=int, default=4096)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_fast", action="store_true")
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=args.use_fast)
    except Exception:
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=args.use_fast)

    texts = load_texts(args.data_path, args.max_records, args.seed)
    token_lists = (
        tokenizer(text + "\n\n", add_special_tokens=False).input_ids
        for text in texts
    )
    samples = chunked_tokens(token_lists, int(args.sequence_length), int(args.max_tokens))
    if not samples:
        raise ValueError("No calibration samples were produced.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    torch.save(samples, args.output_path)
    print(
        json.dumps(
            {
                "output_path": args.output_path,
                "num_samples": len(samples),
                "sequence_length": int(args.sequence_length),
                "tokens": len(samples) * int(args.sequence_length),
                "data_path": args.data_path,
                "tokenizer": args.tokenizer_name_or_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
