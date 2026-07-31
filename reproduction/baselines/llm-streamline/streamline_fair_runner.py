#!/usr/bin/env python3
"""Fair-comparison helpers for LLM-Streamline.

This runner keeps the Streamline compression idea intact, but makes the
training/evaluation setup explicit:

* task-fair mode can train on a local instruction JSON such as
  new_idea/data/datasets/commonsense_170k.json.
* paper-style mode can train on the original SlimPajama source.
* summaries are written as CSV/JSON so the results can be dropped into a table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LLM_Streamline.get_cosine import get_cosine_similarity  # noqa: E402
from LLM_Streamline.scheduler import get_cosine_schedule_with_warmup  # noqa: E402
from LLM_Streamline.train_lightweightnetwork import process_datasets  # noqa: E402


TASKS_USER = [
    "ARC-Challenge",
    "ARC-Easy",
    "hellaswag",
    "openbookqa",
    "piqa",
    "social_i_qa",
    "winogrande",
]

TASKS_PAPER = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "hellaswag",
    "openbookqa",
    "rte",
    "winogrande",
]


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return str(value)


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def dtype_from_name(name: str) -> torch.dtype:
    s = str(name).strip().lower()
    if s in {"auto", ""}:
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def model_family_label(config: AutoConfig, model_name: str) -> str:
    model_type = str(getattr(config, "model_type", "")).lower()
    if "llama" in model_type:
        return "llama"
    if "opt" in model_type:
        return "opt"
    lowered = str(model_name).lower()
    if "llama" in lowered:
        return "llama"
    if "opt" in lowered:
        return "opt"
    raise ValueError(
        f"Unsupported model_type={model_type!r}. This wrapper currently supports Llama/OPT Streamline copying."
    )


def prune_model(model, pruned_model, best_layer: int, layer_gap: int, model_family: str, num_layers: int):
    """Copy original weights into a layer-removed Streamline model.

    This is the same weight-copying idea as LLM_Streamline.train_llmloss.prune_model,
    kept local to avoid importing that module's optional deepspeed dependency.
    layer_gap is remove_layers + 1; the removed indices are
    best_layer+1 ... best_layer+layer_gap-1.
    """
    pruned_layers = set(range(best_layer + 1, best_layer + layer_gap))
    pruned_weight = pruned_model.state_dict()
    weight = model.state_dict()

    if model_family == "llama":
        for key in ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"):
            if key in pruned_weight and key in weight:
                pruned_weight[key] = weight[key]

        j = 0
        suffixes = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]
        for i in range(num_layers):
            if i in pruned_layers:
                continue
            for suffix in suffixes:
                dst = f"model.layers.{j}.{suffix}"
                src = f"model.layers.{i}.{suffix}"
                if dst in pruned_weight and src in weight:
                    pruned_weight[dst] = weight[src]
            j += 1

    elif model_family == "opt":
        for key in (
            "model.decoder.embed_tokens.weight",
            "model.decoder.embed_positions.weight",
            "model.decoder.final_layer_norm.weight",
            "model.decoder.final_layer_norm.bias",
            "lm_head.weight",
        ):
            if key in pruned_weight and key in weight:
                pruned_weight[key] = weight[key]

        j = 0
        suffixes = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.out_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.bias",
            "self_attn_layer_norm.weight",
            "self_attn_layer_norm.bias",
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
            "final_layer_norm.weight",
            "final_layer_norm.bias",
        ]
        for i in range(num_layers):
            if i in pruned_layers:
                continue
            for suffix in suffixes:
                dst = f"model.decoder.layers.{j}.{suffix}"
                src = f"model.decoder.layers.{i}.{suffix}"
                if dst in pruned_weight and src in weight:
                    pruned_weight[dst] = weight[src]
            j += 1

    else:
        raise ValueError(f"Unsupported model_family={model_family!r}")

    pruned_model.load_state_dict(pruned_weight)
    return pruned_model


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


def load_local_text_dataset(path: str, max_records: int, seed: int) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for key in ("train", "data", "examples"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list or dict containing a list: {path}")
    records = list(raw)
    rng = random.Random(seed)
    rng.shuffle(records)
    if max_records > 0:
        records = records[:max_records]
    texts = []
    for ex in records:
        if not isinstance(ex, dict):
            texts.append(str(ex))
        elif "text" in ex:
            texts.append(str(ex["text"]))
        else:
            texts.append(format_instruction_example(ex))
    return Dataset.from_dict({"text": texts})


def tokenize_and_group(
    dataset: Dataset,
    tokenizer,
    block_size: int,
    num_proc: int,
    desc_prefix: str,
) -> Dataset:
    column_names = dataset.column_names
    text_column = "text" if "text" in column_names else column_names[0]

    def tokenize_fn(examples):
        return tokenizer(examples[text_column])

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=max(1, int(num_proc)),
        remove_columns=column_names,
        desc=f"{desc_prefix}: tokenize",
    )

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped = tokenized.map(
        group_texts,
        batched=True,
        num_proc=max(1, int(num_proc)),
        desc=f"{desc_prefix}: group block_size={block_size}",
    )
    if len(grouped) <= 1:
        raise ValueError(
            f"Only {len(grouped)} LM chunks after grouping. Lower --block_size or increase data."
        )
    return grouped


def prepare_lm_datasets(args: argparse.Namespace, tokenizer) -> Tuple[Dataset, Dataset]:
    tokenizer.pad_token = tokenizer.eos_token
    source = str(args.data_source).strip().lower()
    if source == "local_json":
        raw = load_local_text_dataset(args.data_path, args.max_records, args.seed)
        split = raw.train_test_split(
            test_size=float(args.val_ratio),
            seed=int(args.seed),
            shuffle=True,
        )
        train_ds = tokenize_and_group(
            split["train"],
            tokenizer,
            int(args.block_size),
            int(args.num_proc),
            "local train",
        )
        eval_ds = tokenize_and_group(
            split["test"],
            tokenizer,
            int(args.block_size),
            int(args.num_proc),
            "local eval",
        )
        return train_ds, eval_ds

    if source == "slimpajama":
        raw = load_dataset("DKYoon/SlimPajama-6B")["train"]
        if bool(args.slimpajama_original_mix):
            return process_datasets(raw, int(args.train_num_data), tokenizer)
        if int(args.max_records) > 0:
            raw = raw.shuffle(seed=int(args.seed)).select(range(int(args.max_records)))
        split = raw.train_test_split(
            test_size=float(args.val_ratio),
            seed=int(args.seed),
            shuffle=True,
        )
        return (
            tokenize_and_group(split["train"], tokenizer, int(args.block_size), int(args.num_proc), "slimpajama train"),
            tokenize_and_group(split["test"], tokenizer, int(args.block_size), int(args.num_proc), "slimpajama eval"),
        )

    raise ValueError("--data_source must be local_json or slimpajama")


def valid_model(model, eval_dataloader, accelerator: Accelerator) -> torch.Tensor:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            loss = outputs.loss.detach()
            losses.append(accelerator.gather_for_metrics(loss.reshape(1)).cpu())
            del outputs
    if not losses:
        return torch.tensor(float("inf"))
    return torch.cat(losses).mean()


def train_llmloss(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))
    ensure_dir(args.output_dir)
    metadata_path = os.path.join(args.output_dir, "streamline_metadata.json")

    accelerator = Accelerator(
        mixed_precision=str(args.mixed_precision),
        gradient_accumulation_steps=int(args.gradient_accumulation_step),
    )
    dtype = dtype_from_name(args.torch_dtype)
    if accelerator.is_main_process:
        print(f"[StreamlineFair] loading model={args.model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=bool(args.trust_remote_code))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_ds = prepare_lm_datasets(args, tokenizer)
    original_num_layers = int(config.num_hidden_layers)
    hidden_gap = int(args.layer_intervals) + 1
    if hidden_gap >= original_num_layers:
        raise ValueError(
            f"layer_intervals={args.layer_intervals} is too large for num_layers={original_num_layers}"
        )
    cosine_n = min(int(args.cosine_num_data), len(train_ds) - 1)
    if cosine_n <= 0:
        raise ValueError("Need at least two training chunks for cosine layer selection.")

    if accelerator.is_main_process:
        print(
            f"[StreamlineFair] train_chunks={len(train_ds)} eval_chunks={len(eval_ds)} "
            f"remove_layers={args.layer_intervals} hidden_gap={hidden_gap}",
            flush=True,
        )
    best_layer = get_cosine_similarity(
        model,
        train_ds,
        cosine_n,
        args.cosine_device,
        hidden_gap,
        original_num_layers,
    )

    pruned_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=bool(args.trust_remote_code))
    pruned_config.num_hidden_layers = original_num_layers - int(args.layer_intervals)
    pruned_model = AutoModelForCausalLM.from_config(pruned_config, trust_remote_code=bool(args.trust_remote_code))
    family = model_family_label(config, args.model_name)
    pruned_model = prune_model(
        model,
        pruned_model,
        best_layer,
        hidden_gap,
        family,
        original_num_layers,
    )
    pruned_model.to(dtype=dtype)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for name, param in pruned_model.named_parameters():
        param.requires_grad = f"layers.{best_layer}" in name
    trainable = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pruned_model.parameters())

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=int(args.batch_size),
    )
    eval_loader = DataLoader(
        eval_ds,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=int(args.batch_size),
    )
    optimizer = torch.optim.AdamW(
        [p for p in pruned_model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.wd),
        betas=(0.9, 0.95),
    )
    num_steps = max(1, len(train_loader) * int(args.epoches))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=max(1, int(num_steps * float(args.warmup_ratio))),
        num_training_steps=max(1, int(num_steps * float(args.cosine_steps_ratio))),
        max_learning_rate=float(args.lr),
        min_learning_rate=float(args.min_lr),
    )

    train_loader, eval_loader, pruned_model, optimizer, scheduler = accelerator.prepare(
        train_loader,
        eval_loader,
        pruned_model,
        optimizer,
        scheduler,
    )

    best_loss = valid_model(pruned_model, eval_loader, accelerator)
    if accelerator.is_main_process:
        print(f"[StreamlineFair] before training valid_loss={float(best_loss):.6f}", flush=True)

    best_epoch = -1
    for epoch in range(int(args.epoches)):
        pruned_model.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(pruned_model):
                outputs = pruned_model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pruned_model.parameters(), float(args.grad_clip))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.is_main_process and int(args.log_every) > 0 and step % int(args.log_every) == 0:
                print(
                    f"[StreamlineFair] epoch={epoch} step={step}/{len(train_loader)} loss={float(loss.detach()):.6f}",
                    flush=True,
                )
            del outputs, loss

        valid_loss = valid_model(pruned_model, eval_loader, accelerator)
        if accelerator.is_main_process:
            print(f"[StreamlineFair] epoch={epoch} valid_loss={float(valid_loss):.6f}", flush=True)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(pruned_model)
                unwrapped.save_pretrained(args.output_dir, safe_serialization=True)
                tokenizer.save_pretrained(args.output_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if best_epoch < 0:
            unwrapped = accelerator.unwrap_model(pruned_model)
            unwrapped.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
        metadata = {
            "method": "LLM-Streamline-llmloss",
            "model_name": args.model_name,
            "data_source": args.data_source,
            "data_path": args.data_path,
            "train_chunks": len(train_ds),
            "eval_chunks": len(eval_ds),
            "original_num_layers": original_num_layers,
            "pruned_num_layers": original_num_layers - int(args.layer_intervals),
            "remove_layers": int(args.layer_intervals),
            "hidden_gap": hidden_gap,
            "best_layer": int(best_layer),
            "pruned_layer_indices": list(range(int(best_layer) + 1, int(best_layer) + hidden_gap)),
            "trainable_params": int(trainable),
            "total_params": int(total),
            "best_eval_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "args": {
                k: json_safe(v)
                for k, v in vars(args).items()
                if k != "func"
            },
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[StreamlineFair] saved model -> {args.output_dir}", flush=True)
        print(f"[StreamlineFair] metadata -> {metadata_path}", flush=True)


def model_stats(args: argparse.Namespace) -> None:
    dtype = dtype_from_name(args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )
    config = getattr(model, "config", None)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layers = getattr(getattr(model, "model", None), "layers", None)
    num_layers = len(layers) if layers is not None else getattr(config, "num_hidden_layers", None)
    disk_bytes = 0
    if os.path.isdir(args.model_name):
        for root, _, files in os.walk(args.model_name):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    disk_bytes += os.path.getsize(fp)
                except OSError:
                    pass
    stats = {
        "model_name": args.model_name,
        "model_type": getattr(config, "model_type", None),
        "num_layers": num_layers,
        "total_params": int(total),
        "trainable_params": int(trainable),
        "disk_bytes": int(disk_bytes),
        "disk_gib": disk_bytes / (1024**3) if disk_bytes else None,
    }
    if args.output_json:
        ensure_dir(os.path.dirname(os.path.abspath(args.output_json)))
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


def read_eval_csv(path: str, label: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            acc_raw = row.get("accuracy", "")
            try:
                acc = float(acc_raw)
            except Exception:
                continue
            rows.append(
                {
                    "label": label,
                    "variant": row.get("variant", ""),
                    "dataset": row.get("dataset", ""),
                    "accuracy": acc,
                    "throughput_toks_per_s": row.get("throughput_toks_per_s", ""),
                    "avg_latency_ms": row.get("avg_latency_ms", ""),
                    "source_csv": path,
                }
            )
    return rows


def summarize(args: argparse.Namespace) -> None:
    all_rows: List[Dict[str, Any]] = []
    for item in args.csv:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).stem
        all_rows.extend(read_eval_csv(path, label))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        variant = str(row.get("variant", "") or "").strip()
        key = row["label"] if not variant else f"{row['label']}:{variant}"
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for label, rows in grouped.items():
        mean_acc = sum(r["accuracy"] for r in rows) / max(1, len(rows))
        summary_rows.append(
            {
                "label": label,
                "mean_accuracy": mean_acc,
                "num_tasks": len(rows),
                "tasks": " ".join(r["dataset"] for r in rows),
            }
        )
    summary_rows.sort(key=lambda r: r["mean_accuracy"], reverse=True)

    ensure_dir(os.path.dirname(os.path.abspath(args.output_csv)))
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        columns = ["label", "mean_accuracy", "num_tasks", "tasks"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary_rows, "per_task": all_rows}, f, indent=2)

    for row in summary_rows:
        print(f"{row['label']}: mean_accuracy={row['mean_accuracy']:.4f} tasks={row['num_tasks']}")
    print(f"[StreamlineFair] summary -> {args.output_csv}")


def _resolve_lm_eval_json(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)
    candidates = sorted(
        [x for x in p.rglob("*.json") if x.is_file()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("results"), dict):
                return str(candidate)
        except Exception:
            continue
    raise FileNotFoundError(f"No lm-eval results JSON found under: {path}")


def _metric_from_lm_eval_task(task_result: Dict[str, Any]) -> Tuple[str, float] | Tuple[str, None]:
    priority = [
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
        "acc_norm",
        "acc",
        "exact_match",
    ]
    for key in priority:
        if key in task_result:
            try:
                return key, float(task_result[key])
            except Exception:
                pass
    for key, value in task_result.items():
        if key.endswith(",none") or key in {"acc", "acc_norm", "exact_match"}:
            try:
                return key, float(value)
            except Exception:
                continue
    return "", None


def summarize_lm_eval(args: argparse.Namespace) -> None:
    per_task: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for item in args.json:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).stem
        resolved = _resolve_lm_eval_json(path)
        with open(resolved, "r", encoding="utf-8") as f:
            obj = json.load(f)
        results = obj.get("results", {})
        vals = []
        for task, task_result in sorted(results.items()):
            if not isinstance(task_result, dict):
                continue
            metric, value = _metric_from_lm_eval_task(task_result)
            if value is None:
                continue
            vals.append(value)
            per_task.append(
                {
                    "label": label,
                    "task": task,
                    "metric": metric,
                    "score": value,
                    "source_json": resolved,
                }
            )
        mean_score = sum(vals) / max(1, len(vals))
        summary.append(
            {
                "label": label,
                "mean_score": mean_score,
                "num_tasks": len(vals),
                "source_json": resolved,
            }
        )

    summary.sort(key=lambda r: r["mean_score"], reverse=True)
    ensure_dir(os.path.dirname(os.path.abspath(args.output_csv)))
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "mean_score", "num_tasks", "source_json"])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_task": per_task}, f, indent=2)
    for row in summary:
        print(f"{row['label']}: mean_score={row['mean_score']:.4f} tasks={row['num_tasks']}")
    print(f"[StreamlineFair] lm-eval summary -> {args.output_csv}")


def add_common_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_source", choices=["local_json", "slimpajama"], default="local_json")
    p.add_argument("--data_path", type=str, default="")
    p.add_argument("--max_records", type=int, default=100000)
    p.add_argument("--train_num_data", type=int, default=100000)
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--num_proc", type=int, default=1)
    p.add_argument("--slimpajama_original_mix", type=str2bool, default=True)
    p.add_argument("--layer_intervals", type=int, default=8, help="Number of Transformer layers to remove.")
    p.add_argument("--cosine_num_data", type=int, default=50)
    p.add_argument("--cosine_device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_step", type=int, default=16)
    p.add_argument("--epoches", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--cosine_steps_ratio", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch_dtype", type=str, default="auto")
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--trust_remote_code", type=str2bool, default=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fair LLM-Streamline comparison pieces.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train-llmloss", help="Train Streamline replacement layer with LM CE loss.")
    add_common_train_args(p_train)
    p_train.set_defaults(func=train_llmloss)

    p_stats = sub.add_parser("model-stats", help="Write parameter/disk statistics for a HF checkpoint.")
    p_stats.add_argument("--model_name", type=str, required=True)
    p_stats.add_argument("--output_json", type=str, default="")
    p_stats.add_argument("--torch_dtype", type=str, default="auto")
    p_stats.add_argument("--trust_remote_code", type=str2bool, default=True)
    p_stats.set_defaults(func=model_stats)

    p_sum = sub.add_parser("summarize", help="Summarize new_idea eval CSVs.")
    p_sum.add_argument("--csv", action="append", required=True, help="label=/path/to/results.csv")
    p_sum.add_argument("--output_csv", type=str, required=True)
    p_sum.add_argument("--output_json", type=str, default="")
    p_sum.set_defaults(func=summarize)

    p_lm = sub.add_parser("summarize-lm-eval", help="Summarize lm-eval result JSON files.")
    p_lm.add_argument("--json", action="append", required=True, help="label=/path/to/lm_eval_output_or_json")
    p_lm.add_argument("--output_csv", type=str, required=True)
    p_lm.add_argument("--output_json", type=str, default="")
    p_lm.set_defaults(func=summarize_lm_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
