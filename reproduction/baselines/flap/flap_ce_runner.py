#!/usr/bin/env python3
"""FLAP-style MLP pruning plus CE-only recovery.

This runner is built for the task-fair comparison used in the local
LLM-Streamline experiments:

1. Start from the same merged LoRA teacher.
2. Prune FFN intermediate neurons with a FLAP/WIFV-style metric.
3. Save a real smaller Hugging Face checkpoint.
4. Recover with CE-only language-model training using LoRA updates, then merge.

The pruning is intentionally MLP-only. It keeps the checkpoint reloadable by
standard Hugging Face Llama classes and avoids GQA attention-shape pitfalls.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


TASKS_USER = [
    "ARC-Challenge",
    "ARC-Easy",
    "hellaswag",
    "openbookqa",
    "piqa",
    "social_i_qa",
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


def load_json_records(path: str, max_records: int, seed: int) -> List[Any]:
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
    return records


def record_to_text(ex: Any) -> str:
    if not isinstance(ex, dict):
        return str(ex)
    if "text" in ex:
        return str(ex["text"])
    return format_instruction_example(ex)


def load_local_text_dataset(path: str, max_records: int, seed: int) -> Dataset:
    records = load_json_records(path, max_records, seed)
    return Dataset.from_dict({"text": [record_to_text(ex) for ex in records]})


def tokenize_and_group(dataset: Dataset, tokenizer, block_size: int, num_proc: int, desc_prefix: str) -> Dataset:
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
        raise ValueError(f"Only {len(grouped)} LM chunks. Lower --block_size or increase data.")
    return grouped


def prepare_lm_datasets(args: argparse.Namespace, tokenizer) -> Tuple[Dataset, Dataset]:
    tokenizer.pad_token = tokenizer.eos_token
    raw = load_local_text_dataset(args.data_path, args.max_records, args.seed)
    split = raw.train_test_split(test_size=float(args.val_ratio), seed=int(args.seed), shuffle=True)
    train_ds = tokenize_and_group(split["train"], tokenizer, int(args.block_size), int(args.num_proc), "local train")
    eval_ds = tokenize_and_group(split["test"], tokenizer, int(args.block_size), int(args.num_proc), "local eval")
    return train_ds, eval_ds


def compute_target_intermediate(config, current_total: int, target_compression: float, multiple: int) -> Tuple[int, int, float]:
    hidden = int(config.hidden_size)
    inter = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    mlp_params = layers * 3 * hidden * inter
    non_mlp = current_total - mlp_params
    target_total = current_total * (1.0 - float(target_compression))
    raw_inter = (target_total - non_mlp) / max(1, layers * 3 * hidden)
    if raw_inter <= 0:
        raise ValueError(
            f"target_compression={target_compression} is too high for MLP-only pruning; raw_inter={raw_inter}"
        )
    multiple = max(1, int(multiple))
    new_inter = max(multiple, int(round(raw_inter / multiple) * multiple))
    new_inter = min(inter, new_inter)
    new_total = non_mlp + layers * 3 * hidden * new_inter
    actual = 1.0 - (new_total / current_total)
    return int(new_inter), int(new_total), float(actual)


def make_linear(weight: torch.Tensor, bias: torch.Tensor | None, device: torch.device) -> nn.Linear:
    out_features, in_features = weight.shape
    layer = nn.Linear(
        in_features,
        out_features,
        bias=bias is not None,
        dtype=weight.dtype,
        device=device,
    )
    layer.weight = nn.Parameter(weight.contiguous())
    if bias is not None:
        layer.bias = nn.Parameter(bias.contiguous())
    return layer


def calibration_texts(path: str, nsamples: int, seed: int) -> List[str]:
    records = load_json_records(path, nsamples, seed)
    return [record_to_text(ex) for ex in records]


def collect_mlp_wifv_metrics(
    model,
    tokenizer,
    data_path: str,
    nsamples: int,
    seqlen: int,
    seed: int,
    device: torch.device,
) -> List[torch.Tensor]:
    layers = list(model.model.layers)
    inter = int(model.config.intermediate_size)
    sums = [torch.zeros(inter, dtype=torch.float64) for _ in layers]
    sums_sq = [torch.zeros(inter, dtype=torch.float64) for _ in layers]
    counts = [0 for _ in layers]

    def make_hook(idx: int):
        def hook(_module, inp, _out):
            x = inp[0].detach().float()
            x = x.reshape(-1, x.shape[-1])
            sums[idx].add_(x.sum(dim=0).cpu().double())
            sums_sq[idx].add_((x * x).sum(dim=0).cpu().double())
            counts[idx] += int(x.shape[0])

        return hook

    handles = [layer.mlp.down_proj.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    model.eval()
    texts = calibration_texts(data_path, nsamples, seed)
    print(f"[FLAP-CE] collecting MLP WIFV metrics nsamples={len(texts)} seqlen={seqlen}", flush=True)
    with torch.no_grad():
        for i, text in enumerate(texts):
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=int(seqlen),
            )
            if enc["input_ids"].shape[1] < 2:
                continue
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)
            if (i + 1) % 20 == 0:
                print(f"[FLAP-CE] calibration {i + 1}/{len(texts)}", flush=True)

    for handle in handles:
        handle.remove()

    metrics = []
    for idx, layer in enumerate(layers):
        n = max(1, counts[idx])
        mean = sums[idx] / n
        var = torch.clamp((sums_sq[idx] / n) - (mean * mean), min=0.0).float()
        weight_norm = layer.mlp.down_proj.weight.detach().float().pow(2).sum(dim=0).cpu()
        metrics.append(var * weight_norm)
    return metrics


def prune_mlp_flap(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))
    ensure_dir(args.output_dir)
    dtype = dtype_from_name(args.torch_dtype)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[FLAP-CE] loading teacher={args.model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = model.config
    original_intermediate_size = int(config.intermediate_size)
    original_total = sum(p.numel() for p in model.parameters())
    new_inter, expected_total, actual_compression = compute_target_intermediate(
        config,
        original_total,
        float(args.target_compression),
        int(args.intermediate_multiple),
    )
    print(
        f"[FLAP-CE] original_total={original_total} original_intermediate={config.intermediate_size} "
        f"target={float(args.target_compression):.4f} new_intermediate={new_inter} "
        f"expected_total={expected_total} actual_compression={actual_compression:.4%}",
        flush=True,
    )

    metrics = collect_mlp_wifv_metrics(
        model,
        tokenizer,
        args.calib_data_path,
        int(args.nsamples),
        int(args.seqlen),
        int(args.seed),
        device,
    )

    selected_indices: List[List[int]] = []
    for idx, (layer, score) in enumerate(zip(model.model.layers, metrics)):
        top = torch.topk(score, k=int(new_inter), largest=True).indices.sort().values.to(device)
        selected_indices.append([int(x) for x in top.cpu().tolist()])
        mlp = layer.mlp
        mlp.gate_proj = make_linear(mlp.gate_proj.weight.detach()[top, :].clone(), None, device)
        mlp.up_proj = make_linear(mlp.up_proj.weight.detach()[top, :].clone(), None, device)
        down_bias = mlp.down_proj.bias.detach().clone() if mlp.down_proj.bias is not None else None
        mlp.down_proj = make_linear(mlp.down_proj.weight.detach()[:, top].clone(), down_bias, device)
        mlp.intermediate_size = int(new_inter)
        if idx % 4 == 0:
            print(f"[FLAP-CE] pruned layer={idx} keep={new_inter}", flush=True)

    model.config.intermediate_size = int(new_inter)
    model.config.use_cache = True
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    final_total = sum(p.numel() for p in model.parameters())
    metadata = {
        "method": "FLAP-MLP-WIFV",
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "calib_data_path": args.calib_data_path,
        "nsamples": int(args.nsamples),
        "seqlen": int(args.seqlen),
        "original_total_params": int(original_total),
        "compressed_total_params": int(final_total),
        "target_compression": float(args.target_compression),
        "actual_compression": float(1.0 - final_total / original_total),
        "original_intermediate_size": int(original_intermediate_size),
        "compressed_intermediate_size": int(new_inter),
        "selected_indices_path": os.path.join(args.output_dir, "flap_mlp_selected_indices.json"),
        "args": {k: json_safe(v) for k, v in vars(args).items() if k != "func"},
    }
    with open(metadata["selected_indices_path"], "w", encoding="utf-8") as f:
        json.dump(selected_indices, f)
    with open(os.path.join(args.output_dir, "flap_mlp_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[FLAP-CE] saved pruned model -> {args.output_dir}", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


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


def train_ce_lora(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))
    ensure_dir(args.output_dir)

    accelerator = Accelerator(
        mixed_precision=str(args.mixed_precision),
        gradient_accumulation_steps=int(args.gradient_accumulation_step),
    )
    dtype = dtype_from_name(args.torch_dtype)
    if accelerator.is_main_process:
        print(f"[FLAP-CE] loading compressed model={args.model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    if bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable()

    train_ds, eval_ds = prepare_lm_datasets(args, tokenizer)

    target_modules = [x.strip() for x in str(args.lora_target_modules).split(",") if x.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if accelerator.is_main_process:
        model.print_trainable_parameters()
        print(f"[FLAP-CE] train_chunks={len(train_ds)} eval_chunks={len(eval_ds)}", flush=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=data_collator, batch_size=int(args.batch_size))
    eval_loader = DataLoader(eval_ds, shuffle=False, collate_fn=data_collator, batch_size=int(args.batch_size))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.wd),
        betas=(0.9, 0.95),
    )
    total_steps = max(1, math.ceil(len(train_loader) / int(args.gradient_accumulation_step)) * int(args.epoches))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=float(args.min_lr))

    train_loader, eval_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader,
        eval_loader,
        model,
        optimizer,
        scheduler,
    )

    best_loss = valid_model(model, eval_loader, accelerator)
    if accelerator.is_main_process:
        print(f"[FLAP-CE] before training valid_loss={float(best_loss):.6f}", flush=True)

    best_adapter_dir = os.path.join(args.output_dir, "_best_adapter")
    best_epoch = -1
    global_update = 0
    for epoch in range(int(args.epoches)):
        model.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    global_update += 1
                optimizer.zero_grad(set_to_none=True)
            if accelerator.is_main_process and int(args.log_every) > 0 and step % int(args.log_every) == 0:
                print(
                    f"[FLAP-CE] epoch={epoch} step={step}/{len(train_loader)} "
                    f"updates={global_update} loss={float(loss.detach()):.6f}",
                    flush=True,
                )
            del outputs, loss

        valid_loss = valid_model(model, eval_loader, accelerator)
        if accelerator.is_main_process:
            print(f"[FLAP-CE] epoch={epoch} valid_loss={float(valid_loss):.6f}", flush=True)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(best_adapter_dir, safe_serialization=True)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if best_epoch >= 0:
            # The active adapter has the best weights for single-epoch runs; saving the
            # best adapter above keeps an audit trail. Merge current adapter for eval.
            pass
        merged = unwrapped.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        metadata = {
            "method": "FLAP-MLP + CE-only LoRA recovery",
            "model_name": args.model_name,
            "output_dir": args.output_dir,
            "data_path": args.data_path,
            "train_chunks": len(train_ds),
            "eval_chunks": len(eval_ds),
            "best_eval_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_target_modules": target_modules,
            "args": {k: json_safe(v) for k, v in vars(args).items() if k != "func"},
        }
        with open(os.path.join(args.output_dir, "flap_ce_recovery_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[FLAP-CE] saved recovered merged model -> {args.output_dir}", flush=True)
        print(json.dumps(metadata, indent=2), flush=True)


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
        "hidden_size": getattr(config, "hidden_size", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
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
            try:
                acc = float(row.get("accuracy", ""))
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
        writer.writerows(summary_rows)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary_rows, "per_task": all_rows}, f, indent=2)
    for row in summary_rows:
        print(f"{row['label']}: mean_accuracy={row['mean_accuracy']:.4f} tasks={row['num_tasks']}")
    print(f"[FLAP-CE] summary -> {args.output_csv}")


def add_common_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--max_records", type=int, default=100000)
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--num_proc", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_step", type=int, default=16)
    p.add_argument("--epoches", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=5e-5)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch_dtype", type=str, default="auto")
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--trust_remote_code", type=str2bool, default=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FLAP-MLP + CE recovery comparison pieces.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prune = sub.add_parser("prune-mlp", help="FLAP/WIFV MLP structured pruning into a smaller HF checkpoint.")
    p_prune.add_argument("--model_name", type=str, required=True)
    p_prune.add_argument("--output_dir", type=str, required=True)
    p_prune.add_argument("--calib_data_path", type=str, required=True)
    p_prune.add_argument("--target_compression", type=float, required=True)
    p_prune.add_argument("--intermediate_multiple", type=int, default=64)
    p_prune.add_argument("--nsamples", type=int, default=128)
    p_prune.add_argument("--seqlen", type=int, default=128)
    p_prune.add_argument("--seed", type=int, default=42)
    p_prune.add_argument("--torch_dtype", type=str, default="auto")
    p_prune.add_argument("--device", type=str, default="cuda")
    p_prune.add_argument("--trust_remote_code", type=str2bool, default=True)
    p_prune.set_defaults(func=prune_mlp_flap)

    p_train = sub.add_parser("train-ce-lora", help="CE-only LoRA recovery and merge into the compressed model.")
    add_common_train_args(p_train)
    p_train.add_argument("--lora_r", type=int, default=16)
    p_train.add_argument("--lora_alpha", type=int, default=32)
    p_train.add_argument("--lora_dropout", type=float, default=0.05)
    p_train.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p_train.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    p_train.set_defaults(func=train_ce_lora)

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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
