#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import inspect
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def torch_load_compat(path, *args, **kwargs):
    if "weights_only" not in kwargs:
        try:
            sig = inspect.signature(torch.load)
            if "weights_only" in sig.parameters:
                kwargs["weights_only"] = False
        except Exception:
            pass
    return torch.load(path, *args, **kwargs)


torch.set_grad_enabled(False)
if torch.cuda.is_available():
    try:
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        else:
            torch.backends.cuda.matmul.allow_tf32 = False

        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
            cudnn_conv.fp32_precision = "ieee"
        else:
            torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def _infer_eval_dtype():
    if torch.cuda.is_available():
        major_cc = torch.cuda.get_device_capability(0)[0]
        if major_cc >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


DTYPE_INFER = _infer_eval_dtype()


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_eval_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _sync_if_cuda(device: torch.device):
    try:
        if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        pass


def _infer_input_device(model) -> torch.device:
    dev_map = getattr(model, "hf_device_map", None)
    if isinstance(dev_map, dict):
        for key, dv in dev_map.items():
            if any(tag in key for tag in ["embed_tokens", "tok_embeddings", "wte"]):
                if isinstance(dv, str):
                    if dv in {"cpu", "meta", "disk"}:
                        return torch.device("cpu")
                    return torch.device(dv)
                return dv
        for dv in dev_map.values():
            if isinstance(dv, str) and dv not in {"cpu", "meta", "disk"}:
                return torch.device(dv)
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(tok_src: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    except Exception:
        tokenizer = LlamaTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_dataset_from_path(root: str, dataset_name: str):
    fp = os.path.join(root, dataset_name, "test.json")
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def create_batch(dataset: List[Dict[str, Any]], batch_size: int):
    n = len(dataset)
    return [dataset[i * batch_size : min((i + 1) * batch_size, n)] for i in range((n + batch_size - 1) // batch_size)]


def _answer_hint_for_dataset(ds: str) -> str:
    if ds == "boolq":
        return "Answer with exactly one word: true or false."
    if ds == "piqa":
        return "Answer with exactly one token: solution1 or solution2."
    if ds == "winogrande":
        return "Answer with exactly one token: option1 or option2."
    if ds == "hellaswag":
        return "Answer with exactly one token: ending1, ending2, ending3, or ending4."
    if ds in {"ARC-Challenge", "ARC-Easy", "openbookqa"}:
        return "Answer with exactly one token: answer1, answer2, answer3, or answer4."
    if ds == "csqa":
        return "Answer with exactly one token: answer1, answer2, answer3, answer4, or answer5."
    if ds == "social_i_qa":
        return "Answer with exactly one token: answer1, answer2, or answer3."
    return "Answer concisely."


def _candidates_for_dataset(ds: str) -> List[str]:
    if ds == "boolq":
        return ["true", "false"]
    if ds == "piqa":
        return ["solution1", "solution2"]
    if ds == "winogrande":
        return ["option1", "option2"]
    if ds == "hellaswag":
        return ["ending1", "ending2", "ending3", "ending4"]
    if ds in {"ARC-Challenge", "ARC-Easy", "openbookqa"}:
        return ["answer1", "answer2", "answer3", "answer4"]
    if ds == "csqa":
        return ["answer1", "answer2", "answer3", "answer4", "answer5"]
    if ds == "social_i_qa":
        return ["answer1", "answer2", "answer3"]
    return ["true", "false"]


def build_plain_prompt(instr: str, input_text: Optional[str] = None, hint: Optional[str] = None) -> str:
    hint = hint or ""
    instr = instr or ""
    if input_text:
        return (
            f"### Instruction:\n{instr}\n\n### Input:\n{input_text}\n\n"
            f"{hint}\n\n### Response:\nAnswer: "
        )
    return f"### Instruction:\n{instr}\n\n{hint}\n\n### Response:\nAnswer: "


@torch.no_grad()
def score_candidates_logprob(
    model,
    tokenizer,
    prompt: str,
    candidates: List[str],
    length_norm: str = "avg",
    return_perf: bool = False,
    force_autoregressive: bool = False,
):
    if bool(force_autoregressive):
        return score_candidates_logprob_autoregressive(
            model,
            tokenizer,
            prompt,
            candidates,
            length_norm=length_norm,
            return_perf=return_perf,
        )

    input_dev = _infer_input_device(model)
    # IMPORTANT:
    # Tokenization at the prompt/candidate boundary can differ if you encode them separately and concatenate.
    # Always encode the full string `prompt + candidate` and slice by the prompt prefix length.
    enc = tokenizer(prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"][0]
    base_len = int(prompt_ids.shape[0])

    concat: List[torch.Tensor] = []
    candidate_ids: List[List[int]] = []
    for c in candidates:
        # `build_plain_prompt()` ends with "Answer: " so this is usually the desired join.
        joiner = "" if (prompt.endswith(" ") or prompt.endswith("\n") or prompt.endswith("\t")) else " "
        full = tokenizer(prompt + joiner + c, return_tensors="pt")["input_ids"][0]
        if int(full.shape[0]) < base_len or not torch.equal(full[:base_len], prompt_ids):
            # Fallback to legacy heuristic if prefix mismatch (should be rare).
            ids = tokenizer.encode(" " + c, add_special_tokens=False) or tokenizer.encode(c, add_special_tokens=False)
            full = torch.cat([prompt_ids, torch.tensor(ids, dtype=torch.long)], dim=0)
        ids = full[base_len:].tolist()
        concat.append(full)
        candidate_ids.append(ids)
    max_len = max(x.shape[0] for x in concat)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    batch = torch.full((len(concat), max_len), fill_value=pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(batch)
    for i, seq in enumerate(concat):
        ln = seq.shape[0]
        batch[i, :ln] = seq
        attention_mask[i, :ln] = 1

    batch = batch.to(input_dev)
    attention_mask = attention_mask.to(input_dev)

    out = model(input_ids=batch, attention_mask=attention_mask, return_dict=True, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[0]
    logp = torch.log_softmax(logits.float()[:, :-1, :], dim=-1)

    scores: Dict[str, float] = {}
    for i, ids in enumerate(candidate_ids):
        if len(ids) == 0:
            scores[candidates[i]] = float("-inf")
            continue
        ids_t = torch.tensor(ids, dtype=torch.long, device=input_dev)
        start = base_len - 1
        stop = start + len(ids)
        token_lp = logp[i, start:stop, :].gather(1, ids_t[:, None]).squeeze(1)
        sc = token_lp.sum()
        if length_norm == "avg":
            sc = sc / max(1, len(ids))
        scores[candidates[i]] = float(sc.item())

    best = max(scores.items(), key=lambda x: x[1])[0]
    if return_perf:
        return best, scores, {"forward_tokens": int(attention_mask.sum().item())}
    return best, scores


@torch.no_grad()
def score_candidates_logprob_autoregressive(
    model,
    tokenizer,
    prompt: str,
    candidates: List[str],
    length_norm: str = "avg",
    return_perf: bool = False,
):
    input_dev = _infer_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"][0].to(input_dev)
    base_len = int(prompt_ids.shape[0])

    scores: Dict[str, float] = {}
    total_forward_tokens = 0
    for cand in candidates:
        joiner = "" if (prompt.endswith(" ") or prompt.endswith("\n") or prompt.endswith("\t")) else " "
        full = tokenizer(prompt + joiner + cand, return_tensors="pt")["input_ids"][0]
        if int(full.shape[0]) < base_len or not torch.equal(full[:base_len], prompt_ids.cpu()):
            # Fallback to legacy heuristic if prefix mismatch (should be rare).
            ids = tokenizer.encode(" " + cand, add_special_tokens=False) or tokenizer.encode(cand, add_special_tokens=False)
        else:
            ids = full[base_len:].tolist()
        if len(ids) == 0:
            scores[cand] = float("-inf")
            continue

        seq = prompt_ids.unsqueeze(0)
        attention_mask = torch.ones_like(seq, device=input_dev, dtype=torch.long)
        sc = torch.tensor(0.0, device=input_dev, dtype=torch.float32)

        for tid in ids:
            out = model(input_ids=seq, attention_mask=attention_mask, return_dict=True, use_cache=False)
            logits = out.logits if hasattr(out, "logits") else out[0]
            next_lp = torch.log_softmax(logits.float()[:, -1, :], dim=-1)[0, int(tid)]
            sc = sc + next_lp
            total_forward_tokens += int(attention_mask.sum().item())

            tid_t = torch.tensor([[int(tid)]], device=input_dev, dtype=torch.long)
            seq = torch.cat([seq, tid_t], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=input_dev, dtype=attention_mask.dtype)],
                dim=1,
            )

        if length_norm == "avg":
            sc = sc / max(1, len(ids))
        scores[cand] = float(sc.item())

    best = max(scores.items(), key=lambda x: x[1])[0]
    if return_perf:
        return best, scores, {"forward_tokens": int(total_forward_tokens)}
    return best, scores


@torch.no_grad()
def predict_by_generate(model, tokenizer, prompt: str, candidates: List[str], max_new_tokens: int = 4, return_perf: bool = False):
    input_dev = _infer_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt").to(input_dev)
    gen = model.generate(
        **enc,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.decode(gen[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip().lower()
    out_head = out.split()[0] if out else ""

    def _pick_candidate_from_text(text: str) -> Tuple[str, str]:
        head = text.split()[0] if text else ""
        # 1) strict head match (legacy behavior)
        for cand in candidates:
            if head.startswith(cand.lower()):
                return cand, "head"

        # 2) whole-token substring match (more robust for outputs like "the answer is answer2")
        best = None
        best_pos = None
        for cand in candidates:
            pat = re.compile(rf"(?<![a-z0-9_]){re.escape(cand.lower())}(?![a-z0-9_])")
            m = pat.search(text)
            if m is None:
                continue
            if best_pos is None or int(m.start()) < int(best_pos):
                best = cand
                best_pos = int(m.start())
        if best is not None:
            return str(best), "substring"

        # 3) explicit "answer: 2" digit shortcut (common failure mode for MCQ)
        m = re.search(r"(?:^|[^a-z0-9_])(answer|ans)\s*[:=：]?\s*([0-9])(?:[^a-z0-9_]|$)", text)
        if m is not None:
            d = str(m.group(2))
            for cand in candidates:
                if cand.lower().endswith(d):
                    return cand, "answer_digit"

        # 4) digit shortcut (e.g. model outputs "2" for candidates ["answer1","answer2",...])
        if head.isdigit():
            for cand in candidates:
                if cand.lower().endswith(head):
                    return cand, "digit_suffix"

        return candidates[0], "default"

    picked, match_mode = _pick_candidate_from_text(out)
    meta = {
        "raw_head": out_head,
        "raw": (out[:256] if out else ""),
        "match": match_mode,
    }
    if return_perf:
        return picked, meta, {"forward_tokens": int(gen.numel())}
    return picked, meta


@torch.no_grad()
def evaluate_model_on_dataset(args, model, tokenizer, dataset_name: str, variant_name: str):
    dataset = load_dataset_from_path(args.test_data_path, dataset_name)
    batches = create_batch(dataset, int(args.batch_size))
    hint = _answer_hint_for_dataset(dataset_name)
    candidates = _candidates_for_dataset(dataset_name)

    correct = 0
    total = 0
    ok_cnt = 0
    skip_cnt = 0
    first_err = None
    logprob_fallback_cnt = 0
    first_logprob_fallback = None
    uses_predictor_model = bool(getattr(model, "uses_predictor", False))
    requested_fast_logprob = bool(getattr(args, "predictor_fast_logprob", False))
    predictor_ode_steps = int(getattr(model, "ode_steps", getattr(args, "student_ode_steps", 1)))
    predictor_injection_window = int(
        getattr(model, "injection_window", getattr(args, "student_injection_window", 0))
    )
    predictor_eval_mode = str(
        getattr(model, "predictor_eval_mode", getattr(args, "student_predictor_eval_mode", "auto"))
    ).strip().lower()
    predictor_use_last_token = bool(
        getattr(model, "use_last_token", getattr(args, "student_use_last_token", True))
    )
    hf_hook_enabled = bool(getattr(model, "_use_hf_forward_hooks", False))
    predictor_requires_ar = bool(
        uses_predictor_model
        and (
            (predictor_eval_mode == "rollout" and predictor_ode_steps != 1)
            or predictor_injection_window != 0
            or (not predictor_use_last_token)
        )
    )
    force_ar_override = bool(getattr(args, "force_ar_logprob", False))
    # Allow fast one-pass logprob on predictor models when injection is single-step last-token only.
    use_fast_logprob_cfg = bool(requested_fast_logprob or (uses_predictor_model and not predictor_requires_ar))
    force_ar_for_logprob_cfg = bool(uses_predictor_model and predictor_requires_ar)
    if force_ar_override:
        force_ar_for_logprob_cfg = True
        use_fast_logprob_cfg = False
    ar_rescore_topk_cfg = int(getattr(args, "predictor_ar_rescore_topk", 0))
    logprob_fast_cnt = 0
    logprob_forced_ar_cnt = 0
    logprob_ar_rescore_cnt = 0
    pred_counts = {c: 0 for c in candidates}
    pred_other_cnt = 0
    total_forward_tokens = 0
    t0 = time.perf_counter()
    if hasattr(model, "reset_profile"):
        try:
            model.reset_profile()
        except Exception:
            pass

    iterator = batches
    if tqdm is not None:
        iterator = tqdm(batches, desc=f"{variant_name}|{dataset_name}")
    if args.eval_mode == "logprob":
        print(
            f"[Eval logprob] {variant_name}|{dataset_name} "
            f"uses_predictor={uses_predictor_model} fast={use_fast_logprob_cfg} "
            f"force_ar={force_ar_for_logprob_cfg} ar_rescore_topk={ar_rescore_topk_cfg} "
            f"| ode_steps={predictor_ode_steps} injection_window={predictor_injection_window} "
            f"use_last_token={predictor_use_last_token} predictor_mode={predictor_eval_mode} "
            f"hf_hook={hf_hook_enabled}",
            flush=True,
        )

    for batch in iterator:
        for dp in batch:
            try:
                instr = dp.get("instruction") or dp.get("question") or ""
                inp = dp.get("input") or dp.get("context") or None
                ans = dp.get("answer") or dp.get("label")
                if isinstance(ans, int):
                    if 0 <= ans < len(candidates):
                        ans = candidates[ans]
                    else:
                        ans = str(ans)
                if isinstance(ans, bool):
                    ans = str(ans).lower()
                if isinstance(ans, str):
                    ans = ans.strip().lower()

                prompt = build_plain_prompt(instr, inp, hint)

                if args.eval_mode == "logprob":
                    try:
                        if force_ar_for_logprob_cfg:
                            logprob_forced_ar_cnt += 1
                        else:
                            logprob_fast_cnt += 1
                        pred, scores, perf = score_candidates_logprob(
                            model,
                            tokenizer,
                            prompt,
                            candidates,
                            args.length_norm,
                            return_perf=True,
                            force_autoregressive=force_ar_for_logprob_cfg,
                        )
                        if (
                            uses_predictor_model
                            and use_fast_logprob_cfg
                            and ar_rescore_topk_cfg > 0
                            and len(candidates) > 1
                        ):
                            k = max(1, min(int(ar_rescore_topk_cfg), len(candidates)))
                            topk_cands = [
                                cand for cand, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
                            ]
                            _, ar_scores, ar_perf = score_candidates_logprob_autoregressive(
                                model,
                                tokenizer,
                                prompt,
                                topk_cands,
                                length_norm=args.length_norm,
                                return_perf=True,
                            )
                            scores.update(ar_scores)
                            pred = max(scores.items(), key=lambda x: x[1])[0]
                            perf["forward_tokens"] = int(perf.get("forward_tokens", 0)) + int(
                                ar_perf.get("forward_tokens", 0)
                            )
                            logprob_ar_rescore_cnt += 1
                    except Exception as e:
                        logprob_fallback_cnt += 1
                        if first_logprob_fallback is None:
                            first_logprob_fallback = f"{type(e).__name__}: {e}"
                        pred, scores, perf = predict_by_generate(
                            model, tokenizer, prompt, candidates, args.max_new_tokens, return_perf=True
                        )
                else:
                    pred, scores, perf = predict_by_generate(
                        model, tokenizer, prompt, candidates, args.max_new_tokens, return_perf=True
                    )
                total_forward_tokens += int(perf.get("forward_tokens", 0))

                if args.print_scores:
                    print(f"[SAMPLE] pred={pred} ans={ans} scores={scores}")

                if isinstance(ans, str) and pred == ans:
                    correct += 1
                if pred in pred_counts:
                    pred_counts[pred] += 1
                else:
                    pred_other_cnt += 1
                total += 1
                ok_cnt += 1
            except Exception as e:
                skip_cnt += 1
                if first_err is None:
                    first_err = f"{type(e).__name__}: {e}"

    elapsed = max(1e-9, time.perf_counter() - t0)
    accuracy = (correct / total) if total > 0 else None
    throughput_toks = (float(total_forward_tokens) / elapsed) if total_forward_tokens > 0 else None
    avg_latency_ms = (1000.0 * elapsed / max(1, ok_cnt)) if ok_cnt > 0 else None
    base_time_sec = None
    predictor_time_sec = None
    combined_time_sec = None
    predictor_steps = None
    if hasattr(model, "get_profile"):
        try:
            prof = model.get_profile() or {}
            base_time_sec = float(prof.get("base_time_sec", 0.0))
            predictor_time_sec = float(prof.get("predictor_time_sec", 0.0))
            combined_time_sec = float(prof.get("combined_time_sec", 0.0))
            predictor_steps = int(prof.get("predictor_steps", 0))
        except Exception:
            pass
    if logprob_fallback_cnt > 0:
        print(
            f"[Warn] {variant_name}|{dataset_name} logprob->generate fallback count={logprob_fallback_cnt}; "
            f"first_error={first_logprob_fallback}",
            flush=True,
        )
    if args.eval_mode == "logprob":
        print(
            f"[Eval logprob summary] {variant_name}|{dataset_name} "
            f"fast_calls={logprob_fast_cnt} forced_ar_calls={logprob_forced_ar_cnt} "
            f"ar_rescore_calls={logprob_ar_rescore_cnt} fallbacks={logprob_fallback_cnt}",
            flush=True,
        )
    pred_parts = [f"{k}:{v}" for k, v in pred_counts.items()]
    if pred_other_cnt > 0:
        pred_parts.append(f"other:{pred_other_cnt}")
    print(
        f"[Pred dist] {variant_name}|{dataset_name} total={total} "
        + " ".join(pred_parts),
        flush=True,
    )

    return (
        accuracy,
        ok_cnt,
        skip_cnt,
        first_err,
        throughput_toks,
        avg_latency_ms,
        total_forward_tokens,
        elapsed,
        base_time_sec,
        predictor_time_sec,
        combined_time_sec,
        predictor_steps,
    )


def _read_json_if_exists(path: str) -> dict:
    try:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _find_latest_student_final(out_root: str) -> Optional[str]:
    if not out_root:
        return None
    pattern = os.path.join(out_root, "*", "student_final")
    candidates = [path for path in glob.glob(pattern) if os.path.isdir(path)]
    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def _infer_num_layers(model: nn.Module) -> int:
    try:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if isinstance(layers, nn.ModuleList):
            return len(layers)
    except Exception:
        pass
    try:
        return int(getattr(model.config, "num_hidden_layers", 0) or 0)
    except Exception:
        return 0


def _expand_or_validate_list(value, n_layers: int, name: str):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [value] * n_layers
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return [value[0]] * n_layers
        if len(value) == n_layers:
            return list(value)
    raise ValueError(f"Invalid {name}: expected scalar/list with len 1 or {n_layers}, got {type(value)}")


def _apply_stateft_control_from_dir(
    model: nn.Module,
    control_dir: str,
    *,
    enable: bool,
    label: str,
    missing_policy: str = "error",
) -> Tuple[nn.Module, bool, str]:
    if not bool(enable):
        return model, False, "disabled"
    policy = str(missing_policy).strip().lower()
    if policy not in {"error", "disable"}:
        raise ValueError(f"{label}: invalid missing_policy={missing_policy}; expected error|disable")
    cfg_path = os.path.join(control_dir, "control_config.json")
    state_path = os.path.join(control_dir, "control_state.pt")
    if not os.path.exists(cfg_path) or not os.path.exists(state_path):
        if policy == "disable":
            msg = f"disabled(missing_control_files:{control_dir})"
            print(f"[Control] {label}: {msg}")
            return model, False, msg
        raise FileNotFoundError(
            f"{label}: student_use_control=True but missing control files under {control_dir}"
        )

    try:
        import stateft as _stateft
    except Exception as e:
        raise RuntimeError(f"{label}: cannot import stateft.py ({e})") from e

    cfg = _read_json_if_exists(cfg_path)
    n_layers = _infer_num_layers(model)
    if n_layers <= 0:
        raise RuntimeError(f"{label}: cannot infer layer count before control wrapping")
    model_device = None
    model_dtype = None
    try:
        ref_param = None
        for p in model.parameters():
            if p.is_floating_point():
                ref_param = p
                break
        if ref_param is None:
            ref_param = next(model.parameters())
        model_device = ref_param.device
        if ref_param.is_floating_point():
            model_dtype = ref_param.dtype
    except Exception:
        model_device = None
        model_dtype = None

    ranks = _expand_or_validate_list(cfg.get("ranks"), n_layers, "ranks")
    alphas = _expand_or_validate_list(cfg.get("alphas"), n_layers, "alphas")
    if ranks is None or alphas is None:
        base_rank = int(cfg.get("rank", 64))
        base_alpha = float(cfg.get("alpha", 1.0))
        ranks, alphas = _stateft.build_rank_alpha_schedules(
            n_layers=n_layers,
            base_rank=base_rank,
            base_alpha=base_alpha,
            rank_schedule=str(cfg.get("rank_schedule", "fixed")),
            alpha_schedule=str(cfg.get("alpha_schedule", "fixed")),
            rank_min=cfg.get("rank_min", None),
            rank_max=cfg.get("rank_max", None),
            preserve_rank_budget=bool(cfg.get("preserve_rank_budget", True)),
            rank_list_str=cfg.get("rank_list_str", None),
            alpha_list_str=cfg.get("alpha_list_str", None),
        )
    ranks = [int(x) for x in ranks]
    alphas = [float(x) for x in alphas]

    model = _stateft.wrap_model_with_control(
        model=model,
        method=str(cfg.get("method", "double")),
        ranks=ranks,
        alphas=alphas,
        dropout=float(cfg.get("dropout", 0.0)),
        checkpoint_base=bool(cfg.get("checkpoint_base", False)),
        alpha_mode=str(cfg.get("alpha_mode", "scalar")),
        alpha_init=float(cfg.get("alpha_init", 1.0)),
        alpha_learnable=bool(cfg.get("alpha_learnable", True)),
        gate_type=str(cfg.get("gate_type", "none")),
        gate_tokenwise=bool(cfg.get("gate_tokenwise", False)),
        gate_init_w0=float(cfg.get("gate_init_w0", 0.0)),
        gate_init_w1=float(cfg.get("gate_init_w1", 1.0)),
        eps=float(cfg.get("eps", 1e-6)),
        hash_enable=bool(cfg.get("hash_enable", False)),
        hash_m_in=int(cfg.get("hash_m_in", 0) or 0),
        hash_m_out=int(cfg.get("hash_m_out", 0) or 0),
        hash_seed_in=int(cfg.get("hash_seed_in", 13) or 13),
        hash_seed_out=int(cfg.get("hash_seed_out", 17) or 17),
        hash_signed=bool(cfg.get("hash_signed", True)),
    )
    _stateft.load_control_state_if_any(model, control_dir)
    try:
        if model_device is not None and model_dtype is not None:
            model = model.to(device=model_device, dtype=model_dtype)
        elif model_device is not None:
            model = model.to(device=model_device)
    except Exception as e:
        raise RuntimeError(f"{label}: failed to move controlled model back to target device ({e})") from e
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    msg = f"enabled(control_dir={control_dir}, layers={n_layers})"
    print(f"[Control] {label}: {msg}")
    return model, True, msg


def _build_stateft_teacher_local(
    teacher_ckpt_dir: str,
    *,
    base_model: str,
    target_dtype: torch.dtype,
):
    cfg = _read_json_if_exists(os.path.join(teacher_ckpt_dir, "control_config.json"))

    import stateft as _stateft

    base_model_name = base_model or cfg.get("base_model") or cfg.get("model_name_or_path")
    if not base_model_name:
        raise ValueError("Missing base model for StateFT teacher.")

    config = None
    local_cfg = os.path.join(teacher_ckpt_dir, "config.json")
    if os.path.exists(local_cfg):
        try:
            config = AutoConfig.from_pretrained(teacher_ckpt_dir)
            print(f"✅ Loaded patched config from {teacher_ckpt_dir}")
        except Exception as e:
            print(f"⚠️ Failed to load local config: {e}")

    teacher = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        config=config,
        torch_dtype=target_dtype,
    )
    teacher.config.use_cache = False
    teacher.config.output_hidden_states = True

    try:
        n_layers = len(teacher.model.layers)  # type: ignore[attr-defined]
    except Exception:
        n_layers = int(getattr(teacher.config, "num_hidden_layers", 0))
    if n_layers <= 0:
        raise RuntimeError("Cannot infer number of layers from base model.")

    ranks = cfg.get("ranks")
    alphas = cfg.get("alphas")
    if ranks is None or alphas is None:
        base_rank = int(cfg.get("rank", 64))
        base_alpha = float(cfg.get("alpha", 1.0))
        ranks, alphas = _stateft.build_rank_alpha_schedules(
            n_layers=n_layers,
            base_rank=base_rank,
            base_alpha=base_alpha,
            rank_schedule=str(cfg.get("rank_schedule", "fixed")),
            alpha_schedule=str(cfg.get("alpha_schedule", "fixed")),
            rank_min=cfg.get("rank_min", None),
            rank_max=cfg.get("rank_max", None),
            preserve_rank_budget=bool(cfg.get("preserve_rank_budget", True)),
            rank_list_str=cfg.get("rank_list_str", None),
            alpha_list_str=cfg.get("alpha_list_str", None),
        )
    ranks = [int(x) for x in ranks]
    alphas = [float(x) for x in alphas]

    teacher = _stateft.wrap_model_with_control(
        model=teacher,
        method=str(cfg.get("method", "double")),
        ranks=ranks,
        alphas=alphas,
        dropout=float(cfg.get("dropout", 0.0)),
        checkpoint_base=bool(cfg.get("checkpoint_base", False)),
        alpha_mode=str(cfg.get("alpha_mode", "scalar")),
        alpha_init=float(cfg.get("alpha_init", 1.0)),
        alpha_learnable=bool(cfg.get("alpha_learnable", True)),
        gate_type=str(cfg.get("gate_type", "none")),
        gate_tokenwise=bool(cfg.get("gate_tokenwise", False)),
        gate_init_w0=float(cfg.get("gate_init_w0", 0.0)),
        gate_init_w1=float(cfg.get("gate_init_w1", 1.0)),
        eps=float(cfg.get("eps", 1e-6)),
        hash_enable=bool(cfg.get("hash_enable", False)),
        hash_m_in=int(cfg.get("hash_m_in", 0) or 0),
        hash_m_out=int(cfg.get("hash_m_out", 0) or 0),
        hash_seed_in=int(cfg.get("hash_seed_in", 13) or 13),
        hash_seed_out=int(cfg.get("hash_seed_out", 17) or 17),
        hash_signed=bool(cfg.get("hash_signed", True)),
    )
    _stateft.load_control_state_if_any(teacher, teacher_ckpt_dir)

    teacher.eval()
    teacher.to(dtype=target_dtype)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _build_native_teacher_local(
    teacher_source: str,
    *,
    target_dtype: torch.dtype,
) -> nn.Module:
    teacher_source = str(teacher_source or "").strip()
    if not teacher_source:
        raise ValueError("teacher_source is required for native teacher loader.")
    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers is required.")

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_source,
        torch_dtype=target_dtype,
    )
    teacher.config.use_cache = False
    teacher.config.output_hidden_states = True
    teacher.eval()
    teacher.to(dtype=target_dtype)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def load_stateft_teacher_model(args):
    device = _resolve_eval_device(args.device)
    target_dtype = DTYPE_INFER if device.type == "cuda" else torch.float32

    teacher_loader = str(getattr(args, "teacher_loader", "auto")).lower().strip()
    if teacher_loader not in {"auto", "stateft", "native"}:
        raise ValueError(f"Invalid teacher_loader={teacher_loader!r} (expected: auto|stateft|native).")
    has_stateft_cfg = bool(getattr(args, "teacher_ckpt", "")) and os.path.exists(
        os.path.join(str(args.teacher_ckpt), "control_config.json")
    )
    resolved_loader = "stateft" if (teacher_loader == "auto" and has_stateft_cfg) else (
        "native" if teacher_loader == "auto" else teacher_loader
    )
    setattr(args, "_resolved_teacher_loader", resolved_loader)

    teacher = None
    err_msgs = []
    try:
        from phase1_metric_pca import build_frozen_teacher

        teacher = build_frozen_teacher(
            teacher_ckpt_dir=args.teacher_ckpt,
            base_model=args.base_model,
            torch_dtype=target_dtype,
            teacher_loader=resolved_loader,
        )
    except Exception as e:
        err_msgs.append(f"phase1_metric_pca loader failed: {e}")

    if teacher is None and resolved_loader == "stateft":
        try:
            teacher = _build_stateft_teacher_local(
                args.teacher_ckpt,
                base_model=args.base_model,
                target_dtype=target_dtype,
            )
        except Exception as e:
            err_msgs.append(f"local loader failed: {e}")
    elif teacher is None and resolved_loader == "native":
        try:
            native_source = str(args.teacher_ckpt or args.base_model).strip()
            teacher = _build_native_teacher_local(
                native_source,
                target_dtype=target_dtype,
            )
        except Exception as e:
            err_msgs.append(f"local native loader failed: {e}")

    if teacher is None:
        raise RuntimeError(" ; ".join(err_msgs))

    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _read_anchors(anchors_json: str) -> List[int]:
    with open(anchors_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    anchors = obj.get("anchors")
    if not isinstance(anchors, list) or len(anchors) < 2:
        raise ValueError(f"Invalid anchors_json: {anchors_json}")
    return [int(x) for x in anchors]


def load_pca_q_dir_for_eval(pca_q_dir: str) -> Tuple[Dict[int, torch.Tensor], int]:
    if not pca_q_dir or not os.path.isdir(pca_q_dir):
        raise ValueError(f"pca_q_dir not found: {pca_q_dir}")

    q_map: Dict[int, torch.Tensor] = {}
    rank = None

    segment_q_path = os.path.join(pca_q_dir, "segment_q.pt")
    if os.path.isfile(segment_q_path):
        seg_pack = torch_load_compat(segment_q_path, map_location="cpu")
        q_by_anchor = seg_pack.get("Q_by_anchor_start", {})
        anchors = [int(x) for x in seg_pack.get("anchors", []) if x is not None]
        for j in range(max(0, len(anchors) - 1)):
            a = int(anchors[j])
            q = q_by_anchor.get(str(a))
            if not torch.is_tensor(q):
                continue
            q = q.detach().float().cpu()
            if rank is None:
                rank = int(q.shape[1])
            elif int(q.shape[1]) != int(rank):
                raise ValueError(f"Inconsistent rank in segment_q.pt: anchor={a}")
            q_map[int(a)] = q

    if not q_map:
        seg_files = sorted([f for f in os.listdir(pca_q_dir) if f.startswith("Q_seg") and f.endswith(".pt")])
        for fn in seg_files:
            ckpt = torch_load_compat(os.path.join(pca_q_dir, fn), map_location="cpu")
            q = ckpt.get("Q")
            if not torch.is_tensor(q):
                continue
            q = q.detach().float().cpu()
            a = ckpt.get("start_anchor", ckpt.get("layer_id"))
            if a is None:
                continue
            if rank is None:
                rank = int(q.shape[1])
            elif int(q.shape[1]) != int(rank):
                raise ValueError(f"Inconsistent rank in {fn}")
            q_map[int(a)] = q

    if not q_map:
        files = sorted([f for f in os.listdir(pca_q_dir) if f.startswith("Q_block") and f.endswith(".pt")])
        is_block = True
        if not files:
            files = sorted([f for f in os.listdir(pca_q_dir) if f.startswith("Q_pca_layer") and f.endswith(".pt")])
            is_block = False
        if not files:
            raise ValueError(f"No Q files found in {pca_q_dir}")

        for fn in files:
            ckpt = torch_load_compat(os.path.join(pca_q_dir, fn), map_location="cpu")
            q = ckpt["Q"].detach().float().cpu()
            if rank is None:
                rank = int(q.shape[1])
            elif int(q.shape[1]) != int(rank):
                raise ValueError(f"Inconsistent rank in {fn}")

            if is_block:
                for lid in ckpt.get("block_layers", []):
                    q_map[int(lid)] = q
            else:
                lid = ckpt.get("layer_id", None)
                if lid is not None:
                    q_map[int(lid)] = q
    if rank is None:
        raise ValueError("No valid PCA Q loaded.")
    return q_map, int(rank)


def _build_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    bsz, seqlen = attention_mask.shape
    device = attention_mask.device
    neg_inf = torch.finfo(dtype).min
    causal = torch.triu(torch.full((seqlen, seqlen), neg_inf, device=device, dtype=dtype), diagonal=1)
    causal = causal[None, None, :, :].expand(bsz, 1, seqlen, seqlen)
    key_pad = (1.0 - attention_mask.float())[:, None, None, :].to(dtype) * neg_inf
    return causal + key_pad


def _call_decoder_layer(
    layer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings=None,
) -> torch.Tensor:
    call_specs = [
        {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            "use_cache": False,
            "output_attentions": False,
        },
        {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        },
        {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
            "output_attentions": False,
        },
    ]
    last_error = None
    for spec in call_specs:
        try:
            out = layer(hidden_states, **spec)
            return out[0] if isinstance(out, (tuple, list)) else out
        except TypeError as e:
            last_error = e
            continue
    raise RuntimeError(f"Failed to call decoder layer: {last_error}")


def _last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    idx = attention_mask.to(dtype=torch.long).sum(dim=1) - 1
    return torch.clamp(idx, min=0)


def _last_pred_index(attention_mask: torch.Tensor, *, append_eos: bool) -> torch.Tensor:
    shift = 1 if bool(append_eos) else 0
    return torch.clamp(_last_token_index(attention_mask) - shift, min=0)


def _select_last_pred_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor, *, append_eos: bool):
    idx = _last_pred_index(attention_mask, append_eos=append_eos)
    b = torch.arange(hidden_states.size(0), device=hidden_states.device)
    h_last = hidden_states[b, idx].unsqueeze(1)
    return h_last, idx


def _build_last_pred_injection_mask(
    attention_mask: torch.Tensor,
    pred_idx: torch.Tensor,
    *,
    window_radius: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    bsz, seqlen = attention_mask.shape
    if int(window_radius) <= 0:
        return F.one_hot(pred_idx, num_classes=seqlen).to(dtype=dtype, device=device).unsqueeze(-1)
    pos = torch.arange(seqlen, device=device).view(1, seqlen)
    center = pred_idx.view(bsz, 1)
    mask = (torch.abs(pos - center) <= int(window_radius)).to(dtype=dtype)
    mask = mask * attention_mask.to(dtype=dtype)
    mask = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return mask.unsqueeze(-1)


def _apply_rank_linear(x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, mat.transpose(-1, -2))


def _clip_delta_h_per_token(delta_h: torch.Tensor, clip_norm: float, eps: float = 1e-6) -> torch.Tensor:
    clip_v = float(clip_norm)
    if clip_v <= 0.0:
        return delta_h
    norms = delta_h.to(torch.float32).norm(dim=-1, keepdim=True).clamp(min=float(eps))
    scales = torch.clamp(torch.tensor(clip_v, device=delta_h.device, dtype=norms.dtype) / norms, max=1.0)
    return delta_h * scales.to(dtype=delta_h.dtype)


class PredictorMLP(nn.Module):
    def __init__(
        self,
        rank: int,
        mlp_mult: int = 4,
        mlp_depth: int = 2,
        dropout: float = 0.0,
        time_hidden: int = 128,
    ):
        super().__init__()
        hid = int(rank * mlp_mult)
        self.ln = nn.LayerNorm(rank)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_hidden),
            nn.SiLU(),
            nn.Linear(time_hidden, rank),
        )
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(rank),
            nn.Linear(rank, rank),
        )
        depth = max(1, int(mlp_depth))
        layers = [nn.Linear(rank, hid), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hid, hid), nn.SiLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hid, rank)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor, z_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if t.dim() == 3:
            t_in = t[:, 0, :]
        else:
            t_in = t
        t_embed = self.time_mlp(t_in).unsqueeze(1).expand(-1, z.size(1), -1)
        z_in = z + t_embed
        if z_cond is not None:
            z_in = z_in + self.cond_proj(z_cond)
        return self.mlp(self.ln(z_in))


def _predict_z(
    predictor: PredictorMLP,
    z_cond: torch.Tensor,
    *,
    ode_steps: int = 1,
    z0_mode: str = "zero",
    z0_std: float = 1.0,
):
    if z0_mode not in {"random", "zero"}:
        raise ValueError("z0_mode must be random or zero")
    if z0_mode == "random":
        z = torch.randn_like(z_cond) * float(z0_std)
    else:
        z = torch.zeros_like(z_cond)
    n_steps = max(1, int(ode_steps))
    dt = 1.0 / float(n_steps)
    for k in range(n_steps):
        t_val = (k + 0.5) / float(n_steps)
        t = torch.full((z.size(0), z.size(1), 1), fill_value=t_val, device=z.device, dtype=z.dtype)
        v = predictor(z, t, z_cond)
        z = z + dt * v
    return z


def _predict_delta_u(
    predictor: PredictorMLP,
    z_state: torch.Tensor,
    *,
    predictor_eval_mode: str = "direct",
    ode_steps: int = 1,
    z0_mode: str = "zero",
    z0_std: float = 1.0,
) -> torch.Tensor:
    mode = str(predictor_eval_mode).strip().lower()
    if mode == "direct":
        t0 = torch.zeros((z_state.size(0), 1, 1), device=z_state.device, dtype=z_state.dtype)
        return predictor(z_state, t0, z_cond=z_state)
    if mode == "rollout":
        return _predict_z(
            predictor,
            z_state,
            ode_steps=int(ode_steps),
            z0_mode=str(z0_mode),
            z0_std=float(z0_std),
        )
    raise ValueError(f"Unknown predictor_eval_mode={predictor_eval_mode!r} (expected direct|rollout)")


class _EvalOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def __getitem__(self, idx):
        return (self.logits,)[idx]


class StudentPredictorCausalLM(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        anchors: List[int],
        q_map: Dict[int, torch.Tensor],
        predictors: nn.ModuleList,
        *,
        mu_delta: Optional[torch.Tensor],
        use_last_token: bool,
        append_eos: bool,
        use_mu_delta: bool,
        ode_steps: int,
        pred_z0_mode: str,
        pred_z0_std: float,
        predictor_eval_mode: str = "direct",
        injection_window: int = 0,
        use_whiten: bool = False,
        seg_L: Optional[List[torch.Tensor]] = None,
        seg_Linv: Optional[List[torch.Tensor]] = None,
        segment_gate_logits: Optional[torch.Tensor] = None,
        segment_gate_enabled: bool = False,
        guiders: Optional[nn.ModuleList] = None,
        gate_heads: Optional[nn.ModuleList] = None,
        enable_gating: bool = False,
        gate_threshold: float = 0.5,
        gate_use_soft_scale: bool = True,
        gate_scale_floor: float = 0.0,
        gate_scale_temp: float = 1.0,
        seg_chart_risk: Optional[List[float]] = None,
        profile_sync_timing: bool = False,
        injection_debug: bool = False,
        injection_debug_max_prints: int = 1,
        delta_h_clip_norm: float = 0.0,
        segment_gate_fixed_value: Optional[float] = None,
        enable_hf_hook: bool = True,
    ):
        super().__init__()
        self.student = student
        self.anchors = [int(x) for x in anchors]
        self.num_intervals = len(self.anchors) - 1
        self.predictors = predictors
        self.mu_delta = mu_delta
        self.use_last_token = bool(use_last_token)
        self.append_eos = bool(append_eos)
        self.use_mu_delta = bool(use_mu_delta)
        self.ode_steps = int(ode_steps)
        self.pred_z0_mode = str(pred_z0_mode)
        self.pred_z0_std = float(pred_z0_std)
        self.predictor_eval_mode = str(predictor_eval_mode).strip().lower()
        if self.predictor_eval_mode not in {"direct", "rollout"}:
            raise ValueError(
                f"Invalid predictor_eval_mode={predictor_eval_mode!r} (expected direct|rollout)"
            )
        self.rank = int(self.predictors[0].ln.weight.numel())
        self.injection_window = max(0, int(injection_window))
        self.use_whiten = bool(use_whiten)
        self.guiders = guiders
        self.gate_heads = gate_heads
        self.enable_gating = bool(enable_gating) and (gate_heads is not None)
        self.gate_threshold = float(gate_threshold)
        self.gate_use_soft_scale = bool(gate_use_soft_scale)
        self.gate_scale_floor = float(gate_scale_floor)
        self.gate_scale_temp = float(max(1e-3, gate_scale_temp))
        self.profile_sync_timing = bool(profile_sync_timing)
        self.injection_debug = bool(injection_debug)
        self.injection_debug_max_prints = max(0, int(injection_debug_max_prints))
        self._injection_debug_count = 0
        self.delta_h_clip_norm = float(max(0.0, float(delta_h_clip_norm)))
        self.segment_gate_fixed_value = (
            None if segment_gate_fixed_value is None else float(segment_gate_fixed_value)
        )
        self.enable_hf_hook = bool(enable_hf_hook)
        self.config = getattr(student, "config", None)
        self.generation_config = getattr(student, "generation_config", None)
        self.uses_predictor = True
        self._profile = {
            "forward_calls": 0,
            "predictor_steps": 0,
            "base_time_sec": 0.0,
            "predictor_time_sec": 0.0,
            "combined_time_sec": 0.0,
        }

        # PCA Q is required for each *segment start* anchor. The final anchor is a boundary only and is not
        # used as a segment start (see forward(): accesses q_map[self.anchors[j]] for j in range(num_intervals)).
        required_anchors = self.anchors[:-1] if self.num_intervals > 0 else []
        missing = [a for a in required_anchors if a not in q_map]
        if missing:
            available = sorted(int(k) for k in q_map.keys())
            raise ValueError(
                "PCA Q missing for segment-start anchors: "
                f"{missing}. Available Q layers: {available}. "
                "Fix by pointing --pca_q_dir to a compatible metric_pca_segment_q output (or rebuild it)."
            )
        dev = next(self.student.parameters()).device
        self.q_map: Dict[int, torch.Tensor] = {
            int(a): q_map[int(a)].to(device=dev, dtype=torch.float32) for a in required_anchors
        }
        eye_r = torch.eye(int(self.rank), device=dev, dtype=torch.float32)
        if seg_L is None:
            seg_L = [eye_r for _ in range(self.num_intervals)]
        if seg_Linv is None:
            seg_Linv = [eye_r for _ in range(self.num_intervals)]
        if len(seg_L) != self.num_intervals or len(seg_Linv) != self.num_intervals:
            raise ValueError("seg_L/seg_Linv must match num_intervals.")
        self.seg_L = [x.to(device=dev, dtype=torch.float32) for x in seg_L]
        self.seg_Linv = [x.to(device=dev, dtype=torch.float32) for x in seg_Linv]
        if segment_gate_logits is not None:
            seg_gate = segment_gate_logits.detach().to(device=dev, dtype=torch.float32).view(-1)
            if int(seg_gate.numel()) != int(self.num_intervals):
                raise ValueError(
                    f"segment_gate_logits size={int(seg_gate.numel())} "
                    f"must match num_intervals={int(self.num_intervals)}"
                )
            self.segment_gate_logits = nn.Parameter(seg_gate.clone(), requires_grad=False)
            self.segment_gate_enabled = bool(segment_gate_enabled)
        else:
            # Default gate logits are near-identity scaling (sigmoid(12) ~= 1).
            self.segment_gate_logits = nn.Parameter(
                torch.full((self.num_intervals,), 12.0, device=dev, dtype=torch.float32),
                requires_grad=False,
            )
            self.segment_gate_enabled = bool(segment_gate_enabled)
        if seg_chart_risk is None:
            seg_chart_risk = [1.0 for _ in range(self.num_intervals)]
        if len(seg_chart_risk) != self.num_intervals:
            raise ValueError("seg_chart_risk must match num_intervals.")
        self.seg_chart_risk = [max(1e-6, float(x)) for x in seg_chart_risk]
        self._hook_ctx: Optional[Dict[str, Any]] = None
        self._hf_hook_handles: List[Any] = []
        self._use_hf_forward_hooks = False
        self._hf_hook_expected = int(self.num_intervals)
        self._hf_hook_reason = "not_attempted"
        try:
            layers = getattr(getattr(self.student, "model", None), "layers", None)
            if not bool(self.enable_hf_hook):
                self._hf_hook_reason = "disabled_by_flag"
            elif self.num_intervals <= 0:
                self._hf_hook_reason = "no_intervals"
            elif layers is None:
                self._hf_hook_reason = "missing_student_model_layers"
            else:
                try:
                    layer_count = int(len(layers))
                except Exception:
                    layer_count = -1
                if layer_count <= 0:
                    self._hf_hook_reason = f"invalid_layers_container:{type(layers).__name__}"
                else:
                    max_anchor = max(int(a) for a in self.anchors[:-1])
                    if layer_count == int(len(self.anchors)):
                        hook_layer_indices = [int(seg_idx) for seg_idx in range(self.num_intervals)]
                        hook_mode = "anchor_aligned_layers"
                    elif max_anchor < layer_count:
                        hook_layer_indices = [int(self.anchors[seg_idx]) for seg_idx in range(self.num_intervals)]
                        hook_mode = "full_depth_layers"
                    else:
                        hook_layer_indices = []
                        hook_mode = "unresolved"
                        self._hf_hook_reason = f"anchor_oob:max_anchor={max_anchor},layers={layer_count}"

                    if hook_layer_indices:
                        for seg_idx, layer_idx in enumerate(hook_layer_indices):
                            layer = layers[layer_idx]
                            if not isinstance(layer, nn.Module):
                                raise TypeError(
                                    f"layers[{layer_idx}] is {type(layer).__name__}, expected nn.Module"
                                )
                            handle = layer.register_forward_hook(self._make_layer_injection_hook(seg_idx))
                            self._hf_hook_handles.append(handle)
                        self._use_hf_forward_hooks = len(self._hf_hook_handles) == int(self.num_intervals)
                        if self._use_hf_forward_hooks:
                            self._hf_hook_reason = (
                                f"enabled:{hook_mode}:layers={hook_layer_indices}"
                            )
                        else:
                            self._hf_hook_reason = (
                                f"partial_hooks:{len(self._hf_hook_handles)}/{int(self.num_intervals)}"
                            )
        except Exception as e:
            for handle in self._hf_hook_handles:
                try:
                    handle.remove()
                except Exception:
                    pass
            self._hf_hook_handles = []
            self._use_hf_forward_hooks = False
            self._hf_hook_reason = f"exception:{type(e).__name__}:{e}"

    @property
    def device(self):
        return next(self.student.parameters()).device

    def reset_profile(self):
        self._profile = {
            "forward_calls": 0,
            "predictor_steps": 0,
            "base_time_sec": 0.0,
            "predictor_time_sec": 0.0,
            "combined_time_sec": 0.0,
        }

    def get_profile(self) -> Dict[str, float]:
        return dict(self._profile)

    def _maybe_sync(self, device: torch.device):
        if bool(self.profile_sync_timing):
            _sync_if_cuda(device)

    def _pool_rank_state(self, z_state: torch.Tensor, attn_local: torch.Tensor) -> torch.Tensor:
        if z_state.dim() == 2:
            return z_state
        if self.use_last_token:
            return z_state.squeeze(1)
        m = attn_local.to(z_state.dtype).unsqueeze(-1)
        return (z_state * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def _direction_energy_proxy(self, delta_value: torch.Tensor, *, seg_local: int) -> torch.Tensor:
        flat = delta_value.reshape(delta_value.size(0), -1)
        e = (flat * flat).sum(dim=1) / float(max(1, int(self.rank)))
        r = float(self.seg_chart_risk[max(0, min(int(self.num_intervals) - 1, int(seg_local)))])
        return e / max(r, 1e-6)

    def _gate_features(
        self,
        z_main: torch.Tensor,
        z_cond: torch.Tensor,
        attn_local: torch.Tensor,
        energy_proxy: torch.Tensor,
    ) -> torch.Tensor:
        p_main = self._pool_rank_state(z_main, attn_local)
        p_cond = self._pool_rank_state(z_cond, attn_local)
        return torch.cat([p_main, p_cond, energy_proxy.view(-1, 1)], dim=-1)

    def _gate_scale_from_logits(self, gate_logit: torch.Tensor) -> torch.Tensor:
        if bool(self.gate_use_soft_scale):
            prob = torch.sigmoid(gate_logit / float(self.gate_scale_temp))
            return prob.clamp(min=float(self.gate_scale_floor), max=1.0)
        prob = torch.sigmoid(gate_logit)
        return (prob >= float(self.gate_threshold)).to(dtype=torch.float32)

    def _segment_gate_scale(self, seg_idx: int, *, dtype: torch.dtype) -> torch.Tensor:
        if self.segment_gate_fixed_value is not None:
            return torch.tensor(
                float(self.segment_gate_fixed_value),
                device=self.segment_gate_logits.device,
                dtype=dtype,
            )
        if not bool(self.segment_gate_enabled):
            return torch.ones((), device=self.segment_gate_logits.device, dtype=dtype)
        seg_idx = max(0, min(int(self.num_intervals) - 1, int(seg_idx)))
        return torch.sigmoid(self.segment_gate_logits[seg_idx]).to(dtype=dtype)

    def _maybe_log_injection_debug(
        self,
        *,
        path: str,
        seg_idx: int,
        h_before: torch.Tensor,
        h_after: torch.Tensor,
        delta_h: torch.Tensor,
        gate_scale: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        pred_idx: Optional[torch.Tensor] = None,
    ) -> None:
        if not bool(self.injection_debug):
            return
        if int(self._injection_debug_count) >= int(self.injection_debug_max_prints):
            return
        with torch.no_grad():
            delta_h_norm = float(delta_h.detach().to(torch.float32).norm().item())
            h_delta_norm = float((h_after.detach() - h_before.detach()).to(torch.float32).norm().item())
            gate_mean = float(gate_scale.detach().to(torch.float32).mean().item())
            msg = (
                f"[Predictor inject debug] path={path} seg={int(seg_idx)} "
                f"delta_h_norm={delta_h_norm:.6e} gate={gate_mean:.6f} "
                f"h_delta_norm={h_delta_norm:.6e}"
            )
            if token_mask is not None:
                masked_norm = float((token_mask.detach() * delta_h.detach()).to(torch.float32).norm().item())
                nnz = int((token_mask.detach() > 0).sum().item())
                msg += f" masked_delta_h_norm={masked_norm:.6e} token_mask_nnz={nnz}"
            if pred_idx is not None and pred_idx.numel() > 0:
                pmin = int(pred_idx.min().item())
                pmax = int(pred_idx.max().item())
                msg += f" pred_idx_min={pmin} pred_idx_max={pmax}"
            print(msg, flush=True)
            self._injection_debug_count += 1

    def _apply_segment_update(self, h: torch.Tensor, attention_mask: torch.Tensor, seg_idx: int) -> torch.Tensor:
        q = self.q_map[self.anchors[seg_idx]]
        if self.use_last_token:
            h_sel, _ = _select_last_pred_token(h, attention_mask, append_eos=self.append_eos)
            z_state = torch.matmul(h_sel.to(torch.float32), q)
        else:
            z_state = torch.matmul(h.to(torch.float32), q)

        delta_u_pred = _predict_delta_u(
            self.predictors[seg_idx],
            z_state,
            predictor_eval_mode=self.predictor_eval_mode,
            ode_steps=self.ode_steps,
            z0_mode=self.pred_z0_mode,
            z0_std=self.pred_z0_std,
        )
        delta_u_total = delta_u_pred
        if self.guiders is not None:
            t_one = torch.ones((h.size(0), 1, 1), device=h.device, dtype=torch.float32)
            delta_hat = self.guiders[seg_idx](delta_u_pred, t_one, z_cond=z_state)
            if self.enable_gating and (self.gate_heads is not None):
                energy_hat = self._direction_energy_proxy(delta_hat, seg_local=seg_idx)
                g_feat = self._gate_features(delta_u_pred, z_state, attention_mask, energy_hat)
                gate_logit = self.gate_heads[seg_idx](g_feat).squeeze(-1)
                gate_scale = self._gate_scale_from_logits(gate_logit)
                delta_hat = delta_hat * gate_scale.view(-1, 1, 1)
            delta_u_total = delta_u_total + delta_hat

        delta_z_total = _apply_rank_linear(delta_u_total, self.seg_L[seg_idx])
        if self.use_mu_delta and self.mu_delta is not None:
            mu_j = self.mu_delta[seg_idx].to(device=h.device, dtype=torch.float32).view(1, 1, -1)
            delta_z_total = delta_z_total + mu_j

        delta_h = torch.matmul(delta_z_total, q.t()).to(dtype=h.dtype)
        gate_scale = self._segment_gate_scale(seg_idx, dtype=delta_h.dtype)
        delta_h = delta_h * gate_scale
        delta_h = _clip_delta_h_per_token(delta_h, self.delta_h_clip_norm)

        if self.use_last_token:
            pred_idx = _last_pred_index(attention_mask, append_eos=self.append_eos)
            token_mask = _build_last_pred_injection_mask(
                attention_mask,
                pred_idx,
                window_radius=int(self.injection_window),
                dtype=h.dtype,
                device=h.device,
            )
            h_after = h + token_mask * delta_h
            self._maybe_log_injection_debug(
                path="hook",
                seg_idx=seg_idx,
                h_before=h,
                h_after=h_after,
                delta_h=delta_h,
                gate_scale=gate_scale,
                token_mask=token_mask,
                pred_idx=pred_idx,
            )
            return h_after
        h_after = h + delta_h
        self._maybe_log_injection_debug(
            path="hook",
            seg_idx=seg_idx,
            h_before=h,
            h_after=h_after,
            delta_h=delta_h,
            gate_scale=gate_scale,
            token_mask=None,
            pred_idx=None,
        )
        return h_after

    def _make_layer_injection_hook(self, seg_idx: int):
        def _hook(_module, _inputs, output):
            ctx = self._hook_ctx
            if ctx is None:
                return output
            if isinstance(output, tuple):
                h = output[0]
            elif isinstance(output, list):
                h = output[0]
            else:
                h = output

            if bool(self.profile_sync_timing):
                self._maybe_sync(h.device)
                t_pred0 = time.perf_counter()
                h_new = self._apply_segment_update(h, ctx["attention_mask"], seg_idx)
                self._maybe_sync(h.device)
                ctx["predictor_time_local"] += max(0.0, time.perf_counter() - t_pred0)
            else:
                h_new = self._apply_segment_update(h, ctx["attention_mask"], seg_idx)
            ctx["predictor_steps_local"] += 1

            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            if isinstance(output, list):
                out_list = list(output)
                out_list[0] = h_new
                return out_list
            return h_new

        return _hook

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        use_cache: bool = False,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required")
        del use_cache
        del kwargs

        device = self.device
        self._maybe_sync(device)
        t_forward0 = time.perf_counter()
        predictor_time_local = 0.0
        predictor_steps_local = 0

        input_ids = input_ids.to(device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(device)

        if self._use_hf_forward_hooks:
            ctx_local = {
                "attention_mask": attention_mask,
                "predictor_time_local": 0.0,
                "predictor_steps_local": 0,
            }
            self._hook_ctx = ctx_local
            try:
                out = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
            finally:
                self._hook_ctx = None

            logits = out.logits if hasattr(out, "logits") else out[0]
            predictor_time_local = float(ctx_local.get("predictor_time_local", 0.0))
            predictor_steps_local = int(ctx_local.get("predictor_steps_local", 0))

            self._maybe_sync(device)
            forward_total = max(0.0, time.perf_counter() - t_forward0)
            base_time_local = max(0.0, forward_total - predictor_time_local)
            self._profile["forward_calls"] += 1
            self._profile["predictor_steps"] += predictor_steps_local
            self._profile["predictor_time_sec"] += predictor_time_local
            self._profile["base_time_sec"] += base_time_local
            self._profile["combined_time_sec"] += forward_total

            if return_dict:
                return _EvalOutput(logits)
            return (logits,)

        bsz, seqlen = input_ids.shape
        cache_position = torch.arange(seqlen, device=device, dtype=torch.long)
        pos_ids = cache_position.unsqueeze(0).expand(bsz, seqlen)
        h = self.student.model.embed_tokens(input_ids)
        causal_mask = _build_causal_mask(attention_mask, dtype=h.dtype)

        position_embeddings = None
        try:
            rotary = getattr(self.student.model, "rotary_emb", None)
            if rotary is not None:
                position_embeddings = rotary(h, pos_ids)
        except Exception:
            position_embeddings = None

        for j in range(self.num_intervals):
            h = _call_decoder_layer(
                self.student.model.layers[j],
                h,
                causal_mask,
                pos_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            q = self.q_map[self.anchors[j]]
            if self.use_last_token:
                h_sel, _ = _select_last_pred_token(h, attention_mask, append_eos=self.append_eos)
                z_state = torch.matmul(h_sel.to(torch.float32), q)
            else:
                z_state = torch.matmul(h.to(torch.float32), q)

            if bool(self.profile_sync_timing):
                self._maybe_sync(device)
                t_pred0 = time.perf_counter()
            delta_u_pred = _predict_delta_u(
                self.predictors[j],
                z_state,
                predictor_eval_mode=self.predictor_eval_mode,
                ode_steps=self.ode_steps,
                z0_mode=self.pred_z0_mode,
                z0_std=self.pred_z0_std,
            )
            delta_u_total = delta_u_pred
            if self.guiders is not None:
                t_one = torch.ones((bsz, 1, 1), device=device, dtype=torch.float32)
                delta_hat = self.guiders[j](delta_u_pred, t_one, z_cond=z_state)
                if self.enable_gating and (self.gate_heads is not None):
                    energy_hat = self._direction_energy_proxy(delta_hat, seg_local=j)
                    g_feat = self._gate_features(delta_u_pred, z_state, attention_mask, energy_hat)
                    gate_logit = self.gate_heads[j](g_feat).squeeze(-1)
                    gate_scale = self._gate_scale_from_logits(gate_logit)
                    delta_hat = delta_hat * gate_scale.view(-1, 1, 1)
                delta_u_total = delta_u_total + delta_hat

            delta_z_total = _apply_rank_linear(delta_u_total, self.seg_L[j])
            if self.use_mu_delta and self.mu_delta is not None:
                mu_j = self.mu_delta[j].to(device=device, dtype=torch.float32).view(1, 1, -1)
                delta_z_total = delta_z_total + mu_j

            delta_h = torch.matmul(delta_z_total, q.t()).to(dtype=h.dtype)
            gate_scale = self._segment_gate_scale(j, dtype=delta_h.dtype)
            delta_h = delta_h * gate_scale
            delta_h = _clip_delta_h_per_token(delta_h, self.delta_h_clip_norm)

            h_before = h
            if self.use_last_token:
                pred_idx = _last_pred_index(attention_mask, append_eos=self.append_eos)
                token_mask = _build_last_pred_injection_mask(
                    attention_mask,
                    pred_idx,
                    window_radius=int(self.injection_window),
                    dtype=h.dtype,
                    device=device,
                )
                h = h + token_mask * delta_h
                self._maybe_log_injection_debug(
                    path="manual",
                    seg_idx=j,
                    h_before=h_before,
                    h_after=h,
                    delta_h=delta_h,
                    gate_scale=gate_scale,
                    token_mask=token_mask,
                    pred_idx=pred_idx,
                )
            else:
                h = h + delta_h
                self._maybe_log_injection_debug(
                    path="manual",
                    seg_idx=j,
                    h_before=h_before,
                    h_after=h,
                    delta_h=delta_h,
                    gate_scale=gate_scale,
                    token_mask=None,
                    pred_idx=None,
                )
            if bool(self.profile_sync_timing):
                self._maybe_sync(device)
                predictor_time_local += max(0.0, time.perf_counter() - t_pred0)
            predictor_steps_local += 1

        h = _call_decoder_layer(
            self.student.model.layers[self.num_intervals],
            h,
            causal_mask,
            pos_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if hasattr(self.student.model, "norm"):
            h = self.student.model.norm(h)
        logits = self.student.lm_head(h)

        self._maybe_sync(device)
        forward_total = max(0.0, time.perf_counter() - t_forward0)
        base_time_local = max(0.0, forward_total - predictor_time_local)
        self._profile["forward_calls"] += 1
        self._profile["predictor_steps"] += predictor_steps_local
        self._profile["predictor_time_sec"] += predictor_time_local
        self._profile["base_time_sec"] += base_time_local
        self._profile["combined_time_sec"] += forward_total

        if return_dict:
            return _EvalOutput(logits)
        return (logits,)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 4,
        do_sample: bool = False,
        num_beams: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ):
        del kwargs
        del pad_token_id
        if do_sample:
            raise ValueError("StudentPredictorCausalLM only supports greedy decode.")
        if int(num_beams) != 1:
            raise ValueError("StudentPredictorCausalLM only supports num_beams=1.")

        device = self.device
        seq = input_ids.to(device)
        if attention_mask is None:
            attention_mask = torch.ones_like(seq, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)

        for _ in range(max(0, int(max_new_tokens))):
            out = self.forward(input_ids=seq, attention_mask=attention_mask, return_dict=True, use_cache=False)
            nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=device, dtype=attention_mask.dtype)], dim=1)
            if eos_token_id is not None and bool((nxt == int(eos_token_id)).all()):
                break
        return seq


def _infer_delta_mlp_hparams(state_dict: Dict[str, torch.Tensor], prefix: str = "0") -> Tuple[int, int, int, int]:
    rank_key = f"{prefix}.ln.weight"
    time_key = f"{prefix}.time_mlp.0.weight"
    if rank_key not in state_dict:
        raise ValueError(f"Cannot infer guider rank from state_dict key: {rank_key}")
    rank = int(state_dict[rank_key].numel())
    if time_key not in state_dict:
        raise ValueError(f"Cannot infer guider time_hidden from state_dict key: {time_key}")
    time_hidden = int(state_dict[time_key].shape[0])

    mlp_linear_keys: List[Tuple[int, str]] = []
    for key in state_dict.keys():
        parts = key.split(".")
        if len(parts) != 4:
            continue
        if parts[0] != str(prefix) or parts[1] != "mlp" or parts[3] != "weight":
            continue
        try:
            idx = int(parts[2])
        except Exception:
            continue
        mlp_linear_keys.append((idx, key))
    if not mlp_linear_keys:
        raise ValueError("Cannot infer guider mlp_depth: no mlp.*.weight keys found.")
    mlp_linear_keys.sort(key=lambda x: x[0])
    hidden_dim = int(state_dict[mlp_linear_keys[0][1]].shape[0])
    mlp_depth = max(1, len(mlp_linear_keys) - 1)
    mlp_mult = max(1, int(round(hidden_dim / max(1, rank))))
    return rank, mlp_mult, mlp_depth, time_hidden


def _load_segment_chart_risk(anchors_json: str, num_intervals: int) -> List[float]:
    out = [1.0 for _ in range(int(num_intervals))]
    try:
        with open(anchors_json, "r", encoding="utf-8") as f:
            anch = json.load(f)
        seg_reports = anch.get("segment_reports", None)
        if isinstance(seg_reports, list) and len(seg_reports) == int(num_intervals):
            for idx, rep in enumerate(seg_reports):
                try:
                    out[idx] = max(1e-6, float(rep.get("chart_risk_avg", 1.0)))
                except Exception:
                    out[idx] = 1.0
    except Exception:
        pass
    return out


def _build_guiders_from_ckpt(
    guidance_state: Dict[str, torch.Tensor],
    *,
    num_intervals: int,
    expected_rank: int,
) -> nn.ModuleList:
    inferred_rank, mlp_mult, mlp_depth, time_hidden = _infer_delta_mlp_hparams(guidance_state, prefix="0")
    if int(inferred_rank) != int(expected_rank):
        raise ValueError(
            f"guidance rank={inferred_rank} does not match predictor/Q rank={expected_rank}."
        )
    guiders = nn.ModuleList(
        [
            PredictorMLP(
                int(expected_rank),
                mlp_mult=int(mlp_mult),
                mlp_depth=int(mlp_depth),
                dropout=0.0,
                time_hidden=int(time_hidden),
            )
            for _ in range(int(num_intervals))
        ]
    )
    guiders.load_state_dict(guidance_state, strict=True)
    return guiders


def _build_gate_heads_from_ckpt(
    gate_state: Dict[str, torch.Tensor],
    *,
    num_intervals: int,
    expected_rank: int,
) -> nn.ModuleList:
    lin_key = "0.1.weight"
    if lin_key not in gate_state:
        raise ValueError("Invalid gate_heads state_dict: missing key 0.1.weight")
    gate_hid = int(gate_state[lin_key].shape[0])
    gate_in = int(gate_state[lin_key].shape[1])
    expected_gate_in = int(expected_rank) * 2 + 1
    if gate_in != expected_gate_in:
        print(
            f"[Warn] gate input dim={gate_in} differs from expected {expected_gate_in}; "
            "using checkpoint shape.",
            flush=True,
        )
    gate_heads = nn.ModuleList(
        [
            nn.Sequential(
                nn.LayerNorm(gate_in),
                nn.Linear(gate_in, gate_hid),
                nn.SiLU(),
                nn.Linear(gate_hid, 1),
            )
            for _ in range(int(num_intervals))
        ]
    )
    gate_heads.load_state_dict(gate_state, strict=True)
    return gate_heads


def load_student_predictor_model(args, *, guidance_ckpt: str = "", label: str = "student_predictor"):
    if not args.student_ckpt_dir:
        raise ValueError("--student_ckpt_dir is required")
    if not args.anchors_json:
        raise ValueError("--anchors_json is required")
    if not args.pca_q_dir:
        raise ValueError("--pca_q_dir is required")
    if not args.predictor_ckpt:
        raise ValueError("--predictor_ckpt is required")

    anchors = _read_anchors(args.anchors_json)
    q_map, q_rank = load_pca_q_dir_for_eval(args.pca_q_dir)
    num_intervals = len(anchors) - 1

    device = _resolve_eval_device(args.device)
    target_dtype = DTYPE_INFER if device.type == "cuda" else torch.float32
    student = AutoModelForCausalLM.from_pretrained(
        args.student_ckpt_dir,
        torch_dtype=target_dtype,
        device_map=None,
    ).to(device)
    student, student_ctrl_used, student_ctrl_msg = _apply_stateft_control_from_dir(
        student,
        args.student_ckpt_dir,
        enable=bool(args.student_use_control),
        label=label,
        missing_policy=str(getattr(args, "student_control_missing", "error")),
    )
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)

    s_layers = getattr(getattr(student, "model", None), "layers", None)
    if s_layers is None:
        raise ValueError("student_ckpt does not expose student.model.layers")
    if len(s_layers) != len(anchors):
        raise ValueError(f"Expected student layers={len(anchors)}, got {len(s_layers)}")

    ckpt = torch_load_compat(args.predictor_ckpt, map_location="cpu")
    pred_rank = int(ckpt.get("rank", q_rank))
    mlp_mult = int(ckpt.get("mlp_mult", 4))
    mlp_depth = int(ckpt.get("mlp_depth", 2))
    time_hidden = int(ckpt.get("time_hidden", 128))
    dropout = float(ckpt.get("dropout", 0.0))
    predictors = nn.ModuleList(
        [
            PredictorMLP(
                pred_rank,
                mlp_mult=mlp_mult,
                mlp_depth=mlp_depth,
                dropout=dropout,
                time_hidden=time_hidden,
            )
            for _ in range(num_intervals)
        ]
    )
    try:
        predictors.load_state_dict(ckpt["predictors"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"predictor_ckpt incompatible with eval predictor architecture: {e}") from e
    predictors.to(device)
    predictors.eval()
    for p in predictors.parameters():
        p.requires_grad_(False)

    mu_delta = ckpt.get("mu_delta", None)
    if isinstance(mu_delta, torch.Tensor):
        mu_delta = mu_delta.to(device=device, dtype=torch.float32)
    else:
        mu_delta = None
    segment_gate_logits = ckpt.get("segment_gate_logits", None)
    if isinstance(segment_gate_logits, torch.Tensor):
        segment_gate_logits = segment_gate_logits.to(device=device, dtype=torch.float32).view(-1)
        if int(segment_gate_logits.numel()) != int(num_intervals):
            raise ValueError(
                "predictor_ckpt segment_gate_logits shape mismatch: "
                f"expected num_intervals={int(num_intervals)}, got {int(segment_gate_logits.numel())}"
            )
    else:
        segment_gate_logits = None
    segment_gate_enabled = bool(ckpt.get("segment_gate_enabled", bool(segment_gate_logits is not None)))
    segment_gate_l1_lambda = float(ckpt.get("segment_gate_l1_lambda", 0.0))
    segment_gate_mean = (
        float(torch.sigmoid(segment_gate_logits).mean().item()) if isinstance(segment_gate_logits, torch.Tensor) else 1.0
    )

    use_ckpt_settings = bool(getattr(args, "student_use_ckpt_settings", True))
    resolved_ode_steps = int(args.student_ode_steps)
    resolved_use_last_token = bool(args.student_use_last_token)
    resolved_use_mu_delta = bool(args.student_use_mu_delta)
    resolved_append_eos = bool(args.student_append_eos)
    resolved_injection_window = int(getattr(args, "student_injection_window", 0))
    resolved_use_whiten = bool(ckpt.get("use_whiten", False))
    resolved_delta_h_clip_norm = float(max(0.0, float(getattr(args, "delta_h_clip_norm", 0.0))))
    resolved_enable_hf_hook = not bool(getattr(args, "disable_hf_hook", False))
    gate_eval_enabled_arg = getattr(args, "segment_gate_eval_enabled", None)
    gate_fixed_arg = getattr(args, "segment_gate_fixed_value", None)
    resolved_predictor_eval_mode = str(getattr(args, "student_predictor_eval_mode", "auto")).strip().lower()
    if resolved_predictor_eval_mode not in {"auto", "direct", "rollout"}:
        raise ValueError(
            f"Invalid --student_predictor_eval_mode={resolved_predictor_eval_mode!r} "
            "(expected auto|direct|rollout)"
        )
    if use_ckpt_settings:
        if "ode_steps" in ckpt:
            resolved_ode_steps = int(ckpt["ode_steps"])
        if "use_last_token" in ckpt:
            resolved_use_last_token = bool(ckpt["use_last_token"])
        if "use_mu_delta" in ckpt:
            resolved_use_mu_delta = bool(ckpt["use_mu_delta"])
        if "append_eos" in ckpt:
            resolved_append_eos = bool(ckpt["append_eos"])
        if "injection_window" in ckpt:
            resolved_injection_window = int(ckpt["injection_window"])
    if resolved_predictor_eval_mode == "auto":
        predictor_type = str(ckpt.get("predictor_type", "")).strip().lower()
        ckpt_ode_steps = int(ckpt.get("ode_steps", 1))
        if ("flow" in predictor_type) or ("ode" in predictor_type) or (ckpt_ode_steps > 1):
            resolved_predictor_eval_mode = "rollout"
        else:
            resolved_predictor_eval_mode = "direct"

    if resolved_predictor_eval_mode == "direct" and int(resolved_ode_steps) != 1:
        print(
            f"[Info] predictor_eval_mode=direct ignores ode_steps={int(resolved_ode_steps)} and uses 1 step.",
            flush=True,
        )
        resolved_ode_steps = 1

    resolved_segment_gate_enabled = bool(segment_gate_enabled)
    if gate_eval_enabled_arg is not None:
        resolved_segment_gate_enabled = bool(gate_eval_enabled_arg)
    resolved_segment_gate_fixed_value: Optional[float] = None
    if gate_fixed_arg is not None:
        resolved_segment_gate_fixed_value = float(gate_fixed_arg)
        if not (0.0 <= float(resolved_segment_gate_fixed_value) <= 1.0):
            raise ValueError(
                f"--segment_gate_fixed_value must be in [0,1], got {resolved_segment_gate_fixed_value}"
            )

    if isinstance(mu_delta, torch.Tensor):
        rm = mu_delta
        if rm.dim() == 2 and tuple(rm.shape) == (int(num_intervals), int(pred_rank)):
            pass
        elif (
            rm.dim() == 2
            and int(rm.size(0)) == int(num_intervals)
            and int(rm.size(1)) == int(q_map[anchors[0]].shape[0])
        ):
            mu_rank = torch.zeros((num_intervals, pred_rank), device=device, dtype=torch.float32)
            for j in range(num_intervals):
                anchor_start = anchors[j]
                q_j = q_map[anchor_start].to(device=device, dtype=torch.float32)
                mu_rank[j].copy_(torch.matmul(q_j.transpose(0, 1), rm[j]))
            mu_delta = mu_rank
            print("[Predictor cfg] projected legacy mu_delta from hidden-space to rank-space.", flush=True)
        else:
            print(
                "[Warn] ignore mu_delta due to incompatible shape "
                f"{tuple(rm.shape)} (expected ({num_intervals}, {pred_rank}) or legacy ({num_intervals}, d_model)).",
                flush=True,
            )
            mu_delta = None

    cfg_src = "resolved from ckpt" if use_ckpt_settings else "using CLI args"
    print(
        f"[Predictor cfg] {cfg_src}: "
        f"predictor_mode={resolved_predictor_eval_mode}, "
        f"ode_steps={resolved_ode_steps}, "
        f"use_last_token={resolved_use_last_token}, "
        f"use_mu_delta={resolved_use_mu_delta}, "
        f"append_eos={resolved_append_eos}, "
        f"injection_window={resolved_injection_window}, "
        f"use_whiten={resolved_use_whiten}, "
        f"segment_gate_enabled={resolved_segment_gate_enabled}, "
        f"segment_gate_fixed={resolved_segment_gate_fixed_value}, "
        f"segment_gate_mean={segment_gate_mean:.4f}, "
        f"delta_h_clip_norm={resolved_delta_h_clip_norm:.4f}, "
        f"hf_hook_enabled={resolved_enable_hf_hook}",
        flush=True,
    )
    if "append_eos" not in ckpt and use_ckpt_settings:
        print(
            "[Warn] predictor_ckpt has no append_eos metadata; ensure --student_append_eos matches phase2 training.",
            flush=True,
        )
    if bool(resolved_use_mu_delta) and (mu_delta is None):
        print(
            "[Warn] use_mu_delta=True but no valid rank-space mu_delta was loaded; eval will run without mu compensation.",
            flush=True,
        )
    print(
        f"[Predictor runtime] profile_sync_timing={bool(getattr(args, 'student_profile_sync_timing', False))}",
        flush=True,
    )

    seg_L: List[torch.Tensor] = [torch.eye(pred_rank, device=device, dtype=torch.float32) for _ in range(num_intervals)]
    seg_Linv: List[torch.Tensor] = [torch.eye(pred_rank, device=device, dtype=torch.float32) for _ in range(num_intervals)]
    if bool(resolved_use_whiten):
        L_seg_meta = ckpt.get("whiten_L_segments", None)
        Linv_seg_meta = ckpt.get("whiten_Linv_segments", None)
        if not (torch.is_tensor(L_seg_meta) and torch.is_tensor(Linv_seg_meta)):
            raise ValueError(
                "predictor_ckpt indicates use_whiten=True but has no whiten_L_segments/whiten_Linv_segments. "
                "Please use a Phase-2 checkpoint saved with whiten segment factors."
            )
        if (
            L_seg_meta.dim() != 3
            or Linv_seg_meta.dim() != 3
            or tuple(L_seg_meta.shape) != tuple(Linv_seg_meta.shape)
            or int(L_seg_meta.size(0)) != int(num_intervals)
            or int(L_seg_meta.size(-1)) != int(pred_rank)
        ):
            raise ValueError(
                "Invalid predictor_ckpt whiten segment factors shape; "
                f"expected [{num_intervals},{pred_rank},{pred_rank}]"
            )
        seg_L = [L_seg_meta[j].to(device=device, dtype=torch.float32) for j in range(num_intervals)]
        seg_Linv = [Linv_seg_meta[j].to(device=device, dtype=torch.float32) for j in range(num_intervals)]

    guiders = None
    gate_heads = None
    enable_gating = False
    gate_threshold = 0.5
    gate_use_soft_scale = True
    gate_scale_floor = 0.0
    gate_scale_temp = 1.0
    if guidance_ckpt:
        if not os.path.exists(guidance_ckpt):
            raise FileNotFoundError(f"guidance_ckpt not found: {guidance_ckpt}")
        gckpt = torch_load_compat(guidance_ckpt, map_location="cpu")
        if use_ckpt_settings:
            if "use_last_token" in gckpt:
                resolved_use_last_token = bool(gckpt["use_last_token"])
            if "append_eos" in gckpt:
                resolved_append_eos = bool(gckpt["append_eos"])
            if "injection_window" in gckpt:
                resolved_injection_window = int(gckpt["injection_window"])
        pred_ref = str(gckpt.get("predictor_ckpt", "") or "").strip()
        if pred_ref and os.path.abspath(pred_ref) != os.path.abspath(str(args.predictor_ckpt)):
            print(
                f"[Warn] guidance_ckpt was trained with predictor_ckpt={pred_ref}, "
                f"but eval uses predictor_ckpt={args.predictor_ckpt}",
                flush=True,
            )
        g_state = gckpt.get("guiders", None)
        if not isinstance(g_state, dict):
            raise ValueError("guidance_ckpt is missing guiders state_dict.")
        guiders = _build_guiders_from_ckpt(
            g_state,
            num_intervals=num_intervals,
            expected_rank=pred_rank,
        )
        guiders.to(device)
        guiders.eval()
        for p in guiders.parameters():
            p.requires_grad_(False)

        enable_gating = bool(gckpt.get("enable_gating", False))
        gate_state = gckpt.get("gate_heads", None)
        if enable_gating and isinstance(gate_state, dict):
            gate_heads = _build_gate_heads_from_ckpt(
                gate_state,
                num_intervals=num_intervals,
                expected_rank=pred_rank,
            )
            gate_heads.to(device)
            gate_heads.eval()
            for p in gate_heads.parameters():
                p.requires_grad_(False)
        else:
            gate_heads = None
            enable_gating = False

        gate_threshold = float(gckpt.get("gate_threshold", 0.5))
        gate_use_soft_scale = bool(gckpt.get("gate_use_soft_scale", True))
        gate_scale_floor = float(gckpt.get("gate_scale_floor", 0.0))
        gate_scale_temp = float(gckpt.get("gate_scale_temp", 1.0))
        print(
            "[Guidance cfg] "
            f"enabled={guiders is not None}, gating={enable_gating}, "
            f"use_last_token={resolved_use_last_token}, append_eos={resolved_append_eos}, "
            f"injection_window={resolved_injection_window}",
            flush=True,
        )

    seg_chart_risk = _load_segment_chart_risk(args.anchors_json, num_intervals)
    model = StudentPredictorCausalLM(
        student=student,
        anchors=anchors,
        q_map=q_map,
        predictors=predictors,
        mu_delta=mu_delta,
        use_last_token=bool(resolved_use_last_token),
        append_eos=bool(resolved_append_eos),
        use_mu_delta=bool(resolved_use_mu_delta),
        ode_steps=int(resolved_ode_steps),
        pred_z0_mode=str(args.student_pred_z0_mode),
        pred_z0_std=float(args.student_pred_z0_std),
        predictor_eval_mode=str(resolved_predictor_eval_mode),
        injection_window=int(resolved_injection_window),
        use_whiten=bool(resolved_use_whiten),
        seg_L=seg_L,
        seg_Linv=seg_Linv,
        segment_gate_logits=segment_gate_logits,
        segment_gate_enabled=bool(resolved_segment_gate_enabled),
        guiders=guiders,
        gate_heads=gate_heads,
        enable_gating=bool(enable_gating),
        gate_threshold=float(gate_threshold),
        gate_use_soft_scale=bool(gate_use_soft_scale),
        gate_scale_floor=float(gate_scale_floor),
        gate_scale_temp=float(gate_scale_temp),
        seg_chart_risk=seg_chart_risk,
        profile_sync_timing=bool(getattr(args, "student_profile_sync_timing", False)),
        injection_debug=bool(getattr(args, "predictor_injection_debug", False)),
        injection_debug_max_prints=int(getattr(args, "predictor_injection_debug_max_prints", 1)),
        delta_h_clip_norm=float(resolved_delta_h_clip_norm),
        segment_gate_fixed_value=resolved_segment_gate_fixed_value,
        enable_hf_hook=bool(resolved_enable_hf_hook),
    )
    model.eval()
    hf_hook_enabled = bool(getattr(model, "_use_hf_forward_hooks", False))
    hf_hook_count = int(len(getattr(model, "_hf_hook_handles", [])))
    hf_hook_expected = int(getattr(model, "_hf_hook_expected", 0))
    hf_hook_reason = str(getattr(model, "_hf_hook_reason", "unknown"))
    print(
        f"[HF hook] enabled={hf_hook_enabled} hooks={hf_hook_count}/{hf_hook_expected} reason={hf_hook_reason}",
        flush=True,
    )
    meta = {
        "guidance": bool(guiders is not None),
        "enable_gating": bool(enable_gating),
        "injection_window": int(resolved_injection_window),
        "predictor_eval_mode": str(resolved_predictor_eval_mode),
        "append_eos": bool(resolved_append_eos),
        "use_whiten": bool(resolved_use_whiten),
        "segment_gate_enabled": bool(resolved_segment_gate_enabled),
        "segment_gate_fixed_value": (
            None if resolved_segment_gate_fixed_value is None else float(resolved_segment_gate_fixed_value)
        ),
        "segment_gate_mean": float(segment_gate_mean),
        "segment_gate_l1_lambda": float(segment_gate_l1_lambda),
        "delta_h_clip_norm": float(resolved_delta_h_clip_norm),
        "hf_hook_forced_disable": bool(not resolved_enable_hf_hook),
        "hf_hook_enabled": hf_hook_enabled,
        "hf_hook_count": hf_hook_count,
        "hf_hook_expected": hf_hook_expected,
        "hf_hook_reason": hf_hook_reason,
    }
    return model, pred_rank, student_ctrl_used, student_ctrl_msg, meta


def load_student_base_model(args):
    if not args.student_ckpt_dir:
        raise ValueError("--student_ckpt_dir is required")
    device = _resolve_eval_device(args.device)
    target_dtype = DTYPE_INFER if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.student_ckpt_dir,
        torch_dtype=target_dtype,
        device_map=None,
    ).to(device)
    model, student_ctrl_used, student_ctrl_msg = _apply_stateft_control_from_dir(
        model,
        args.student_ckpt_dir,
        enable=bool(args.student_use_control),
        label="student_base",
        missing_policy=str(getattr(args, "student_control_missing", "error")),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, student_ctrl_used, student_ctrl_msg


def run_pipeline_eval(args):
    if (
        not bool(args.run_stateft)
        and not bool(args.run_student_base)
        and not bool(args.run_student_predictor)
        and not bool(args.run_student_guidance)
    ):
        raise ValueError(
            "At least one of --run_stateft/--run_student_base/--run_student_predictor/--run_student_guidance must be True."
        )

    summary_fp = args.results_csv
    _ensure_parent_dir(summary_fp)

    columns = [
        "variant",
        "dataset",
        "accuracy",
        "throughput_toks_per_s",
        "avg_latency_ms",
        "ok_samples",
        "skipped",
        "forward_tokens",
        "elapsed_sec",
        "student_base_time_sec",
        "student_predictor_time_sec",
        "student_combined_time_sec",
        "student_predictor_steps",
        "method",
        "detail",
        "first_error",
    ]
    rows: List[Dict[str, Any]] = []

    def _append(
        variant: str,
        dataset: str,
        accuracy,
        throughput,
        latency,
        ok_samples,
        skipped,
        forward_tokens,
        elapsed_sec,
        student_base_time_sec,
        student_predictor_time_sec,
        student_combined_time_sec,
        student_predictor_steps,
        method: str,
        detail: str,
        first_error: Optional[str],
    ):
        rows.append(
            {
                "variant": variant,
                "dataset": dataset,
                "accuracy": accuracy,
                "throughput_toks_per_s": throughput,
                "avg_latency_ms": latency,
                "ok_samples": ok_samples,
                "skipped": skipped,
                "forward_tokens": forward_tokens,
                "elapsed_sec": elapsed_sec,
                "student_base_time_sec": student_base_time_sec,
                "student_predictor_time_sec": student_predictor_time_sec,
                "student_combined_time_sec": student_combined_time_sec,
                "student_predictor_steps": student_predictor_steps,
                "method": method,
                "detail": detail,
                "first_error": first_error or "",
            }
        )

    if bool(args.run_stateft):
        teacher_loader = str(getattr(args, "_resolved_teacher_loader", getattr(args, "teacher_loader", "auto")))
        print(f"[PIPELINE] loading teacher: {args.teacher_ckpt} (loader={teacher_loader})")
        model = load_stateft_teacher_model(args)
        tokenizer = _load_tokenizer(args.base_model)
        for ds in args.datasets:
            (
                acc,
                ok_cnt,
                skip_cnt,
                first_err,
                tps,
                lat_ms,
                fwd_toks,
                elapsed,
                base_time_sec,
                predictor_time_sec,
                combined_time_sec,
                predictor_steps,
            ) = evaluate_model_on_dataset(
                args, model, tokenizer, ds, "STATEFT"
            )
            acc_s = "ERROR" if acc is None else f"{acc:.4f}"
            tps_s = "NA" if tps is None else f"{tps:.2f}"
            print(f"[RESULT] teacher_model|{ds} acc={acc_s} throughput={tps_s} toks/s")
            _append(
                "teacher_model",
                ds,
                acc,
                tps,
                lat_ms,
                ok_cnt,
                skip_cnt,
                fwd_toks,
                elapsed,
                base_time_sec,
                predictor_time_sec,
                combined_time_sec,
                predictor_steps,
                f"teacher_{teacher_loader}",
                f"teacher_ckpt={args.teacher_ckpt},loader={teacher_loader}",
                first_err,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if bool(args.run_student_base):
        print(f"[PIPELINE] loading Student base (no predictor): {args.student_ckpt_dir}")
        model, student_ctrl_used, student_ctrl_msg = load_student_base_model(args)
        tokenizer = _load_tokenizer(args.student_ckpt_dir or args.base_model)
        for ds in args.datasets:
            (
                acc,
                ok_cnt,
                skip_cnt,
                first_err,
                tps,
                lat_ms,
                fwd_toks,
                elapsed,
                base_time_sec,
                predictor_time_sec,
                combined_time_sec,
                predictor_steps,
            ) = evaluate_model_on_dataset(
                args, model, tokenizer, ds, "STUDENT_BASE"
            )
            acc_s = "ERROR" if acc is None else f"{acc:.4f}"
            tps_s = "NA" if tps is None else f"{tps:.2f}"
            print(f"[RESULT] student_base|{ds} acc={acc_s} throughput={tps_s} toks/s")
            _append(
                "student_base",
                ds,
                acc,
                tps,
                lat_ms,
                ok_cnt,
                skip_cnt,
                fwd_toks,
                elapsed,
                base_time_sec,
                predictor_time_sec,
                combined_time_sec,
                predictor_steps,
                "student_base",
                f"student_ckpt={args.student_ckpt_dir};control={student_ctrl_used};control_msg={student_ctrl_msg}",
                first_err,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if bool(args.run_student_predictor):
        print(f"[PIPELINE] loading Student+Predictor: {args.student_ckpt_dir} + {args.predictor_ckpt}")
        model, pred_rank, student_ctrl_used, student_ctrl_msg, model_meta = load_student_predictor_model(
            args,
            guidance_ckpt="",
            label="student_predictor",
        )
        tokenizer = _load_tokenizer(args.student_ckpt_dir or args.base_model)
        detail = (
            f"predictor_rank={pred_rank};ode_steps={int(getattr(model, 'ode_steps', args.student_ode_steps))};"
            f"predictor_mode={str(getattr(model, 'predictor_eval_mode', getattr(args, 'student_predictor_eval_mode', 'auto')))};"
            f"use_last_token={bool(getattr(model, 'use_last_token', args.student_use_last_token))};"
            f"append_eos={bool(model_meta.get('append_eos', args.student_append_eos))};"
            f"use_mu_delta={bool(getattr(model, 'use_mu_delta', args.student_use_mu_delta))};"
            f"use_whiten={bool(model_meta.get('use_whiten', False))};"
            f"segment_gate_enabled={bool(model_meta.get('segment_gate_enabled', False))};"
            f"segment_gate_fixed={model_meta.get('segment_gate_fixed_value', None)};"
            f"segment_gate_mean={float(model_meta.get('segment_gate_mean', 1.0)):.4f};"
            f"delta_h_clip_norm={float(model_meta.get('delta_h_clip_norm', 0.0)):.4f};"
            f"injection_window={int(model_meta.get('injection_window', 0))};"
            f"logprob_fast={bool(args.predictor_fast_logprob)};"
            f"logprob_ar_rescore_topk={int(args.predictor_ar_rescore_topk)};"
            f"force_ar_logprob={bool(getattr(args, 'force_ar_logprob', False))};"
            f"logprob_len_norm={str(args.length_norm)};"
            f"profile_sync_timing={bool(getattr(args, 'student_profile_sync_timing', False))};"
            f"hf_hook={bool(model_meta.get('hf_hook_enabled', False))};"
            f"hf_hooks={int(model_meta.get('hf_hook_count', 0))}/{int(model_meta.get('hf_hook_expected', 0))};"
            f"hf_hook_reason={str(model_meta.get('hf_hook_reason', 'unknown'))};"
            f"z0={str(args.student_pred_z0_mode)};"
            f"control={student_ctrl_used};control_msg={student_ctrl_msg}"
        )
        for ds in args.datasets:
            (
                acc,
                ok_cnt,
                skip_cnt,
                first_err,
                tps,
                lat_ms,
                fwd_toks,
                elapsed,
                base_time_sec,
                predictor_time_sec,
                combined_time_sec,
                predictor_steps,
            ) = evaluate_model_on_dataset(
                args, model, tokenizer, ds, "STUDENT"
            )
            acc_s = "ERROR" if acc is None else f"{acc:.4f}"
            tps_s = "NA" if tps is None else f"{tps:.2f}"
            base_s = "NA" if base_time_sec is None else f"{base_time_sec:.3f}"
            pred_s = "NA" if predictor_time_sec is None else f"{predictor_time_sec:.3f}"
            comb_s = "NA" if combined_time_sec is None else f"{combined_time_sec:.3f}"
            step_s = "NA" if predictor_steps is None else str(predictor_steps)
            print(
                f"[RESULT] student_predictor|{ds} acc={acc_s} throughput={tps_s} toks/s "
                f"| base={base_s}s predictor={pred_s}s combined={comb_s}s predictor_steps={step_s}"
            )
            if predictor_steps is not None and int(predictor_steps) <= 0:
                print(f"[WARN] student_predictor|{ds} predictor_steps={predictor_steps} (unexpected)")
            _append(
                "student_predictor",
                ds,
                acc,
                tps,
                lat_ms,
                ok_cnt,
                skip_cnt,
                fwd_toks,
                elapsed,
                base_time_sec,
                predictor_time_sec,
                combined_time_sec,
                predictor_steps,
                "student_predictor",
                detail,
                first_err,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if bool(args.run_student_guidance):
        if not str(args.guidance_ckpt or "").strip():
            raise ValueError("--guidance_ckpt is required when --run_student_guidance=True")
        print(
            f"[PIPELINE] loading Student+Predictor+Guidance: {args.student_ckpt_dir} + "
            f"{args.predictor_ckpt} + {args.guidance_ckpt}"
        )
        model, pred_rank, student_ctrl_used, student_ctrl_msg, model_meta = load_student_predictor_model(
            args,
            guidance_ckpt=str(args.guidance_ckpt),
            label="student_guidance",
        )
        tokenizer = _load_tokenizer(args.student_ckpt_dir or args.base_model)
        detail = (
            f"predictor_rank={pred_rank};ode_steps={int(getattr(model, 'ode_steps', args.student_ode_steps))};"
            f"predictor_mode={str(getattr(model, 'predictor_eval_mode', getattr(args, 'student_predictor_eval_mode', 'auto')))};"
            f"use_last_token={bool(getattr(model, 'use_last_token', args.student_use_last_token))};"
            f"append_eos={bool(model_meta.get('append_eos', args.student_append_eos))};"
            f"use_mu_delta={bool(getattr(model, 'use_mu_delta', args.student_use_mu_delta))};"
            f"use_whiten={bool(model_meta.get('use_whiten', False))};"
            f"segment_gate_enabled={bool(model_meta.get('segment_gate_enabled', False))};"
            f"segment_gate_fixed={model_meta.get('segment_gate_fixed_value', None)};"
            f"segment_gate_mean={float(model_meta.get('segment_gate_mean', 1.0)):.4f};"
            f"delta_h_clip_norm={float(model_meta.get('delta_h_clip_norm', 0.0)):.4f};"
            f"injection_window={int(model_meta.get('injection_window', 0))};"
            f"guidance={bool(model_meta.get('guidance', False))};"
            f"gating={bool(model_meta.get('enable_gating', False))};"
            f"logprob_fast={bool(args.predictor_fast_logprob)};"
            f"logprob_ar_rescore_topk={int(args.predictor_ar_rescore_topk)};"
            f"force_ar_logprob={bool(getattr(args, 'force_ar_logprob', False))};"
            f"logprob_len_norm={str(args.length_norm)};"
            f"profile_sync_timing={bool(getattr(args, 'student_profile_sync_timing', False))};"
            f"hf_hook={bool(model_meta.get('hf_hook_enabled', False))};"
            f"hf_hooks={int(model_meta.get('hf_hook_count', 0))}/{int(model_meta.get('hf_hook_expected', 0))};"
            f"hf_hook_reason={str(model_meta.get('hf_hook_reason', 'unknown'))};"
            f"z0={str(args.student_pred_z0_mode)};"
            f"control={student_ctrl_used};control_msg={student_ctrl_msg};"
            f"guidance_ckpt={args.guidance_ckpt}"
        )
        for ds in args.datasets:
            (
                acc,
                ok_cnt,
                skip_cnt,
                first_err,
                tps,
                lat_ms,
                fwd_toks,
                elapsed,
                base_time_sec,
                predictor_time_sec,
                combined_time_sec,
                predictor_steps,
            ) = evaluate_model_on_dataset(
                args, model, tokenizer, ds, "STUDENT_GUIDANCE"
            )
            acc_s = "ERROR" if acc is None else f"{acc:.4f}"
            tps_s = "NA" if tps is None else f"{tps:.2f}"
            base_s = "NA" if base_time_sec is None else f"{base_time_sec:.3f}"
            pred_s = "NA" if predictor_time_sec is None else f"{predictor_time_sec:.3f}"
            comb_s = "NA" if combined_time_sec is None else f"{combined_time_sec:.3f}"
            step_s = "NA" if predictor_steps is None else str(predictor_steps)
            print(
                f"[RESULT] student_guidance|{ds} acc={acc_s} throughput={tps_s} toks/s "
                f"| base={base_s}s predictor={pred_s}s combined={comb_s}s predictor_steps={step_s}"
            )
            if predictor_steps is not None and int(predictor_steps) <= 0:
                print(f"[WARN] student_guidance|{ds} predictor_steps={predictor_steps} (unexpected)")
            _append(
                "student_guidance",
                ds,
                acc,
                tps,
                lat_ms,
                ok_cnt,
                skip_cnt,
                fwd_toks,
                elapsed,
                base_time_sec,
                predictor_time_sec,
                combined_time_sec,
                predictor_steps,
                "student_guidance",
                detail,
                first_err,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(summary_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in rows:
            w.writerow([row[k] if row[k] is not None else "ERROR" for k in columns])

    print(f"\n[Done] summary -> {summary_fp}")
    variants = sorted({r["variant"] for r in rows})
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant and r["accuracy"] is not None]
        if not subset:
            print(f"[SUMMARY] {variant}: no valid samples")
            continue
        mean_acc = sum(float(r["accuracy"]) for r in subset) / len(subset)
        tps_vals = [float(r["throughput_toks_per_s"]) for r in subset if r["throughput_toks_per_s"] is not None]
        mean_tps = sum(tps_vals) / len(tps_vals) if tps_vals else float("nan")
        print(f"[SUMMARY] {variant}: mean_acc={mean_acc:.4f}, mean_throughput={mean_tps:.2f} toks/s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_profile", choices=["pipeline", "kd_ckd"], default="pipeline")
    p.add_argument("--mode", choices=["pipeline"], default="pipeline")
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--teacher_ckpt", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--teacher_loader", type=str, default="auto", choices=["auto", "stateft", "native"])
    p.add_argument("--test_data_path", type=str, default="/mlsteam/data/itri/stateft/new_idea/datasets")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["boolq", "piqa", "social_i_qa", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa", "hellaswag"],
        choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa", "csqa"],
    )
    p.add_argument("--results_csv", type=str, default="results/summary_pipeline.csv")

    p.add_argument("--run_stateft", type=_str2bool, default=True)
    p.add_argument("--run_student_base", type=_str2bool, default=True)
    p.add_argument("--run_student_predictor", type=_str2bool, default=True)
    p.add_argument("--run_student_guidance", type=_str2bool, default=False)
    p.add_argument("--pca_q_dir", type=str, default="/mlsteam/data/itri/stateft/new_idea/out/metric_pca_block_q_20260207_165946")
    p.add_argument("--anchors_json", type=str, default="/mlsteam/data/itri/stateft/new_idea/out/phase1_5_exp_p15_joint/anchors.json")
    p.add_argument("--student_ckpt_dir", type=str, default="/mlsteam/data/itri/stateft/new_idea/out/phase1_5_exp_p15_joint/student")
    p.add_argument("--predictor_ckpt", type=str, default="/mlsteam/data/itri/stateft/new_idea/out/phase2_predictor_20260207_203107/predictor_final.pt")
    p.add_argument("--guidance_ckpt", type=str, default="")
    p.add_argument("--student_ode_steps", type=int, default=1)
    p.add_argument("--student_injection_window", type=int, default=0)
    p.add_argument("--student_use_last_token", type=_str2bool, default=True)
    p.add_argument("--student_append_eos", type=_str2bool, default=True)
    p.add_argument("--student_use_mu_delta", type=_str2bool, default=False)
    # Thesis mainline usually evaluates a plain HF student copy (no control_config/control_state files).
    p.add_argument("--student_use_control", type=_str2bool, default=False)
    p.add_argument("--student_control_missing", choices=["error", "disable"], default="disable")
    p.add_argument("--student_use_ckpt_settings", type=_str2bool, default=True)
    p.add_argument("--student_predictor_eval_mode", choices=["auto", "direct", "rollout"], default="auto")
    p.add_argument("--student_pred_z0_mode", choices=["random", "zero"], default="zero")
    p.add_argument("--student_pred_z0_std", type=float, default=1.0)
    p.add_argument(
        "--student_profile_sync_timing",
        type=_str2bool,
        default=False,
        help="If True, synchronize CUDA around predictor timing points (more accurate timing, slower inference).",
    )
    p.add_argument(
        "--predictor_injection_debug",
        type=_str2bool,
        default=False,
        help="Print one-shot predictor injection norms (delta_h, gate, h_after-h_before) for fast diagnosis.",
    )
    p.add_argument(
        "--predictor_injection_debug_max_prints",
        type=int,
        default=1,
        help="Maximum number of predictor injection debug lines to print.",
    )
    p.add_argument(
        "--delta_h_clip_norm",
        type=float,
        default=0.0,
        help="Inference safety clip on per-token ||delta_h|| (0 disables).",
    )
    p.add_argument(
        "--segment_gate_eval_enabled",
        type=_str2bool,
        default=None,
        help="Override checkpoint segment gate enabled state during eval.",
    )
    p.add_argument(
        "--segment_gate_fixed_value",
        type=float,
        default=None,
        help="Force a fixed gate scale in [0,1] during eval (None keeps checkpoint/runtime gate).",
    )
    p.add_argument(
        "--disable_hf_hook",
        type=_str2bool,
        default=False,
        help="Disable HF forward hooks and use manual segment injection path.",
    )
    p.add_argument("--kd_ckd_out_root", type=str, default="./out")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_mode", choices=["logprob", "generate"], default="logprob")
    p.add_argument("--length_norm", choices=["none", "avg"], default="avg")
    p.add_argument(
        "--predictor_fast_logprob",
        type=_str2bool,
        default=False,
        help="If True, predictor models in logprob mode use one-pass scoring instead of forced autoregressive scoring.",
    )
    p.add_argument(
        "--predictor_ar_rescore_topk",
        type=int,
        default=0,
        help="When predictor_fast_logprob=True, rescore top-k candidates with autoregressive scoring (0 disables).",
    )
    p.add_argument(
        "--ar_rescore_topk",
        type=int,
        default=None,
        help="Alias of --predictor_ar_rescore_topk.",
    )
    p.add_argument(
        "--force_ar_logprob",
        type=_str2bool,
        default=False,
        help="Force autoregressive logprob scoring for diagnostics.",
    )
    p.add_argument(
        "--force_ar",
        type=_str2bool,
        default=None,
        help="Alias of --force_ar_logprob.",
    )
    p.add_argument(
        "--injection_window",
        type=int,
        default=None,
        help="Alias of --student_injection_window.",
    )
    p.add_argument(
        "--use_last_token",
        type=_str2bool,
        default=None,
        help="Alias of --student_use_last_token.",
    )
    p.add_argument("--print_scores", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.ar_rescore_topk is not None:
        args.predictor_ar_rescore_topk = int(args.ar_rescore_topk)
    if args.force_ar is not None:
        args.force_ar_logprob = bool(args.force_ar)
    if args.injection_window is not None:
        args.student_injection_window = int(args.injection_window)
    if args.use_last_token is not None:
        args.student_use_last_token = bool(args.use_last_token)
    return args


def _resolve_profile_overrides(args):
    profile = str(getattr(args, "eval_profile", "pipeline")).strip().lower()
    if profile != "kd_ckd":
        return

    args.run_stateft = False
    args.run_student_base = True
    args.run_student_predictor = False
    args.run_student_guidance = False
    args.student_use_control = False
    args.student_control_missing = "disable"

    legacy_pipeline_student = "/mlsteam/data/itri/stateft/new_idea/out/phase1_5_exp_p15_joint/student"
    student_ckpt = str(getattr(args, "student_ckpt_dir", "") or "").strip()
    if student_ckpt == legacy_pipeline_student:
        student_ckpt = ""
    if student_ckpt:
        if os.path.isdir(student_ckpt):
            print(f"[Profile] kd_ckd uses local student_ckpt_dir={student_ckpt}")
        else:
            print(f"[Profile] kd_ckd uses explicit student_ckpt={student_ckpt}")
        args.student_ckpt_dir = student_ckpt
        return

    detected = _find_latest_student_final(str(getattr(args, "kd_ckd_out_root", "./out")))
    if detected:
        args.student_ckpt_dir = detected
        print(f"[Profile] kd_ckd auto-detected student_ckpt_dir={detected}")
    else:
        print(
            "[Profile] kd_ckd could not auto-detect student_final under "
            f"{getattr(args, 'kd_ckd_out_root', './out')}; please pass --student_ckpt_dir explicitly."
        )


def main():
    args = parse_args()
    _resolve_profile_overrides(args)
    _set_seed(int(args.seed))
    run_pipeline_eval(args)


if __name__ == "__main__":
    main()
