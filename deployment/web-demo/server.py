#!/usr/bin/env python3
"""Single-process local web server for the FAD student/teacher live demo."""

from __future__ import annotations

import argparse
import gc
import json
import mimetypes
import os
import secrets
import signal
import statistics
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "questions.json"
DEFAULT_STATIC = HERE / "static"
DEFAULT_MODELS = HERE.parent / "models"
DEFAULT_DEPLOY_BUNDLE = DEFAULT_MODELS / "student" / "deploy_bundle.pt"
DEFAULT_TEACHER = DEFAULT_MODELS / "merged_teacher"
DEFAULT_POLICY = DEFAULT_MODELS / "student" / "oracle_distilled_exit_policy.json"
DEFAULT_BASE_MODEL = DEFAULT_MODELS / "base_llama32_3b"
DEFAULT_RUNTIME_CODE = HERE / "runtime_code"


class DemoEngine:
    def __init__(
        self,
        *,
        catalog_path: Path,
        deploy_bundle: Path,
        base_model: Path,
        teacher_path: Path,
        policy_path: Path,
        runtime_code_dir: Path,
        mock: bool,
        warmup_rounds: int,
        benchmark_repeats: int,
        benchmark_warmups: int,
    ) -> None:
        self.catalog_path = catalog_path
        self.catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.questions = {
            str(question["id"]): question for question in self.catalog.get("questions", [])
        }
        if not self.questions:
            raise ValueError(f"question catalog is empty: {catalog_path}")
        self.question_list = list(self.questions.values())
        self.questions_by_dataset: dict[str, list[dict[str, Any]]] = {}
        for question in self.question_list:
            dataset = str(question["dataset"])
            self.questions_by_dataset.setdefault(dataset, []).append(question)
        self.random = secrets.SystemRandom()
        self.deploy_bundle = deploy_bundle
        self.base_model = base_model
        self.teacher_path = teacher_path
        self.policy_path = policy_path
        self.runtime_code_dir = runtime_code_dir
        self.mock = bool(mock)
        self.warmup_rounds = max(0, int(warmup_rounds))
        self.benchmark_repeats = max(1, int(benchmark_repeats))
        self.benchmark_warmups = max(0, int(benchmark_warmups))
        self.lock = threading.Lock()
        self.ready = False
        self.loading_stage = "starting"
        self.request_count = 0
        self.last_error = ""
        self.gpu_name = "Mock GPU" if self.mock else ""
        self.torch_version = "mock" if self.mock else ""
        self.transformers_version = "mock" if self.mock else ""
        self.student = None
        self.teacher = None
        self.tokenizer = None
        self.controller: dict[str, Any] = {}
        self.runtime = None
        self.pipe = None
        self.torch = None
        self.device_total_bytes = 0
        self.model_tensor_bytes: dict[str, int] = {"student": 0, "teacher": 0}

    @staticmethod
    def _module_tensor_storage_bytes(model: Any) -> int:
        """Count unique parameter/buffer tensor storage resident for one model."""
        seen: set[int] = set()
        total = 0
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor is None:
                continue
            storage = tensor.untyped_storage()
            storage_ptr = int(storage.data_ptr())
            if storage_ptr in seen:
                continue
            seen.add(storage_ptr)
            total += int(storage.nbytes())
        return int(total)

    def load(self) -> None:
        if self.mock:
            self.loading_stage = "mock_ready"
            self.ready = True
            return

        for path in (
            self.deploy_bundle,
            self.base_model / "config.json",
            self.teacher_path / "config.json",
            self.policy_path,
            self.runtime_code_dir / "runtime_confidence_exit_eval_final_llama.py",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if str(self.runtime_code_dir) not in sys.path:
            sys.path.insert(0, str(self.runtime_code_dir))

        self.loading_stage = "importing_runtime"
        import torch
        import transformers
        import runtime_confidence_exit_eval_final_llama as runtime

        self.torch = torch
        self.torch_version = str(torch.__version__)
        self.transformers_version = str(transformers.__version__)
        self.runtime = runtime
        self.pipe = runtime.pipe
        self.pipe.set_seed(44)
        device = self.pipe.resolve_device("cuda", -1)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for real-model demo mode.")
        self.device = device
        self.dtype = self.pipe.get_target_dtype(device)
        self.gpu_name = torch.cuda.get_device_name(device)
        self.device_total_bytes = int(torch.cuda.get_device_properties(device).total_memory)

        self.loading_stage = "loading_student"
        bundle = torch.load(str(self.deploy_bundle), map_location="cpu")
        if not isinstance(bundle, dict):
            raise ValueError("student deploy bundle must contain a dictionary")
        shared_payload = bundle["shared_student"]
        self.student, self.quant_report = self.pipe._build_shared_model_for_eval(
            base_model=str(self.base_model),
            atlas_payload=bundle.get("atlas", {}),
            shared_payload=shared_payload,
            quant_bank_int4=bundle.get("quant_bank_int4"),
            use_quant_bank_int4=False,
            device=device,
            dtype=self.dtype,
            trust_remote_code=True,
        )
        self.tokenizer = self.pipe.load_tokenizer(str(self.teacher_path), trust_remote_code=True)
        self.model_tensor_bytes["student"] = self._module_tensor_storage_bytes(self.student)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        del bundle
        gc.collect()
        torch.cuda.empty_cache()

        self.loading_stage = "loading_teacher"
        self.teacher = self.pipe.AutoModelForCausalLM.from_pretrained(
            str(self.teacher_path),
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.model_tensor_bytes["teacher"] = self._module_tensor_storage_bytes(self.teacher)
        self.controller = self.runtime._apply_controller_threshold_override(
            self.runtime._load_exit_controller(str(self.policy_path)), -1.0
        )

        self.loading_stage = "warming_up"
        # The catalog contains the complete 19,149-question test set. Startup
        # warm-up must remain constant-time instead of evaluating every item.
        warmup_question = min(
            self.question_list,
            key=lambda question: len(str(question.get("instruction", ""))),
        )
        for _ in range(self.warmup_rounds):
            self._infer_real("student", warmup_question, measure=False)
            self._infer_real("teacher", warmup_question, measure=False)
        torch.cuda.synchronize()
        self.loading_stage = "ready"
        self.ready = True

    def public_catalog(self) -> dict[str, Any]:
        return {
            "title": self.catalog.get("title", "Model demo"),
            "selection_rule": self.catalog.get("selection_rule", ""),
            "categories": self.catalog.get("categories", []),
            "question_count": len(self.question_list),
            "sample_size": 5,
            "datasets": [
                {"id": dataset, "count": len(questions)}
                for dataset, questions in self.questions_by_dataset.items()
            ],
            "paper_benchmark": self.catalog.get("paper_benchmark", {}),
        }

    @staticmethod
    def _public_question(question: dict[str, Any], draw_rank: int) -> dict[str, Any]:
        """Return question text without leaking historical predictions or gold."""
        return {
            "id": str(question["id"]),
            "dataset": str(question["dataset"]),
            "sample_id": int(question["sample_id"]),
            "draw_rank": int(draw_rank),
            "display_stem": str(question["display_stem"]),
            "choices": list(question.get("choices", [])),
        }

    def random_questions(self, *, count: int = 5, dataset: str = "all") -> dict[str, Any]:
        count = int(count)
        if count < 1 or count > 20:
            raise ValueError("count must be between 1 and 20")
        dataset = str(dataset or "all")
        if dataset == "all":
            pool = self.question_list
            pool_label = "all"
        else:
            pool = self.questions_by_dataset.get(dataset)
            if pool is None:
                raise ValueError(f"unknown dataset: {dataset}")
            pool_label = dataset
        if len(pool) < count:
            raise ValueError(f"sampling pool contains only {len(pool)} questions")
        selected = self.random.sample(pool, count)
        return {
            "ok": True,
            "sampling": "uniform_without_replacement",
            "pool": pool_label,
            "pool_size": len(pool),
            "count": count,
            "questions": [
                self._public_question(question, draw_rank=rank)
                for rank, question in enumerate(selected, start=1)
            ],
        }

    def health(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ready),
            "ready": bool(self.ready),
            "mode": "mock" if self.mock else "live_gpu",
            "loading_stage": self.loading_stage,
            "gpu": self.gpu_name,
            "torch": self.torch_version,
            "transformers": self.transformers_version,
            "request_count": int(self.request_count),
            "busy": self.lock.locked(),
            "last_error": self.last_error,
            "student": "25% FAD adaptive exit (K=17; exits 16/20/24; fallback 28)",
            "teacher": "Merged teacher (full depth, 28 layers)",
            "benchmark_protocol": {
                "name": "per-model warm-up + interleaved median",
                "repeats": int(self.benchmark_repeats),
                "warmups_before_each_trial": int(self.benchmark_warmups),
                "cuda_synchronized": True,
            },
            "vram": {
                "device_total_bytes": int(self.device_total_bytes),
                "student_model_tensor_bytes": int(self.model_tensor_bytes["student"]),
                "teacher_model_tensor_bytes": int(self.model_tensor_bytes["teacher"]),
                "measurement": "PyTorch CUDA allocator peak during inference",
            },
        }

    @staticmethod
    def _choice_text(question: dict[str, Any], prediction: str) -> str:
        for choice in question.get("choices", []):
            if str(choice.get("label", "")) == str(prediction):
                return str(choice.get("text", ""))
        return ""

    def _mock_result(self, model_kind: str, question: dict[str, Any]) -> dict[str, Any]:
        historical = question["historical"]
        if model_kind == "student":
            prediction = str(historical["ours_prediction"])
            latency_ms = float(historical["ours_latency_ms"])
            exit_layer = int(historical["ours_exit_layer"])
        else:
            prediction = str(historical["teacher_prediction"])
            latency_ms = float(historical["teacher_latency_ms"])
            exit_layer = int(historical["teacher_exit_layer"])
        # Small delay makes the preview interaction visible without pretending
        # that the mock result is fresh GPU inference.
        time.sleep(min(0.18, max(0.02, latency_ms / 1000.0)))
        return {
            "model": model_kind,
            "prediction": prediction,
            "answer_text": self._choice_text(question, prediction),
            "gold_answer": str(question["gold_answer"]),
            "correct": prediction == str(question["gold_answer"]),
            "latency_ms": latency_ms,
            "exit_layer": exit_layer,
            "final_layer": 28,
            "confidence": None,
            "source": "historical_mock",
            "vram": None,
        }

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            raise ValueError("cannot summarize an empty latency list")
        if len(ordered) == 1:
            return ordered[0]
        position = min(1.0, max(0.0, float(fraction))) * (len(ordered) - 1)
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def _summarize_trials(self, trials: list[dict[str, Any]]) -> dict[str, Any]:
        if not trials:
            raise ValueError("benchmark produced no trials")
        latencies = [float(trial["latency_ms"]) for trial in trials]
        output = dict(trials[-1])
        output["latency_ms"] = float(statistics.median(latencies))
        output["timing"] = {
            "statistic": "median",
            "protocol": "steady_state_interleaved",
            "repeats": len(latencies),
            "warmups_before_each_trial": int(self.benchmark_warmups),
            "cuda_synchronized": not self.mock,
            "trials_ms": latencies,
            "min_ms": min(latencies),
            "p25_ms": self._percentile(latencies, 0.25),
            "median_ms": float(statistics.median(latencies)),
            "p75_ms": self._percentile(latencies, 0.75),
            "max_ms": max(latencies),
        }
        vram_trials = [trial.get("vram") for trial in trials if trial.get("vram")]
        if vram_trials:
            latest = dict(vram_trials[-1])
            peak_fields = (
                "process_baseline_bytes",
                "process_peak_allocated_bytes",
                "process_peak_reserved_bytes",
                "inference_peak_extra_bytes",
                "device_used_after_bytes",
            )
            for field in peak_fields:
                latest[field] = max(int(row.get(field, 0)) for row in vram_trials)
            latest["statistic"] = "max_over_timed_trials"
            latest["trial_process_peak_allocated_bytes"] = [
                int(row.get("process_peak_allocated_bytes", 0)) for row in vram_trials
            ]
            latest["trial_inference_peak_extra_bytes"] = [
                int(row.get("inference_peak_extra_bytes", 0)) for row in vram_trials
            ]
            output["vram"] = latest
        return output

    def _benchmark(
        self, question: dict[str, Any], models: list[str]
    ) -> dict[str, dict[str, Any]]:
        if self.mock:
            results: dict[str, dict[str, Any]] = {}
            for model_kind in models:
                trial = self._mock_result(model_kind, question)
                results[model_kind] = self._summarize_trials(
                    [dict(trial) for _ in range(self.benchmark_repeats)]
                )
            return results

        trials: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
        for repeat_index in range(self.benchmark_repeats):
            # Reversing every round removes the systematic advantage of always
            # running second.  Each timed run immediately follows its own
            # unmeasured warm-up, matching the steady-state paper protocol and
            # avoiding Windows WDDM residency / GPU clock-ramp bias.
            order = models if repeat_index % 2 == 0 else list(reversed(models))
            for model_kind in order:
                for _ in range(self.benchmark_warmups):
                    self._infer_real(model_kind, question, measure=False)
                trials[model_kind].append(
                    self._infer_real(model_kind, question, measure=True)
                )
        return {
            model_kind: self._summarize_trials(model_trials)
            for model_kind, model_trials in trials.items()
        }

    def _infer_real(
        self, model_kind: str, question: dict[str, Any], *, measure: bool = True
    ) -> dict[str, Any]:
        assert self.runtime is not None and self.pipe is not None and self.torch is not None
        prompt = self.pipe._build_prompt(question, str(question["dataset"]))
        candidates = self.pipe._candidates_for_dataset(str(question["dataset"]))
        if model_kind == "student":
            model = self.student
            threshold = 0.90
            force_full = False
            controller = self.controller
        elif model_kind == "teacher":
            model = self.teacher
            # This deliberately mirrors the saved benchmark: threshold 2.0
            # checks intermediate layers but can only return at layer 28.
            threshold = 2.0
            force_full = False
            controller = None
        else:
            raise ValueError(f"unknown model: {model_kind}")
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats(self.device)
        baseline_allocated = int(self.torch.cuda.memory_allocated(self.device))
        baseline_reserved = int(self.torch.cuda.memory_reserved(self.device))
        started = time.perf_counter()
        result = self.runtime.runtime_confidence_predict(
            model=model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            candidates=candidates,
            exit_layers=[16, 20, 24],
            threshold=threshold,
            device=self.device,
            length_norm="none",
            temperature=1.0,
            force_full=force_full,
            controller=controller,
        )
        self.torch.cuda.synchronize()
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        peak_allocated = int(self.torch.cuda.max_memory_allocated(self.device))
        peak_reserved = int(self.torch.cuda.max_memory_reserved(self.device))
        free_after, total_bytes = self.torch.cuda.mem_get_info(self.device)
        prediction = str(result["pred"])
        output = {
            "model": model_kind,
            "prediction": prediction,
            "answer_text": self._choice_text(question, prediction),
            "gold_answer": str(question["gold_answer"]),
            "correct": prediction == str(question["gold_answer"]),
            "latency_ms": elapsed_ms,
            "exit_layer": int(result["exit_layer"]),
            "final_layer": int(result["final_layer"]),
            "confidence": float(result["confidence"]),
            "source": "live_gpu" if measure else "warmup",
            "vram": {
                "measurement": "pytorch_cuda_allocator",
                "scope": "single_process_with_both_models_resident",
                "device_total_bytes": int(total_bytes),
                "device_used_after_bytes": int(total_bytes - free_after),
                "model_tensor_bytes": int(self.model_tensor_bytes[model_kind]),
                "process_baseline_bytes": baseline_allocated,
                "process_baseline_reserved_bytes": baseline_reserved,
                "process_peak_allocated_bytes": peak_allocated,
                "process_peak_reserved_bytes": peak_reserved,
                "inference_peak_extra_bytes": max(0, peak_allocated - baseline_allocated),
            },
        }
        return output

    def infer(
        self, *, question_id: str, models: list[str], benchmark: bool = False
    ) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError(f"models are not ready (stage={self.loading_stage})")
        question = self.questions.get(str(question_id))
        if question is None:
            raise KeyError(f"unknown question_id: {question_id}")
        normalized: list[str] = []
        for model in models:
            name = str(model).strip().lower()
            if name not in {"student", "teacher"}:
                raise ValueError(f"unsupported model: {model}")
            if name not in normalized:
                normalized.append(name)
        if not normalized:
            raise ValueError("models must contain student and/or teacher")

        with self.lock:
            started = time.perf_counter()
            if benchmark:
                results = self._benchmark(question, normalized)
            else:
                results: dict[str, Any] = {}
                for model_kind in normalized:
                    result = (
                        self._mock_result(model_kind, question)
                        if self.mock
                        else self._infer_real(model_kind, question)
                    )
                    results[model_kind] = result
            self.request_count += 1
            response = {
                "ok": True,
                "question_id": str(question_id),
                "mode": "mock" if self.mock else "live_gpu",
                "execution": "serialized_on_one_gpu",
                "timing_protocol": (
                    "steady_state_interleaved_median" if benchmark else "single_run"
                ),
                "results": results,
                "request_wall_ms": 1000.0 * (time.perf_counter() - started),
            }
            if "student" in results and "teacher" in results:
                student_ms = float(results["student"]["latency_ms"])
                teacher_ms = float(results["teacher"]["latency_ms"])
                response["comparison"] = {
                    "student_time_saved_ms": teacher_ms - student_ms,
                    "student_speedup": teacher_ms / student_ms if student_ms > 0 else None,
                }
            return response


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], *, engine: DemoEngine, static_dir: Path):
        super().__init__(address, handler)
        self.engine = engine
        self.static_dir = static_dir.resolve()


class Handler(BaseHTTPRequestHandler):
    server: DemoHTTPServer

    def _json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, relative: str) -> None:
        allowed = {
            "": "index.html",
            "/": "index.html",
            "/index.html": "index.html",
            "/styles.css": "styles.css",
            "/app.js": "app.js",
        }
        filename = allowed.get(relative)
        if filename is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = self.server.static_dir / filename
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if filename == "index.html":
            html = path.read_text(encoding="utf-8")
            embedded = json.dumps(
                {"ok": True, **self.server.engine.public_catalog()},
                ensure_ascii=False,
            ).replace("</", "<\\/")
            body = html.replace("__QUESTION_CATALOG_JSON__", embedded).encode("utf-8")
        else:
            body = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") else mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._json(self.server.engine.health())
            return
        if path == "/api/questions":
            self._json({"ok": True, **self.server.engine.public_catalog()})
            return
        if path == "/api/sample":
            try:
                query = parse_qs(parsed.query)
                count = int(query.get("count", ["5"])[0])
                dataset = str(query.get("dataset", ["all"])[0])
                self._json(self.server.engine.random_questions(count=count, dataset=dataset))
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/infer":
            self._json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                raise ValueError("request body must be between 1 and 65536 bytes")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            models = payload.get("models", [])
            if isinstance(models, str):
                models = [models]
            if not isinstance(models, list):
                raise ValueError("models must be a list")
            response = self.server.engine.infer(
                question_id=str(payload.get("question_id", "")),
                models=[str(model) for model in models],
                benchmark=bool(payload.get("benchmark", False)),
            )
            self._json(response)
        except KeyError as exc:
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.server.engine.last_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self._json(
                {"ok": False, "error": "inference failed", "detail": str(exc)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} {fmt % args}", flush=True)


def write_service_info(path: Path | None, *, host: str, port: int, engine: DemoEngine) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": host,
        "port": int(port),
        "mode": "mock" if engine.mock else "live_gpu",
        "gpu": engine.gpu_name,
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_node": (
            os.environ.get("SLURMD_NODENAME")
            or os.environ.get("COMPUTERNAME")
            or "local"
        ),
        "ready": bool(engine.ready),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FAD student/teacher live demo server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--static_dir", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--deploy_bundle", type=Path, default=DEFAULT_DEPLOY_BUNDLE)
    parser.add_argument("--base_model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--runtime_code", type=Path, default=DEFAULT_RUNTIME_CODE)
    parser.add_argument("--warmup_rounds", type=int, default=3)
    parser.add_argument("--benchmark_repeats", type=int, default=5)
    parser.add_argument("--benchmark_warmups", type=int, default=1)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--service_info", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = DemoEngine(
        catalog_path=args.questions,
        deploy_bundle=args.deploy_bundle,
        base_model=args.base_model,
        teacher_path=args.teacher,
        policy_path=args.policy,
        runtime_code_dir=args.runtime_code,
        mock=args.mock,
        warmup_rounds=args.warmup_rounds,
        benchmark_repeats=args.benchmark_repeats,
        benchmark_warmups=args.benchmark_warmups,
    )
    print(f"[Demo] mode={'mock' if args.mock else 'live_gpu'}", flush=True)
    engine.load()
    write_service_info(args.service_info, host=args.host, port=args.port, engine=engine)
    server = DemoHTTPServer((args.host, int(args.port)), Handler, engine=engine, static_dir=args.static_dir)

    def stop_server(signum: int, _frame: Any) -> None:
        print(f"[Demo] received signal {signum}; shutting down", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    print(
        f"[Demo] READY http://{args.host}:{args.port} mode={'mock' if args.mock else 'live_gpu'} "
        f"gpu={engine.gpu_name}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        print("[Demo] stopped", flush=True)


if __name__ == "__main__":
    main()
