#!/usr/bin/env python3
"""Check whether the local Python/GPU environment can host both demo models."""

from __future__ import annotations

import json
import platform
import sys


def main() -> None:
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ok": False,
        "recommendation": "Use an NVIDIA GPU with at least 16 GB VRAM; 24 GB is preferred.",
    }
    try:
        import torch
        import transformers

        report["torch"] = torch.__version__
        report["transformers"] = transformers.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gib = props.total_memory / (1024 ** 3)
            report.update({"gpu": props.name, "vram_gib": round(vram_gib, 2)})
            report["ok"] = vram_gib >= 16.0
            if 12.0 <= vram_gib < 16.0:
                report["recommendation"] = (
                    "This GPU may be too small for two resident BF16 models. "
                    "Ask for the sequential-load or quantized package."
                )
            elif vram_gib < 12.0:
                report["recommendation"] = "Use a quantized/sequential version or a GPU with more VRAM."
        else:
            report["recommendation"] = "Install an NVIDIA driver and CUDA-enabled PyTorch."
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["recommendation"] = "Install packages from app/requirements-local.txt first."
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
