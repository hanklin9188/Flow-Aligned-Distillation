# Model mount layout

Weights are intentionally excluded from Git. Mount or link licensed artifacts into this layout:

```text
models/
├── base_llama32_3b/       # base HF snapshot, including config + tokenizer
├── merged_teacher/        # BF16 merged LoRA teacher
└── student/
    ├── deploy_bundle.pt   # BF16 FAD K=17 bundle
    └── oracle_distilled_exit_policy.json
```

Expected SHA-256 values for the audited 2026-07-19 deployment:

```text
809cb6017b1526465074360ffdebafc43cde3a07e968129a3e5e94ef9c860470  deploy_bundle.pt
8848796726e0bb63577bb1c5c18df84b4cdd0bfae1544240f2c391143dc3b55e  oracle_distilled_exit_policy.json
bb6cf425e2395ab7d7c9e8d55a37dd66b5af3907bacf0bdfe6f4fff39de09d4d  merged_teacher/model.safetensors
```

Verify with `sha256sum <file>`. The base and teacher are subject to the Meta Llama license and are not redistributed here.
