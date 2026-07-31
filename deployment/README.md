# Deploying the 25% student + teacher demo

The GitHub Pages portfolio is static. Real inference needs a CUDA host because the audited demo keeps the BF16 25% FAD student and merged teacher resident together.

## Architecture

```text
browser ──HTTP/JSON──> single-process demo server ──GPU lock──> FAD-AE student
                                      │                       exits 16/20/24/28
                                      └──────────────────────> merged teacher
                                                              full depth 28
```

The student retains the 28-layer Transformer skeleton but maps those layers to 17 unique FFN prototypes: layers 0–13 and 27 remain private; layers 14–18 share one prototype and 19–26 share another. This saves 11 FFN parameter sets, approximately 0.83B weights or 25.8% of the original ~3.21B parameters. It is structural sharing, not “keeping only 25% of the model.”

The adaptive controller observes four runtime-only features—normalized entropy, top-answer surprisal, Jensen–Shannon change from the previous exit and information-energy gap. A standardized linear sigmoid head uses decision threshold 0.50. The answer-confidence gate is 0.90. No teacher or task label is required at deployment.

## Local mock preview

```bash
python deployment/web-demo/server.py --mock \
  --questions deployment/web-demo/questions.json \
  --static_dir deployment/web-demo/static --port 8765
```

Mock mode is visibly labelled and never claims live latency. The public question catalog contains ten HellaSwag examples selected for presentation, including labels and historical student/teacher outputs used by mock mode. Point `--questions` at a separately obtained full catalog only in a controlled environment.

## Real GPU mode

Requirements: Python 3.11+, CUDA-capable PyTorch, 16 GB VRAM minimum (24 GB recommended), and roughly 16 GB disk for all model artifacts.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r deployment/web-demo/requirements.txt

export FAD_BASE_MODEL=/models/base_llama32_3b
export FAD_TEACHER_MODEL=/models/merged_teacher
export FAD_STUDENT_BUNDLE=/models/student/deploy_bundle.pt
export FAD_EXIT_POLICY=/models/student/oracle_distilled_exit_policy.json
bash deployment/web-demo/start.sh 7860
```

Run the endpoint smoke test in another shell:

```bash
python deployment/web-demo/smoke_test.py --base_url http://127.0.0.1:7860
```

## Container

```bash
docker build -t fad-demo deployment/web-demo
docker run --rm --gpus all -p 127.0.0.1:7860:7860 \
  -v /models:/app/models:ro fad-demo
```

## Public access and safety

The included Python service is a controlled research demo, not a hardened multi-tenant API. Bind it to localhost and use an SSH tunnel, trusted VPN or authenticated TLS reverse proxy. Before exposing it publicly, add authentication, request-size limits, rate limiting, an inference queue, timeouts, health monitoring and origin restrictions. Do not publish the licensed model directories.

For an SSH tunnel from a presentation laptop:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@gpu-host
```

Then open <http://127.0.0.1:7860>. Student and teacher requests are serialized with a lock so they do not compete for GPU resources during latency comparison.

## Audited result

On 19,149 paired questions, the adaptive student answered 17,079 correctly versus the teacher's 16,726. There were 15,890 both-correct, 1,189 student-only-correct, 836 teacher-only-correct and 1,234 both-wrong cases. These are seven-task multiple-choice results; they must not be generalized to free-form chat, MMLU or mathematical reasoning.
