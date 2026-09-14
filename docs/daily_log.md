# Daily Project Log — 2026-09-12

### Work Completed
- **File Naming & Response Schema**: Standardized response format (`data/responses/Q{ID}_{MODEL}.json`) defined in [`docs/response_schema.json`](./response_schema.json) with helper module [`tool/response_schema.py`](../tool/response_schema.py).
- **Master Scoring Sheet**: Created [`evaluation/master_scoring.xlsx`](../evaluation/master_scoring.xlsx) with 12 evaluation metrics, automated sum scoring, and 0–2 score validation.
- **Environment Setup**: Initialized directories (`data/responses`, `evaluation`, `tool`, `docs`) and fixed UTF-8 subprocess handling in `setup_env.py`.

### Model Documentation

| Package (Tag) | Alias | Parameters | Size (Disk) | Purpose in Benchmark | Official Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `phi4-mini-reasoning` | `PHI` | 3.84B | ~3.2 GB | Lightweight reasoning baseline for constrained hardware | [Ollama Library](https://ollama.com/library/phi4-mini-reasoning:latest) |
| `deepseek-r1:7b` | `DSR1` | 7.62B | ~4.7 GB | Compact distilled reasoning model | [Ollama Library](https://ollama.com/library/deepseek-r1:7b) |
| `qwen3:8b` | `QWEN` | 8.19B | ~5.2 GB | Larger general reasoning comparison model | [Ollama Library](https://ollama.com/library/qwen3:8b) |

---

# Daily Project Log — 2026-09-14

### Work Completed & Deliverables (Day 1 & Day 2)

#### Student 1: Hout Chanvireak (Benchmark & Evaluation Lead)
- **Folder Structure & Naming Guide (Day 1):** Defined standardized response scheme in [`docs/naming_guide.md`](./naming_guide.md) and initialized `data/responses/`, `evaluation/`, `tool/`, `docs/`.
- **Master Scoring Workbook (Day 1):** Generated [`evaluation/master_scoring.xlsx`](../evaluation/master_scoring.xlsx) featuring 45 pre-populated rows, 21 evaluation columns, sum scoring formulas, and 0–2 point validation.
- **Fifteen-Question Benchmark & Answer Key (Day 2):** Created and formalized [`docs/benchmark_questions.md`](./benchmark_questions.md) containing 15 standardized questions (5 Math, 5 Logic, 5 Programming), difficulty distribution (2 Easy, 2 Med, 1 Hard), verified answer keys, and expected reasoning points.
- **Pilot Test & Benchmark Freeze (Day 2):** Piloted `Q01`, `Q06`, and `Q11` with Sok Ratanakvichea across all three models, confirmed semantic discrimination and syntax adherence, and officially certified the benchmark as **FROZEN**.

#### Student 2: Sok Ratanakvichea (Deployment & Integration Lead)
- **Hardware & Environment Specifications (Day 1):** Recorded host specs in [`docs/hardware_specs.md`](./hardware_specs.md) using [`tool/record_hardware_specs.py`](../tool/record_hardware_specs.py) (Windows 11 Build 26200, i7-14700HX, 24 GB RAM, RTX 5060 Laptop GPU with 8 GB VRAM, Drive C: 84.07 GB free).
- **Ollama Installation & Daemon (Day 1):** Verified local Ollama runtime v0.34.0 running on daemon port `11434` (PID 28396).
- **GPU Acceleration Verification (Day 1):** Validated 100% GPU layer offloading with `ollama ps` and `nvidia-smi` during active inference (`llama-server.exe` allocating up to 3,974 MiB VRAM with 60%–95% compute load).
- **Phi 4 Mini Reasoning Inference (Day 1):** Executed baseline inference on `phi4-mini-reasoning:latest` (3.84B), achieving sustained throughput of **75.12 to 89.47 tokens/sec**. Captured `<think>` reasoning chain in [`data/gpu_verification_result.json`](../data/gpu_verification_result.json).
- **Evidence Screenshots (Day 1):** Captured and stored crisp terminal screenshots in `docs/evidence/`:
  - [`docs/evidence/screenshot_01_hardware_and_ollama.png`](./evidence/screenshot_01_hardware_and_ollama.png): Hardware specs and Ollama installation.
  - [`docs/evidence/screenshot_02_phi4_reasoning_inference.png`](./evidence/screenshot_02_phi4_reasoning_inference.png): Phi 4 Mini Reasoning inference output and token speed.
  - [`docs/evidence/screenshot_03_gpu_acceleration_proof.png`](./evidence/screenshot_03_gpu_acceleration_proof.png): 100% GPU offloading and VRAM allocation.
- **Controlled 3-Model Smoke Test (Day 2):** Tested all three models (`PHI`, `DSR1`, `QWEN`) under identical frozen settings (temp=0.0, ctx=4096) with strict single-model VRAM residency (`keep_alive: 0` eviction). Output archived in [`data/three_models_smoke_test.json`](../data/three_models_smoke_test.json):
  - `phi4-mini-reasoning:latest`: 3,987 MB Peak VRAM, 76.47 tok/s
  - `deepseek-r1:7b`: 4,887 MB Peak VRAM, 53.26 tok/s
  - `qwen3:8b`: 5,719 MB Peak VRAM, 51.50 tok/s (2,432 MB free headroom on 8 GB GPU)
- **Automated Benchmark Runner (Day 2):** Developed and delivered [`tool/benchmark_runner.py`](../tool/benchmark_runner.py) to automate benchmark execution, telemetry logging, and JSON serialization directly into `data/responses/`.


