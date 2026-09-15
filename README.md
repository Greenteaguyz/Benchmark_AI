# Benchmarking Open Source Reasoning LLMs on an 8 GB VRAM Computer

[![Ollama Version](https://img.shields.io/badge/Ollama-0.34.0-blue.svg)](https://ollama.com/)
[![CUDA Version](https://img.shields.io/badge/CUDA-13.4-green.svg)](https://developer.nvidia.com/cuda-zone)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20RTX%205060%20(8GB)-76B900.svg)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-Academic%20Research-orange.svg)](./)

A controlled comparative benchmark of open-weight reasoning Large Language Models (LLMs) deployed locally on a consumer laptop with an **8 GB VRAM GPU constraint**.

---

## 👥 Research Team & Assigned Roles

* **Student 1: Hout Chanvireak** — *Benchmark & Evaluation Lead*
  * Responsible for benchmark question design, verified answer keys, scoring rubrics, master spreadsheet, evaluation analysis, and findings presentation.
* **Student 2: Sok Ratanakvichea** — *Deployment & Integration Lead*
  * Responsible for local Ollama installation, hardware monitoring, GPU acceleration proof, automated collection runner, comparison tools, and technical demonstration.

---

## 🎯 Main Research Question

> **"Which open source reasoning model provides the best balance of answer accuracy, reasoning quality, and local performance on an 8 GB VRAM computer?"**

### Evaluated Models

| Model | Ollama Tag | Parameters | Quantization | Size on Disk | Target VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phi 4 Mini Reasoning** | `phi4-mini-reasoning:latest` | 3.84B | Q4_K_M | ~3.2 GB | ~3.9 GB |
| **DeepSeek R1 Distill Qwen 7B** | `deepseek-r1:7b` | 7.62B | Q4_K_M | ~4.7 GB | ~4.9 GB |
| **Qwen3 8B** | `qwen3:8b` | 8.19B | Q4_K_M | ~5.2 GB | ~5.7 GB |

*See [`docs/model_summary.md`](./docs/model_summary.md) for full architectural profiles.*

---

## 💻 Hardware & Test Environment

* **Operating System:** Windows 11 Home (Build 10.0.26200, 64-bit AMD64)
* **Processor (CPU):** Intel(R) Core(TM) i7-14700HX (20 Cores, 28 Threads)
* **System Memory:** 23.73 GB (~24 GB DDR5)
* **Graphics Card (GPU):** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MB Dedicated VRAM)
* **NVIDIA Driver / CUDA:** Driver 616.92 | CUDA UMD 13.4
* **Ollama Runtime:** Version 0.34.0 (Listening on `http://127.0.0.1:11434`)

*Full hardware sheet documented in [`docs/hardware_specs.md`](./docs/hardware_specs.md).*

---

## 📁 Repository Structure

```text
AI_Project/
├── data/
│   ├── responses/               # 45 official unedited JSON response files (Q01_PHI.json, etc.)
│   ├── comparison_log.csv       # Telemetry log across runs (latency, tok/s, VRAM)
│   ├── gpu_verification_result.json # Raw GPU acceleration benchmark data
│   └── three_models_smoke_test.json # Controlled 3-model smoke test metrics
├── docs/
│   ├── benchmark_questions.md   # 15 approved questions, answer key, rubric & freeze record
│   ├── daily_log.md             # Chronological project log (Day 1 & Day 2 deliverables)
│   ├── hardware_specs.md        # Official hardware & environment specification sheet
│   ├── model_summary.md         # Detailed architectural profiles and citations
│   ├── naming_guide.md          # Response schema and file naming convention
│   ├── response_schema.json     # Standardized JSON schema for model responses
│   ├── technical_setup_log.md   # Technical verification report and VRAM findings
│   └── evidence/                # Terminal screenshots validating local inference & GPU offload
│       ├── screenshot_01_hardware_and_ollama.png
│       ├── screenshot_02_phi4_reasoning_inference.png
│       └── screenshot_03_gpu_acceleration_proof.png
├── evaluation/
│   └── master_scoring.xlsx      # Master evaluation spreadsheet with 12 metrics & rubrics
├── test_runs/
│   ├── app.py                   # Streamlit comparison and benchmarking UI
│   ├── web_server.py            # Fast Starlette backend web dashboard
│   └── web/                     # Web dashboard HTML/CSS/JS frontend
├── tool/
│   ├── benchmark_runner.py      # Automated CLI runner saving responses & CSV logs
│   ├── generate_evidence_screenshots.py # Evidence screenshot generator
│   ├── generate_scoring_sheet.py # Generates master_scoring.xlsx
│   ├── record_hardware_specs.py # Hardware auto-detection script
│   ├── run_all_models_smoke_test.py # Controlled 3-model smoke test script
│   └── verify_gpu_and_run_phi.py # GPU verification and baseline inference script
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Prerequisites & Environment Setup
Clone the repository and initialize the Python virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Ollama Runtime & Model Download
Ensure Ollama is running locally:
```powershell
ollama --version
ollama pull phi4-mini-reasoning
ollama pull deepseek-r1:7b
ollama pull qwen3:8b
```

### 3. Record Hardware Specifications
Extract system specifications directly into `docs/hardware_specs.md`:
```powershell
python tool/record_hardware_specs.py
```

### 4. Verify GPU Acceleration & Baseline Inference
Run the baseline smoke test on `phi4-mini-reasoning` to confirm 100% GPU offload:
```powershell
python tool/verify_gpu_and_run_phi.py
```

### 5. Execute Controlled 3-Model Smoke Test
Run the common test across all three models with single-model VRAM eviction:
```powershell
python tool/run_all_models_smoke_test.py
```

### 6. Run Benchmark Questions via CLI
Run any question (e.g. `Q01`) against a specific model or all models:
```powershell
# Run Q01 on Phi 4 Mini Reasoning
python tool/benchmark_runner.py --qid Q01 --model PHI

# Run Q02 on all three models sequentially
python tool/benchmark_runner.py --qid Q02 --model ALL
```
Outputs will be saved automatically to `data/responses/Q{ID}_{MODEL}.json` and logged to `data/comparison_log.csv`.

---

## 📊 Controlled Smoke Test Telemetry (Day 2 Deliverable)

Under frozen settings (`temp=0.0`, `num_ctx=4096`, single model resident in VRAM):

| Model Package | Alias | Parameters | Peak VRAM | Offload | Generation Speed | Headroom (8 GB GPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `phi4-mini-reasoning:latest` | `PHI` | 3.84B | 3,987 MB | **100% GPU** | **76.47 tok/s** | +4,164 MB free |
| `deepseek-r1:7b` | `DSR1` | 7.62B | 4,887 MB | **100% GPU** | **53.26 tok/s** | +3,264 MB free |
| `qwen3:8b` | `QWEN` | 8.19B | 5,719 MB | **100% GPU** | **51.50 tok/s** | +2,432 MB free |

---

## ⚖️ Mandatory Research Rules Adherence

* **Zero Fine-Tuning:** Uses stock quantized models exclusively.
* **Controlled Environment:** All comparisons executed on the identical host machine with identical context limits.
* **Single Model in VRAM:** Explicit model eviction (`keep_alive: 0`) between trials.
* **No Regeneration:** Every inference run is accepted as produced; technical anomalies are logged.
* **Privacy & Isolation:** Local Ollama service is bound strictly to `127.0.0.1` and never exposed publicly.

---

## 📄 Google Docs Sync (Optional)

The Streamlit UI (`test_runs/app.py`) can append each scored result to a **Google Doc** of your choice using a free **Google Apps Script** web app (no Google Cloud, no card needed).

### One-time Google setup (~2 minutes)
1. Open the Google Doc you want results appended to → **Extensions → Apps Script**.
2. Paste this script and save:

```javascript
function doPost(e) {
  try {
    const doc = DocumentApp.getActiveDocument();
    const body = doc.getBody();
    const data = JSON.parse(e.postData.contents);
    body.appendParagraph("——————————————");
    body.appendParagraph(data.section_title);
    body.appendListItem("Fully Correct Rate: " + data.fully_correct_rate + "%");
    body.appendListItem("Average Quality Score: " + data.avg_quality_score + " / 8");
    body.appendListItem("Average Response Time: " + data.avg_response_time + " s");
    body.appendListItem("Tokens/Second: " + data.tokens_per_sec);
    body.appendListItem("Total Tokens: " + data.total_tokens);
    return ContentService.createTextOutput(JSON.stringify({ status: "ok" }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({ status: "error", message: String(err) }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}
```

3. **Deploy → New deployment → ⚙️ Web app** → *Execute as:* **Me**, *Who has access:* **Anyone** → **Deploy** → authorize → copy the **`…/exec` URL**.
4. Open the Streamlit app → sidebar **📄 Google Docs Sync** → paste the URL, tick **Enable**.

### How it works
After each test run, expand **"📝 Score this answer"**, enter the 4 rubric scores (0–2 each, matching the Page 3 rubric), and click **💾 Save Score & Sync to Docs**. The app:

1. Saves the score to `data/scoring_log.csv`.
2. Computes per-model metrics for the active mode (Test/Official) — Fully Correct Rate, Average Quality Score, Average Response Time, Tokens/sec, Total Tokens.
3. POSTs those metrics to your Apps Script URL, which appends them to your Google Doc.

Your `/exec` URL is kept in `google_docs_config.json` (gitignored — never push it).
