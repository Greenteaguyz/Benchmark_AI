# Technical Setup & GPU Verification Log

**Date:** 2026-09-14  
**Project:** Local Reasoning LLM Benchmark on Constrained Hardware (8 GB VRAM Study)  
**Status:** Completed & Verified  

---

## 1. Executive Summary

All prerequisites for the local LLM reasoning benchmark study have been established, validated, and documented.
- **Hardware Profile:** Successfully cataloged the host machine (Intel Core i7-14700HX, 24 GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB VRAM).
- **Ollama Runtime:** Verified local installation of Ollama version `0.34.0` running as a background service listening on `127.0.0.1:11434`.
- **Model Verification:** Verified local presence of the required baseline model `phi4-mini-reasoning:latest` (3.84B parameters, ~3.2 GB on disk, Q4_K_M quantization).
- **GPU Acceleration:** Confirmed full GPU layer offload (`100% GPU` indicated in `ollama ps`), with `llama-server.exe` actively consuming 3.97 GB of VRAM during execution.
- **Inference Performance:** Baseline model successfully solved a multi-step logical riddle with full internal `<think>` reasoning tokens at a throughput of **75.12 tokens/second**.

---

## 2. Hardware & Environment Specifications

| Specification Category | Parameter | Recorded Value |
| :--- | :--- | :--- |
| **Operating System** | OS Name & Build | Windows 11 Home (Build 10.0.26200, 64-bit AMD64) |
| **Processor (CPU)** | Model | Intel(R) Core(TM) i7-14700HX (20 Cores: 8 P-Cores, 12 E-Cores, 28 Threads) |
| **System Memory** | Total Physical RAM | 23.73 GB (~24 GB DDR5) |
| **Graphics Processor** | GPU Model | NVIDIA GeForce RTX 5060 Laptop GPU |
| **GPU Architecture** | Driver & CUDA | Driver Version: 616.92 \| CUDA UMD Version: 13.4 |
| **Dedicated VRAM** | Capacity | 7.96 GB (8,151 MB Total VRAM) |
| **Ollama Runtime** | Version | 0.34.0 |
| **Model Storage Path** | Directory & Free Space | Drive `C:\Users\User\.ollama\models` (84.07 GB free of 475.82 GB) |

*Full specification sheet is documented in [`docs/hardware_specs.md`](./hardware_specs.md).*

---

## 3. Ollama Installation & Service Status

- **Executable Location:** `C:\Users\User\AppData\Local\Programs\Ollama\ollama.exe`
- **Daemon Process ID:** `28396` (Running as active local service)
- **Local Endpoint:** `http://127.0.0.1:11434`
- **Network Listening State:**
  ```text
  TCP    127.0.0.1:11434    0.0.0.0:0    LISTENING    28396
  ```

### Local Model Registry
```text
NAME                          ID              SIZE      MODIFIED
phi4-mini-reasoning:latest    3ca8c2865ce9    3.2 GB    36 hours ago
deepseek-r1:7b                755ced02ce7b    4.7 GB    15 hours ago
qwen3:8b                      500a1f067a9f    5.2 GB    14 hours ago
```

---

## 4. GPU Acceleration Verification

GPU acceleration was evaluated during model load and active token generation using both the Ollama runtime inspection CLI and the NVIDIA Management Interface (`nvidia-smi`).

### 4.1 Process Offload State (`ollama ps`)
```text
NAME                          ID              SIZE      PROCESSOR    CONTEXT    UNTIL
phi4-mini-reasoning:latest    3ca8c2865ce9    3.7 GB    100% GPU     4096       4 minutes from now
```
> **Confirmation:** `100% GPU` confirms that zero transformer layers were offloaded to CPU or system RAM. All weights and KV cache reside strictly in high-speed GDDR6 VRAM.

### 4.2 NVIDIA System Management Interface (`nvidia-smi`)
```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 616.92                 KMD Version: 616.92        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   64C    P0             63W /   65W |    3973MiB /   8151MiB |     60%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           17268      C   ...a\lib\ollama\llama-server.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

- **VRAM Allocated:** 3,973 MiB (~48.7% of total 8 GB VRAM budget).
- **Compute Load:** GPU utilization reached 60% with P0 full-performance power state (63W).
- **Process Binding:** Verified compute process `llama-server.exe` (PID 17268) attached to CUDA device 0.

---

## 5. Local Inference Smoke Test: Phi 4 Mini Reasoning

### 5.1 Test Configuration
- **Model:** `phi4-mini-reasoning:latest` (3.84B parameters)
- **Temperature:** `0.0` (Deterministic greedy decoding)
- **Context Limit:** `4,096` tokens
- **Test Prompt:**
  ```text
  Solve this step-by-step:
  A farmer has 15 sheep. All but 8 die. How many sheep does the farmer have left?
  Explain your reasoning concisely.
  ```

### 5.2 Performance Metrics
| Metric | Cold Run (First Execution) | Warm Run (Sustained In-Memory) |
| :--- | :--- | :--- |
| **Model Load Duration** | 7.286 seconds | 0.002 seconds (Instant) |
| **Prompt Tokens Evaluated** | 66 tokens | 66 tokens |
| **Completion Tokens Generated** | 445 tokens | 512 tokens |
| **Generation Duration** | 5.92 seconds | 5.72 seconds |
| **Generation Throughput** | **75.12 tokens/sec** | **89.47 tokens/sec** |
| **Observed Peak VRAM** | 3,973 MiB | 3,974 MiB |
| **GPU Utilization** | 60% (P0 State, 63W) | **95% (P2 State, 79W)** |

### 5.3 Generated Reasoning & Output
```text
<think>
Okay, let's see. The problem says a farmer has 15 sheep, and all but 8 die. How many does the farmer have left? Hmm. The key here is understanding the phrase "all but 8 die." Sometimes these tricky questions can be confusing because of the wording.

So, "all but 8" die. That means "all except 8" die. So, out of the 15 sheep, 8 survive, and the rest die. Wait, but the total is 15. So if 8 survive, then the number that died is 15 minus 8. Let me check that again. If you take away the 8 that didn't die, the rest died. So 15 total sheep, subtract the 8 that are left, so 15 - 8 = 7. So 7 sheep died, and 8 survived. Therefore, the farmer has 8 sheep left. 

But wait, maybe I should make sure I'm interpreting "all but 8" correctly. "All but 8" dying would mean that every sheep except for 8 dies. So yes, except 8, so 8 survive. So the answer should be 8. Let me think of another way. If all but 8 die, then the farmer is left with 8 sheep. Because "all" refers to all the sheep, but there's a exception of 8 that don't die. So the farmer still has those 8 alive, and the others (which are 15-8=7) died. So total alive is 8. Yeah, that makes sense. Sometimes these problems try to trick you into subtracting again, but here the 8 that survive are the ones remaining. So the answer is 8.
</think>

The farmer initially has 15 sheep. The phrase "all but 8 die" means every sheep except 8 dies.  
- **Total sheep**: 15  
- **Die**: 15 - 8 = 7  
- **Survive**: 8  

Thus, the farmer has **8 sheep left**.  

**Answer:** \boxed{8}
```

---

## 6. Day 2 Deliverable: Controlled 3-Model Smoke Test Table

Conforming to **Student 2 Day 2** obligations (*"Install DeepSeek R1 7B and Qwen3 8B. Test all three models with one common prompt. Record model tags and initial VRAM use."*), each model was tested in isolation (evicting previous models with `keep_alive: 0`):

| Model Package | Alias | Params | Disk Size | Load Time (s) | Peak VRAM | Allocated VRAM | GPU Offload | Generation Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `phi4-mini-reasoning:latest` | `PHI` | 3.84B | ~3.2 GB | 4.975s | 3,987 MB | ~3,738 MB | **100.0% GPU** (3.49 GB) | **76.47 tok/s** |
| `deepseek-r1:7b` | `DSR1` | 7.62B | ~4.7 GB | 6.017s | 4,887 MB | ~4,625 MB | **100.0% GPU** (4.42 GB) | **53.26 tok/s** |
| `qwen3:8b` | `QWEN` | 8.19B | ~5.2 GB | 6.348s | 5,719 MB | ~5,469 MB | **100.0% GPU** (5.20 GB) | **51.50 tok/s** |

> **VRAM Headroom Analysis:**
> - Largest model (`qwen3:8b`) peaked at **5,719 MB**, leaving **2,432 MB of free VRAM headroom** out of the 8,151 MB GPU limit.
> - **100% GPU offloading** verified across all three models.
> - Raw data archived in [`data/three_models_smoke_test.json`](../data/three_models_smoke_test.json).

---

## 7. Deliverables & Evidence Index

1. **Hardware Specifications Sheet:**
   - Markdown: [`docs/hardware_specs.md`](./hardware_specs.md)
   - Automated Extraction Script: [`tool/record_hardware_specs.py`](../tool/record_hardware_specs.py)
2. **Benchmark Question Bank & Rubric (Student 1 Day 2):**
   - Markdown: [`docs/benchmark_questions.md`](./benchmark_questions.md)
   - Master Workbook: [`evaluation/master_scoring.xlsx`](../evaluation/master_scoring.xlsx)
3. **Automated Benchmark Runner (Student 2 Day 2):**
   - CLI Tool: [`tool/benchmark_runner.py`](../tool/benchmark_runner.py)
4. **Technical Logs:**
   - Technical Setup Log: [`docs/technical_setup_log.md`](./technical_setup_log.md)
   - Daily Project Log: [`docs/daily_log.md`](./daily_log.md)
5. **Evidence Screenshots:**
   - **Screenshot 1 — Hardware Specs & Ollama Verification:** [`docs/evidence/screenshot_01_hardware_and_ollama.png`](./evidence/screenshot_01_hardware_and_ollama.png)
   - **Screenshot 2 — Phi 4 Mini Reasoning Inference & 75–89 tok/s:** [`docs/evidence/screenshot_02_phi4_reasoning_inference.png`](./evidence/screenshot_02_phi4_reasoning_inference.png)
   - **Screenshot 3 — GPU Acceleration Proof (100% GPU Offload & Active VRAM):** [`docs/evidence/screenshot_03_gpu_acceleration_proof.png`](./evidence/screenshot_03_gpu_acceleration_proof.png)
6. **Machine-Readable Verification Data:**
   - Phi-4 Verification: [`data/gpu_verification_result.json`](../data/gpu_verification_result.json)
   - 3-Model Smoke Test: [`data/three_models_smoke_test.json`](../data/three_models_smoke_test.json)

