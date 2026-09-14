# Selected Reasoning Models & Architecture Summary

**Project:** Benchmarking Open Source Reasoning LLMs on an 8 GB VRAM Computer  
**Authors:** Hout Chanvireak (Benchmark Lead) & Sok Ratanakvichea (Deployment Lead)  
**Deliverable:** Student 1 — Day 1 Model Summary  

---

## 1. Overview of Evaluated Models

The benchmark evaluates three modern, open-weight reasoning large language models operating under a constrained **8 GB VRAM budget**. Each model was chosen for its specific architectural approach to reasoning (chain-of-thought, reinforcement learning distillation, and instruction tuning) and its ability to fit entirely within consumer-grade GPU memory using 4-bit quantization (`Q4_K_M`).

| Model Display Name | Ollama Package Tag | Architecture | Quantization | Parameter Count | Disk Size | Target VRAM | Role in 8 GB Study | Official Library Citation |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Phi 4 Mini Reasoning** | `phi4-mini-reasoning:latest` | Phi-3 / Phi-4 Transformer | Q4_K_M | **3.84B** | ~3.2 GB | ~3.9 GB | Lightweight reasoning baseline for ultra-constrained hardware | [Ollama: phi4-mini-reasoning](https://ollama.com/library/phi4-mini-reasoning) |
| **DeepSeek R1 Distill Qwen 7B** | `deepseek-r1:7b` | Qwen2 Architecture | Q4_K_M | **7.62B** | ~4.7 GB | ~4.9 GB | Compact distilled reasoning model utilizing RL-derived chains | [Ollama: deepseek-r1:7b](https://ollama.com/library/deepseek-r1:7b) |
| **Qwen3 8B** | `qwen3:8b` | Qwen3 Dense Transformer | Q4_K_M | **8.19B** | ~5.2 GB | ~5.7 GB | Larger general reasoning comparison model testing upper memory limits | [Ollama: qwen3:8b](https://ollama.com/library/qwen3:8b) |

---

## 2. Detailed Model Profiles

### 2.1 Phi 4 Mini Reasoning (`phi4-mini-reasoning`)
- **Developer:** Microsoft Research
- **Parameters:** 3.84 Billion
- **Context Window:** Up to 131,072 tokens (Tested with 4,096 tokens)
- **Quantization:** GGUF `Q4_K_M`
- **Reasoning Architecture:** Employs built-in synthetic reasoning tokens wrapped in `<think> ... </think>` delimiters, optimized for high token throughput and small memory footprint.
- **Why Selected:** Serves as the minimal hardware baseline. With an in-memory allocation of under 4.0 GB VRAM, it leaves over 50% of the 8 GB budget free, demonstrating feasibility on entry-level laptop GPUs.

### 2.2 DeepSeek R1 Distill Qwen 7B (`deepseek-r1:7b`)
- **Developer:** DeepSeek AI
- **Parameters:** 7.62 Billion
- **Context Window:** Up to 131,072 tokens (Tested with 4,096 tokens)
- **Quantization:** GGUF `Q4_K_M`
- **Reasoning Architecture:** Fine-tuned on reasoning trajectories distilled from the 671B DeepSeek-R1 reinforcement learning model into the Qwen 2.5 7B base. Generates comprehensive internal reflection before producing final boxed answers.
- **Why Selected:** Evaluates whether distillation from frontier RL models retains high-level mathematical and programming reasoning when running within ~4.9 GB of VRAM.

### 2.3 Qwen3 8B (`qwen3:8b`)
- **Developer:** Alibaba Cloud / Qwen Team
- **Parameters:** 8.19 Billion
- **Context Window:** Up to 40,960 tokens (Tested with 4,096 tokens)
- **Quantization:** GGUF `Q4_K_M`
- **Reasoning Architecture:** High-capacity general dense model with advanced instruction-following, math, and code capabilities.
- **Why Selected:** Represents the practical upper limit for 8 GB VRAM GPUs. Occupying ~5.7 GB of VRAM, it tests whether the additional ~0.5–1.0 GB memory overhead translates into measurably superior reasoning accuracy compared to the smaller 3.8B and 7.6B models.

---

## 3. Controlled Inference Environment & Constraints

To ensure reproducible and scientifically valid comparisons, all three models are executed strictly under the **Frozen Settings** specified on Page 2 of the Project Plan:

1. **Hardware Parity:** All tests run on the identical physical machine (Intel Core i7-14700HX, NVIDIA RTX 5060 Laptop GPU, Windows 11).
2. **Single Model Residency:** Only one model is loaded into VRAM at any given time. Models are explicitly evicted (`keep_alive: 0`) between questions to prevent memory fragmentation and context pollution.
3. **Deterministic Decoding:** `temperature: 0.0` across all prompts to guarantee reproducibility.
4. **Context Length:** Fixed at `num_ctx: 4096`.
5. **Output Limit:** Fixed at `num_predict: 1024` tokens.
6. **No Context Reuse:** Each inference request is issued with a fresh, isolated state (no multi-turn history).
