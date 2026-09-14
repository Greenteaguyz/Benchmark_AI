# Hardware & Software Specifications Sheet
*Generated on: 2026-09-14 10:23:22*  
*Conforms to: Student Internship Project Plan (Page 2 & 5: Computer information to record)*

---

## 1. Operating System
- **OS Name & Version:** Windows 11 (Build 10.0.26200)
- **Architecture:** AMD64 (64-bit)

## 2. GPU & VRAM State
- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **Driver Version:** 616.92
- **Available VRAM:** 7.96 GB (8151 MB)

```text
Mon Sep 14 10:23:22 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 616.92                 KMD Version: 616.92        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   52C    P2             20W /   80W |     142MiB /   8151MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           20804    C+G   ...Browser\Application\brave.exe      N/A      |
|    0   N/A  N/A           27712    C+G   ...Browser\Application\brave.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

## 3. CPU & System Memory
- **Processor:** Intel(R) Core(TM) i7-14700HX
- **Total System RAM:** 23.73 GB

## 4. Ollama Runtime
- **Ollama Version:** 0.34.0

## 5. Model Storage & Disk Space
- **Storage Drive & Path:** `C:\Users\User\.ollama\models`
- **Available Disk Space:** C: (84.1 GB free of 475.82 GB)

## 6. Selected Models & Download Sizes 
| Package (Tag) | Alias | Parameters | Size (Disk) | Purpose in Benchmark | Official Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `phi4-mini-reasoning` | `PHI` | 3.84B | ~3.2 GB | Lightweight reasoning baseline for constrained hardware | [Ollama Library](https://ollama.com/library/phi4-mini-reasoning:latest) |
| `deepseek-r1:7b` | `DSR1` | 7.62B | ~4.7 GB | Compact distilled reasoning model | [Ollama Library](https://ollama.com/library/deepseek-r1:7b) |
| `qwen3:8b` | `QWEN` | 8.19B | ~5.2 GB | Larger general reasoning comparison model | [Ollama Library](https://ollama.com/library/qwen3:8b) |
