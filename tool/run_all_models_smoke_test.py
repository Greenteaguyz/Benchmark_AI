"""
Run Day 2 Smoke Test on all three required models:
- phi4-mini-reasoning:latest
- deepseek-r1:7b
- qwen3:8b

Strict adherence to Page 2 rules:
- One model loaded at a time (evicted with keep_alive: 0 before loading next)
- Context length: 4096
- Temperature: 0.0
- Max tokens: 512
- Common prompt across all three models
- Records model tag, size, initial VRAM, peak VRAM, load duration, tokens/sec
"""

import os
import json
import time
import subprocess
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "tool" else CURRENT_DIR

OLLAMA_API = "http://127.0.0.1:11434"

MODELS = [
    {
        "alias": "PHI",
        "name": "phi4-mini-reasoning:latest",
        "params": "3.84B",
        "disk_size": "~3.2 GB"
    },
    {
        "alias": "DSR1",
        "name": "deepseek-r1:7b",
        "params": "7.62B",
        "disk_size": "~4.7 GB"
    },
    {
        "alias": "QWEN",
        "name": "qwen3:8b",
        "params": "8.19B",
        "disk_size": "~5.2 GB"
    }
]

COMMON_PROMPT = (
    "Solve this step-by-step:\n"
    "A farmer has 15 sheep. All but 8 die. How many sheep does the farmer have left?\n"
    "Explain your reasoning concisely."
)

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", check=False)
        return res.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def wait_for_model_evicted(model_name: str, timeout: float = 20.0) -> bool:
    """Polls Ollama /api/ps until a model is fully released from VRAM.

    Ollama's keep_alive: 0 request returns before the model is actually
    unmapped from GPU memory. Loading the next model during that window can
    exceed the 8 GB VRAM budget and crash the runner.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            loaded = [m.get("name", "") for m in requests.get(f"{OLLAMA_API}/api/ps").json().get("models", [])]
        except Exception:
            loaded = []
        if not any(model_name == m or m.startswith(f"{model_name}:") for m in loaded):
            return True
        time.sleep(0.5)
    return False


def evict_all_models():
    """Ensure no models remain in VRAM, waiting for each to be released."""
    try:
        ps_res = requests.get(f"{OLLAMA_API}/api/ps").json().get("models", [])
    except Exception:
        return

    for m in ps_res:
        model_name = m.get("name")
        if not model_name:
            continue
        print(f"  Evicting {model_name} from VRAM...")
        try:
            requests.post(f"{OLLAMA_API}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=5)
            wait_for_model_evicted(model_name)
        except Exception as e:
            print(f"  Warning: evict failed for {model_name}: {e}")


def run_generate_with_retry(payload: dict, timeout: int = 180, attempts: int = 4):
    """Calls /api/generate, retrying transient connection drops.

    RemoteDisconnected / ConnectionError means the Ollama runner was killed
    (usually VRAM exhaustion on the 8 GB GPU). We clear VRAM, wait, and retry.
    """
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(f"{OLLAMA_API}/api/generate", json=payload, timeout=timeout)
            if resp.status_code != 200:
                print(f"  ERROR: {resp.status_code} - {resp.text}")
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            print(f"  Connection dropped ({type(e).__name__}) — likely runner crash on 8 GB VRAM OOM. "
                  f"Clearing VRAM and retrying ({attempt}/{attempts})...")
            evict_all_models()
            time.sleep(2 * attempt)
        except Exception as e:
            print(f"  ERROR during request: {e}")
            return None
    return None

def get_vram_used_mb():
    try:
        out = run_cmd(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return int(out.splitlines()[0].strip())
    except:
        return 0

def main():
    print("=" * 70)
    print("  Day 2 Deliverable: 3-Model Controlled Smoke Test on 8 GB VRAM")
    print("=" * 70)

    results = []

    for item in MODELS:
        model_name = item["name"]
        alias = item["alias"]
        print(f"\n>>> Preparing test for [{alias}] {model_name}...")
        
        # 1. Clean slate
        evict_all_models()
        baseline_vram = get_vram_used_mb()
        print(f"  Baseline idle VRAM: {baseline_vram} MB")

        # 2. Run inference
        payload = {
            "model": model_name,
            "prompt": COMMON_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
                "num_predict": 512
            },
            "keep_alive": "5m"
        }

        print(f"  Running common prompt on {model_name}...")
        start_time = time.time()
        resp = run_generate_with_retry(payload, timeout=180)

        if resp is None:
            print(f"  FAILED to run {model_name} after retries (runner crash / VRAM OOM). Skipping.")
            continue

        elapsed = time.time() - start_time

        if resp.status_code != 200:
            print(f"  ERROR: {resp.status_code} - {resp.text}")
            continue

        data = resp.json()
        peak_vram = get_vram_used_mb()
        allocated_vram = peak_vram - baseline_vram

        # 3. Check ollama ps for GPU offload
        ps_info = requests.get(f"{OLLAMA_API}/api/ps").json().get("models", [])
        gpu_processor = "Unknown"
        vram_size_gb = 0
        for m in ps_info:
            if m.get("name") == model_name or model_name.startswith(m.get("name", "")):
                vram_bytes = m.get("size_vram", 0)
                vram_size_gb = round(vram_bytes / (1024**3), 2)
                total_size = m.get("size", 1)
                gpu_pct = round((vram_bytes / max(total_size, 1)) * 100, 1)
                gpu_processor = f"{gpu_pct}% GPU ({vram_size_gb} GB)"

        load_dur = round(data.get("load_duration", 0) / 1e9, 3)
        prompt_tokens = data.get("prompt_eval_count", 0)
        eval_tokens = data.get("eval_count", 0)
        eval_dur = data.get("eval_duration", 0) / 1e9
        tps = round(eval_tokens / eval_dur, 2) if eval_dur > 0 else 0

        print(f"  [+] Success! Load: {load_dur}s | Eval: {eval_tokens} tokens in {eval_dur:.2f}s ({tps} tok/s)")
        print(f"  [+] VRAM Used: {peak_vram} MB (Allocated: ~{allocated_vram} MB) | Offload: {gpu_processor}")

        record = {
            "alias": alias,
            "model": model_name,
            "parameters": item["params"],
            "disk_size": item["disk_size"],
            "load_duration_s": load_dur,
            "prompt_tokens": prompt_tokens,
            "eval_tokens": eval_tokens,
            "eval_duration_s": round(eval_dur, 2),
            "tokens_per_second": tps,
            "peak_vram_mb": peak_vram,
            "allocated_vram_mb": allocated_vram,
            "gpu_offload": gpu_processor,
            "total_turnaround_s": round(elapsed, 2),
            "response_snippet": data.get("response", "")[:250].replace("\n", " ") + "..."
        }
        results.append(record)

    # Save to json in data/
    out_json = os.path.join(PROJECT_ROOT, "data", "three_models_smoke_test.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved complete 3-model smoke test results to: {out_json}")

    # Evict to leave GPU clean
    evict_all_models()

    return results

if __name__ == "__main__":
    main()
