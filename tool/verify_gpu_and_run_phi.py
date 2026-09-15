import os
import json
import time
import subprocess
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "tool" else CURRENT_DIR

OLLAMA_API = "http://127.0.0.1:11434"
MODEL = "phi4-mini-reasoning:latest"

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", check=False)
        return res.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    print("=" * 60)
    print("STEP 1: Verify Ollama Service & Models")
    print("=" * 60)
    tags_res = requests.get(f"{OLLAMA_API}/api/tags")
    print(f"Ollama API Status: {tags_res.status_code}")
    models = tags_res.json().get("models", [])
    phi_found = any("phi4-mini-reasoning" in m["name"] for m in models)
    print(f"Phi 4 Mini Reasoning available: {phi_found}")

    print("\n" + "=" * 60)
    print("STEP 2: Initial GPU State (nvidia-smi)")
    print("=" * 60)
    gpu_initial = run_cmd(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu", "--format=csv,noheader,nounits"])
    print(f"Initial GPU State [Util, Mem Used MB, Mem Free MB, Temp C]: {gpu_initial}")

    print("\n" + "=" * 60)
    print(f"STEP 3: Running Inference on {MODEL} with Reasoning Test")
    print("=" * 60)
    prompt = (
        "Solve this step-by-step:\n"
        "A farmer has 15 sheep. All but 8 die. How many sheep does the farmer have left?\n"
        "Explain your reasoning concisely."
    )
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 512
        }
    }

    start_time = time.time()
    resp = requests.post(f"{OLLAMA_API}/api/generate", json=payload, timeout=120)
    elapsed = time.time() - start_time

    if resp.status_code != 200:
        print(f"Error during inference: {resp.status_code} - {resp.text}")
        return

    data = resp.json()
    response_text = data.get("response", "")
    total_duration = data.get("total_duration", 0) / 1e9
    load_duration = data.get("load_duration", 0) / 1e9
    prompt_eval_count = data.get("prompt_eval_count", 0)
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 0) / 1e9
    tok_per_sec = eval_count / eval_duration if eval_duration > 0 else 0

    print(f"\nResponse Received in {elapsed:.2f}s (Total internal duration: {total_duration:.2f}s)")
    print(f"Model Load Duration: {load_duration:.3f}s")
    print(f"Prompt Tokens: {prompt_eval_count}")
    print(f"Generated Tokens: {eval_count} in {eval_duration:.2f}s ({tok_per_sec:.2f} tokens/sec)")
    print("\n--- Model Response ---")
    print(response_text)
    print("----------------------")

    print("\n" + "=" * 60)
    print("STEP 4: Verifying GPU Acceleration (ollama ps & nvidia-smi)")
    print("=" * 60)
    ps_res = requests.get(f"{OLLAMA_API}/api/ps")
    print(f"Running models status (API):")
    for m in ps_res.json().get("models", []):
        print(f"  - Model: {m.get('name')}")
        print(f"    Size: {m.get('size') / (1024**3):.2f} GB")
        print(f"    Size VRAM: {m.get('size_vram') / (1024**3):.2f} GB ({m.get('size_vram') / max(m.get('size', 1), 1) * 100:.1f}% on GPU)")
        details = m.get("details", {})
        print(f"    Format: {details.get('format')}, Parameter Size: {details.get('parameter_size')}")

    ollama_path = r"C:\Users\User\AppData\Local\Programs\Ollama\ollama.exe"
    cli_ps = run_cmd([ollama_path, "ps"])
    print(f"\nOllama CLI `ps` Output:\n{cli_ps}")

    smi_out = run_cmd(["nvidia-smi"])
    print(f"\nnvidia-smi Output:\n{smi_out}")

    # Save results to a verification json file
    verification_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "prompt": prompt,
        "response": response_text,
        "performance": {
            "elapsed_seconds": elapsed,
            "load_duration_seconds": load_duration,
            "prompt_tokens": prompt_eval_count,
            "eval_tokens": eval_count,
            "eval_duration_seconds": eval_duration,
            "tokens_per_second": round(tok_per_sec, 2)
        },
        "running_models": ps_res.json().get("models", []),
        "ollama_ps_cli": cli_ps,
        "nvidia_smi": smi_out
    }
    output_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "gpu_verification_result.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(verification_data, f, indent=2)
    print(f"\nSaved verification data to {out_file}")

if __name__ == "__main__":
    main()
