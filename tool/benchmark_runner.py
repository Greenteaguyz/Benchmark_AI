"""
Automated Reasoning LLM Benchmark Runner (CLI Version)
Compliant with: KiTH AI Research Internship Project Plan (Page 5 & Page 6)

Features:
- Calls local Ollama REST API (http://127.0.0.1:11434)
- Enforces frozen experimental settings (temp=0.0, num_ctx=4096, num_predict=1024)
- Enforces single model residency in VRAM (evicts before and after run)
- Saves full responses to data/responses/Q{ID}_{MODEL}.json strictly adhering to response_schema.json
- Appends run telemetry to data/comparison_log.csv
"""

import os
import sys
import json
import time
import argparse
import subprocess
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "tool" else CURRENT_DIR

OLLAMA_API = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
RESPONSES_DIR = os.path.join(PROJECT_ROOT, "data", "responses")
LOG_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "comparison_log.csv")

os.makedirs(RESPONSES_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "PHI": {
        "tag": "phi4-mini-reasoning:latest",
        "name": "Phi 4 Mini Reasoning",
        "params": "3.84B"
    },
    "DSR1": {
        "tag": "deepseek-r1:7b",
        "name": "DeepSeek R1 Distill Qwen 7B",
        "params": "7.62B"
    },
    "QWEN": {
        "tag": "qwen3:8b",
        "name": "Qwen3 8B",
        "params": "8.19B"
    }
}

BENCHMARK_QUESTIONS = {
    "Q01": {"cat": "Mathematical reasoning", "diff": "Easy", "prompt": "Solve for x: 3x + 12 = 30. Show each step of your calculation clearly and state the final numerical value of x."},
    "Q02": {"cat": "Mathematical reasoning", "diff": "Easy", "prompt": "A store offers a 20% discount on an item originally priced at $80. What is the final sale price? Show your work."},
    "Q03": {"cat": "Mathematical reasoning", "diff": "Medium", "prompt": "Calculate the total number of trailing zeroes in 100! (100 factorial). Explain the mathematical reasoning behind your method."},
    "Q04": {"cat": "Mathematical reasoning", "diff": "Medium", "prompt": "Find the next two numbers in the sequence: 2, 6, 12, 20, 30, ... Explain the underlying rule governing the pattern."},
    "Q05": {"cat": "Mathematical reasoning", "diff": "Hard", "prompt": "Three fair standard six-sided dice are rolled simultaneously. What is the exact probability that the sum of the numbers rolled is equal to 10? Express your answer as a simplified fraction and explain each step."},
    "Q06": {"cat": "Logical reasoning", "diff": "Easy", "prompt": "A farmer has 17 sheep, and all but 9 die. How many live sheep does the farmer have left? State your answer and briefly explain the logic."},
    "Q07": {"cat": "Logical reasoning", "diff": "Easy", "prompt": "All roses are flowers. Some flowers fade quickly. Does it strictly follow that some roses fade quickly? Answer with 'Yes' or 'No' and provide a formal logical justification."},
    "Q08": {"cat": "Logical reasoning", "diff": "Medium", "prompt": "Alice, Bob, and Charlie are sitting in a row. Alice is not on the far right. Bob is sitting to the immediate right of Alice. Who is sitting on the far left? Deduce the exact left-to-right seating arrangement step by step."},
    "Q09": {"cat": "Logical reasoning", "diff": "Medium", "prompt": "You have a 3-liter jug and a 5-liter jug, and an unlimited supply of water. How can you measure out exactly 4 liters using only these two jugs without any measuring scales? Detail every pour step."},
    "Q10": {"cat": "Logical reasoning", "diff": "Hard", "prompt": "On an island, inhabitants are either Knights (who always tell the truth) or Knaves (who always lie). You meet two inhabitants, A and B. A says: 'At least one of us is a Knave.' Determine precisely what A is and what B is. Provide a rigorous proof by cases."},
    "Q11": {"cat": "Programming reasoning", "diff": "Easy", "prompt": "Write a clean Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring non-alphanumeric characters and case. Provide a brief trace of why it works."},
    "Q12": {"cat": "Programming reasoning", "diff": "Easy", "prompt": "Trace the execution of this Python code snippet and determine the exact printed output:\n\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x), x == y)\n\nExplain why `len(x)` is affected."},
    "Q13": {"cat": "Programming reasoning", "diff": "Medium", "prompt": "Identify the bug in the following binary search implementation and explain how to fix it:\n\ndef binary_search(arr, target):\n    low = 0\n    high = len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid - 1\n    return -1"},
    "Q14": {"cat": "Programming reasoning", "diff": "Medium", "prompt": "Explain the time and space complexity of computing the Nth Fibonacci number using naive recursion versus dynamic programming (memoization). Show the recurrence relations and dynamic state array."},
    "Q15": {"cat": "Programming reasoning", "diff": "Hard", "prompt": "Design an algorithm to find the longest substring without repeating characters in a given string `s`. State the time and space complexities, and provide a clean Python implementation with an inline trace."}
}

def get_peak_vram_gb():
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, text=True, check=False)
        mb = float(res.stdout.strip().splitlines()[0])
        return round(mb / 1024, 2)
    except Exception:
        return 0.0

def evict_models():
    try:
        ps = requests.get(f"{OLLAMA_API}/api/ps", timeout=5).json().get("models", [])
        for m in ps:
            requests.post(f"{OLLAMA_API}/api/generate", json={"model": m.get("name"), "keep_alive": 0}, timeout=5)
    except Exception:
        pass

def run_benchmark_item(qid: str, model_alias: str) -> dict:
    if qid not in BENCHMARK_QUESTIONS:
        raise ValueError(f"Unknown question ID: {qid}")
    if model_alias not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model alias: {model_alias}. Must be one of: {list(MODEL_CONFIGS.keys())}")

    q_info = BENCHMARK_QUESTIONS[qid]
    m_info = MODEL_CONFIGS[model_alias]
    model_tag = m_info["tag"]

    print(f"\n[*] Executing {qid} on {model_alias} ({model_tag})...")

    # Clean VRAM before running
    evict_models()
    time.sleep(1)

    payload = {
        "model": model_tag,
        "prompt": q_info["prompt"],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096,
            "num_predict": 1024
        },
        "keep_alive": 0  # Evict immediately after generation
    }

    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    start_time = time.time()
    resp = requests.post(f"{OLLAMA_API}/api/generate", json=payload, timeout=300)
    wall_elapsed = time.time() - start_time

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API error ({resp.status_code}): {resp.text}")

    data = resp.json()
    peak_vram = get_peak_vram_gb()

    total_dur_s = round(data.get("total_duration", 0) / 1e9, 3)
    gen_dur_s = round(data.get("eval_duration", 0) / 1e9, 3)
    out_tokens = data.get("eval_count", 0)
    tps = round(out_tokens / gen_dur_s, 2) if gen_dur_s > 0 else 0.0

    response_payload = {
        "response_id": f"{qid}_{model_alias}",
        "question_id": qid,
        "category": q_info["cat"],
        "difficulty": q_info["diff"],
        "model_name": m_info["name"],
        "model_tag": model_tag,
        "parameters": m_info["params"],
        "prompt": q_info["prompt"],
        "response": data.get("response", ""),
        "timestamp": start_iso,
        "status": "Completed",
        "total_duration_s": total_dur_s if total_dur_s > 0 else round(wall_elapsed, 3),
        "generation_duration_s": gen_dur_s,
        "output_tokens": out_tokens,
        "tokens_per_second": tps,
        "peak_vram_gb": peak_vram,
        "evaluation": {
            "final_answer_score": None,
            "reasoning_quality_score": None,
            "instruction_following_score": None,
            "factual_support_score": None,
            "total_score": None,
            "evaluator_comments": ""
        },
        "verification": {
            "status": "Pending",
            "verified_by": None,
            "notes": ""
        }
    }

    # Save to data/responses/Q{ID}_{MODEL}.json
    filename = f"{qid}_{model_alias}.json"
    file_path = os.path.join(RESPONSES_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(response_payload, f, indent=2)

    # Append to data/comparison_log.csv
    csv_header = "timestamp,question_id,model,model_alias,total_duration_s,generation_duration_s,output_tokens,tokens_per_sec,peak_vram_gb,completion_status,evidence_file\n"
    csv_exists = os.path.exists(LOG_CSV_PATH)
    with open(LOG_CSV_PATH, "a", encoding="utf-8") as f:
        if not csv_exists:
            f.write(csv_header)
        f.write(f"{start_iso},{qid},{m_info['name']},{model_alias},{response_payload['total_duration_s']},{gen_dur_s},{out_tokens},{tps},{peak_vram},{response_payload['status']},{file_path}\n")

    print(f"  [+] Saved response to: {file_path}")
    print(f"  [+] Telemetry logged to: {LOG_CSV_PATH}")
    print(f"  [+] Metrics: {out_tokens} tokens in {gen_dur_s}s ({tps} tok/s) | Peak VRAM: {peak_vram} GB")

    return response_payload

def main():
    parser = argparse.ArgumentParser(description="Run local LLM benchmark inference and record response.")
    parser.add_argument("--qid", type=str, default="Q01", help="Question ID (e.g. Q01 to Q15)")
    parser.add_argument("--model", type=str, default="PHI", choices=["PHI", "DSR1", "QWEN", "ALL"], help="Model alias (PHI, DSR1, QWEN, or ALL)")
    args = parser.parse_args()

    target_models = list(MODEL_CONFIGS.keys()) if args.model.upper() == "ALL" else [args.model.upper()]

    for m in target_models:
        run_benchmark_item(args.qid.upper(), m)

if __name__ == "__main__":
    main()
