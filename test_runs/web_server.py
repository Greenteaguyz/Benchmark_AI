"""
High-Performance Local LLM Benchmarking Backend Server
Powered by Starlette & Uvicorn.
Compliant with: KiTH AI Research Internship Project Plan (8 GB VRAM Study)
"""

import os
import re
import json
import time
import asyncio
import subprocess
from typing import AsyncGenerator

import requests
import pandas as pd
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

# --- 1. Constants & Directory Paths ---
OLLAMA_API_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")

REQUIRED_MODELS = {
    "phi4-mini-reasoning": {
        "alias": "PHI",
        "tag": "phi4-mini-reasoning:latest",
        "size": "~3.2 GB",
        "purpose": "Lightweight reasoning baseline for constrained hardware",
    },
    "deepseek-r1:7b": {
        "alias": "DSR1",
        "tag": "deepseek-r1:7b",
        "size": "~4.7 GB",
        "purpose": "Compact distilled reasoning model",
    },
    "qwen3:8b": {
        "alias": "QWEN",
        "tag": "qwen3:8b",
        "size": "~5.2 GB",
        "purpose": "Larger general reasoning comparison model",
    },
}

FROZEN_SETTINGS = {
    "temperature": 0.0,
    "num_ctx": 4096,
    "num_predict": 1024,
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "test_runs" else CURRENT_DIR

OFFICIAL_RESPONSES_DIR = os.path.join(PROJECT_ROOT, "data", "responses")
OFFICIAL_LOG_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "comparison_log.csv")

TEST_RESPONSES_DIR = os.path.join(PROJECT_ROOT, "test_runs", "responses")
TEST_LOG_CSV_PATH = os.path.join(PROJECT_ROOT, "test_runs", "comparison_log.csv")

WEB_DIR = os.path.join(CURRENT_DIR, "web")

BENCHMARK_QUESTIONS = {
    "Q01": {
        "category": "Mathematical reasoning",
        "difficulty": "Easy",
        "prompt": "Solve for x: 3x + 12 = 30. Show each step of your calculation clearly and state the final numerical value of x.",
    },
    "Q02": {
        "category": "Mathematical reasoning",
        "difficulty": "Easy",
        "prompt": "A store offers a 20% discount on an item originally priced at $80. What is the final sale price? Show your work.",
    },
    "Q03": {
        "category": "Mathematical reasoning",
        "difficulty": "Medium",
        "prompt": "Calculate the total number of trailing zeroes in 100! (100 factorial). Explain the mathematical reasoning behind your method.",
    },
    "Q04": {
        "category": "Mathematical reasoning",
        "difficulty": "Medium",
        "prompt": "Find the next two numbers in the sequence: 2, 6, 12, 20, 30, ... Explain the underlying rule governing the pattern.",
    },
    "Q05": {
        "category": "Mathematical reasoning",
        "difficulty": "Hard",
        "prompt": "Three fair standard six-sided dice are rolled simultaneously. What is the exact probability that the sum of the numbers rolled is equal to 10? Express your answer as a simplified fraction and explain each step.",
    },
    "Q06": {
        "category": "Logical reasoning",
        "difficulty": "Easy",
        "prompt": "A farmer has 17 sheep, and all but 9 die. How many live sheep does the farmer have left? State your answer and briefly explain the logic.",
    },
    "Q07": {
        "category": "Logical reasoning",
        "difficulty": "Easy",
        "prompt": "All roses are flowers. Some flowers fade quickly. Does it strictly follow that some roses fade quickly? Answer with 'Yes' or 'No' and provide a formal logical justification.",
    },
    "Q08": {
        "category": "Logical reasoning",
        "difficulty": "Medium",
        "prompt": "Alice, Bob, and Charlie are sitting in a row. Alice is not on the far right. Bob is sitting to the immediate right of Alice. Who is sitting on the far left? Deduce the exact left-to-right seating arrangement step by step.",
    },
    "Q09": {
        "category": "Logical reasoning",
        "difficulty": "Medium",
        "prompt": "You have a 3-liter jug and a 5-liter jug, and an unlimited supply of water. How can you measure out exactly 4 liters using only these two jugs without any measuring scales? Detail every pour step.",
    },
    "Q10": {
        "category": "Logical reasoning",
        "difficulty": "Hard",
        "prompt": "On an island, inhabitants are either Knights (who always tell the truth) or Knaves (who always lie). You meet two inhabitants, A and B. A says: 'At least one of us is a Knave.' Determine precisely what A is and what B is. Provide a rigorous proof by cases.",
    },
    "Q11": {
        "category": "Programming reasoning",
        "difficulty": "Easy",
        "prompt": "Write a clean Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring non-alphanumeric characters and case. Provide a brief trace of why it works.",
    },
    "Q12": {
        "category": "Programming reasoning",
        "difficulty": "Easy",
        "prompt": "Trace the execution of this Python code snippet and determine the exact printed output:\n\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x), x == y)\n\nExplain why `len(x)` is affected.",
    },
    "Q13": {
        "category": "Programming reasoning",
        "difficulty": "Medium",
        "prompt": "Identify the bug in the following binary search implementation and explain how to fix it:\n\ndef binary_search(arr, target):\n    low = 0\n    high = len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid - 1\n    return -1",
    },
    "Q14": {
        "category": "Programming reasoning",
        "difficulty": "Medium",
        "prompt": "Describe Floyd's Cycle-Finding Algorithm (Tortoise and Hare) for detecting a loop in a singly linked list. Explain why the time complexity is O(N) and the auxiliary space complexity is O(1).",
    },
    "Q15": {
        "category": "Programming reasoning",
        "difficulty": "Hard",
        "prompt": "Propose an optimal algorithm in Python to find the Longest Increasing Subsequence (LIS) in O(N log N) time complexity using patience sorting / binary search. Provide the complete code, step-by-step invariant explanation, and complexity analysis.",
    },
}

# --- 2. Helper Functions ---
def wait_for_model_evicted(model_name: str, timeout: float = 20.0) -> bool:
    """Polls Ollama /api/ps until a model is fully released from VRAM."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{OLLAMA_API_BASE}/api/ps", timeout=2)
            if r.status_code == 200:
                loaded = [m["name"] for m in r.json().get("models", [])]
                if not any(
                    model_name == m or m.startswith(f"{model_name}:")
                    for m in loaded
                ):
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def evict_model(model_name: str, wait: bool = True) -> bool:
    """Sends keep_alive: 0 and waits for the model to be unmapped from VRAM."""
    try:
        requests.post(f"{OLLAMA_API_BASE}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=5)
        if wait:
            wait_for_model_evicted(model_name)
        return True
    except Exception:
        return False


def evict_all_models(wait: bool = True) -> None:
    """Evicts every model currently resident in VRAM."""
    try:
        r = requests.get(f"{OLLAMA_API_BASE}/api/ps", timeout=2)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                evict_model(m["name"], wait=wait)
    except Exception:
        pass


def get_peak_vram_mb() -> float:
    """Reads current GPU VRAM utilization via nvidia-smi."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return float(res.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0.0


def parse_reasoning_and_answer(raw_text: str):
    """Separates <think> reasoning process from verified final answer."""
    think_match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    if think_match:
        thought = think_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        return thought, answer
    return None, raw_text.strip()


def get_response_filename(qid: str, model_name: str) -> str:
    alias = REQUIRED_MODELS.get(model_name, {}).get("alias", "MODEL")
    return f"{qid}_{alias}.json"


def save_response_artifact(qid: str, model_name: str, prompt: str, response_text: str, target_dir: str) -> str:
    """Saves strict JSON artifact conforming to docs/response_schema.json."""
    os.makedirs(target_dir, exist_ok=True)
    filename = get_response_filename(qid, model_name)
    filepath = os.path.join(target_dir, filename)

    payload = {
        "question_id": qid,
        "model": model_name,
        "prompt": prompt,
        "response": response_text,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return filepath


def log_comparison_csv(record: dict, csv_path: str):
    """Appends benchmark measurement record to comparison_log.csv."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([record])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)


# --- 3. API Handlers ---
async def api_status(request):
    """Returns Ollama connectivity, installed models, and models currently active in VRAM."""
    online = False
    installed = []
    loaded_in_vram = []

    try:
        r = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=2)
        if r.status_code == 200:
            online = True
            installed = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    if online:
        try:
            r2 = requests.get(f"{OLLAMA_API_BASE}/api/ps", timeout=2)
            if r2.status_code == 200:
                loaded_in_vram = [m["name"] for m in r2.json().get("models", [])]
        except Exception:
            pass

    current_vram = get_peak_vram_mb()

    return JSONResponse({
        "online": online,
        "installed_models": installed,
        "loaded_in_vram": loaded_in_vram,
        "current_vram_mb": current_vram,
        "models": REQUIRED_MODELS,
        "frozen_settings": FROZEN_SETTINGS,
    })


async def api_metadata(request):
    """Returns benchmark metadata: questions, models, and rubric."""
    return JSONResponse({
        "models": REQUIRED_MODELS,
        "questions": BENCHMARK_QUESTIONS,
        "frozen_settings": FROZEN_SETTINGS,
    })


async def api_progress(request):
    """Scans official and test response directories and returns matrix data."""
    total_target = 15 * len(REQUIRED_MODELS)
    official_count = 0
    official_matrix = []
    test_matrix = []

    for qid in sorted(BENCHMARK_QUESTIONS.keys()):
        off_row = {"qid": qid, "category": BENCHMARK_QUESTIONS[qid]["category"]}
        test_row = {"qid": qid, "category": BENCHMARK_QUESTIONS[qid]["category"]}

        for model_key, info in REQUIRED_MODELS.items():
            fn = get_response_filename(qid, model_key)
            off_fp = os.path.join(OFFICIAL_RESPONSES_DIR, fn)
            test_fp = os.path.join(TEST_RESPONSES_DIR, fn)

            has_off = os.path.exists(off_fp)
            has_test = os.path.exists(test_fp)

            if has_off:
                official_count += 1

            off_row[info["alias"]] = has_off
            test_row[info["alias"]] = has_test

        official_matrix.append(off_row)
        test_matrix.append(test_row)

    return JSONResponse({
        "total_target": total_target,
        "official_count": official_count,
        "official_percentage": round((official_count / total_target) * 100, 1),
        "official_matrix": official_matrix,
        "test_matrix": test_matrix,
    })


async def api_vram_load(request):
    """Pre-loads a model into VRAM without sending a prompt.

    Evicts any other resident model first (to avoid 8 GB VRAM OOM) and retries
    transient connection drops caused by a crashed runner.
    """
    data = await request.json()
    model_name = data.get("model", "")
    if not model_name:
        return JSONResponse({"success": False, "error": "Missing model"}, status_code=400)

    evict_all_models(wait=True)

    max_attempts = 4
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(f"{OLLAMA_API_BASE}/api/generate", json={"model": model_name, "keep_alive": "10m"}, timeout=120)
            if r.status_code == 200:
                return JSONResponse({"success": True, "message": f"Loaded {model_name} into VRAM"})
            return JSONResponse({"success": False, "error": r.text}, status_code=500)
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            await asyncio.sleep(2 * attempt)
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    return JSONResponse({
        "success": False,
        "error": f"{last_err}. The Ollama runner likely crashed while loading {model_name} "
                 f"(usually VRAM exhaustion on an 8 GB GPU). Ensure VRAM is free, then retry. "
                 f"Check the `ollama serve` logs for a CUDA OOM error.",
    }, status_code=500)


async def api_vram_evict(request):
    """Evicts a model from VRAM."""
    data = await request.json()
    model_name = data.get("model", "")
    if not model_name:
        return JSONResponse({"success": False, "error": "Missing model"}, status_code=400)

    try:
targets = [model_name]
        if ":" not in model_name:
            targets.append(f"{model_name}:latest")
        else:
            targets.append(model_name.split(":")[0])

        for t in dict.fromkeys(targets):
            evict_model(t, wait=True)
        return JSONResponse({"success": True, "message": f"Evicted {model_name} from VRAM"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


async def api_pull(request):
    """Triggers an Ollama pull request."""
    data = await request.json()
    model_name = data.get("model", "")
    if not model_name:
        return JSONResponse({"success": False, "error": "Missing model"}, status_code=400)

    try:
        r = requests.post(f"{OLLAMA_API_BASE}/api/pull", json={"name": model_name, "stream": False}, timeout=600)
        if r.status_code == 200:
            return JSONResponse({"success": True, "message": f"Successfully pulled {model_name}"})
        return JSONResponse({"success": False, "error": r.text}, status_code=500)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


async def api_generate_stream(request):
    """
    Executes inference with real-time token streaming over Server-Sent Events (SSE).
    Saves strict docs/response_schema.json artifact and comparison_log.csv on completion.
    """
    body = await request.json()
    model_name = body.get("model")
    prompt = body.get("prompt")
    question_id = body.get("question_id", "Q_TEST")
    mode = body.get("mode", "test")  # 'test' or 'official'

    is_official = mode == "official"
    target_dir = OFFICIAL_RESPONSES_DIR if is_official else TEST_RESPONSES_DIR
    target_csv = OFFICIAL_LOG_CSV_PATH if is_official else TEST_LOG_CSV_PATH

    # Hyperparameter compliance
    if is_official:
        temp = FROZEN_SETTINGS["temperature"]
        ctx = FROZEN_SETTINGS["num_ctx"]
        predict = FROZEN_SETTINGS["num_predict"]
    else:
        temp = float(body.get("temperature", 0.0))
        ctx = int(body.get("num_ctx", 4096))
        predict = int(body.get("num_predict", 1024))

    req_payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temp,
            "num_ctx": ctx,
            "num_predict": predict,
        },
    }

    start_vram = get_peak_vram_mb()
    start_time = time.time()

    async def event_generator() -> AsyncGenerator[str, None]:
        full_response = ""
        eval_count = 0
        eval_duration_ns = 0
        completion_status = "Completed"

        try:
            with requests.post(f"{OLLAMA_API_BASE}/api/generate", json=req_payload, stream=True, timeout=300) as resp:
                if resp.status_code != 200:
                    yield f"data: {json.dumps({'error': f'Ollama error {resp.status_code}: {resp.text}'})}\n\n"
                    return

                in_thinking = False
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        th = chunk.get("thinking")
                        token = chunk.get("response", "")

                        if th:
                            if not in_thinking:
                                full_response += "<think>\n"
                                in_thinking = True
                                yield f"data: {json.dumps({'token': '<think>\n', 'done': False})}\n\n"
                            full_response += th
                            yield f"data: {json.dumps({'token': th, 'done': False})}\n\n"
                        elif token:
                            if in_thinking:
                                full_response += "\n</think>\n\n"
                                in_thinking = False
                                yield f"data: {json.dumps({'token': '\n</think>\n\n', 'done': False})}\n\n"
                            full_response += token
                            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                        await asyncio.sleep(0)  # Yield control to event loop

                        if chunk.get("done", False):
                            if in_thinking:
                                full_response += "\n</think>\n\n"
                                in_thinking = False
                            eval_count = chunk.get("eval_count", 0)
                            eval_duration_ns = chunk.get("eval_duration", 0)
                            done_reason = chunk.get("done_reason", "stop")
                            completion_status = "Completed" if done_reason == "stop" else done_reason

            if not full_response.strip():
                yield f"data: {json.dumps({'error': f'Model {model_name} produced an empty response. Free VRAM and retry.'})}\n\n"
                return

            total_duration_sec = round(time.time() - start_time, 3)
            end_vram = get_peak_vram_mb()
            peak_vram = max(start_vram, end_vram)
            eval_duration_sec = round(eval_duration_ns / 1e9, 3) if eval_duration_ns else total_duration_sec
            tokens_per_sec = round(eval_count / eval_duration_sec, 2) if eval_duration_sec > 0 else 0.0

            thought, final_ans = parse_reasoning_and_answer(full_response)
            saved_filepath = save_response_artifact(question_id, model_name, prompt, full_response, target_dir)

            m_info = REQUIRED_MODELS.get(model_name, {"alias": "MODEL"})
            log_record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question_id": question_id,
                "model": model_name,
                "model_alias": m_info["alias"],
                "total_duration_s": total_duration_sec,
                "generation_duration_s": eval_duration_sec,
                "output_tokens": eval_count,
                "tokens_per_sec": tokens_per_sec,
                "peak_vram_mb": peak_vram,
                "completion_status": completion_status,
                "evidence_file": saved_filepath,
            }
            log_comparison_csv(log_record, target_csv)

            final_data = {
                "done": True,
                "raw_response": full_response,
                "thought": thought,
                "final_ans": final_ans,
                "saved_filepath": saved_filepath,
                "metrics": {
                    "total_duration_s": total_duration_sec,
                    "eval_duration_s": eval_duration_sec,
                    "output_tokens": eval_count,
                    "tokens_per_sec": tokens_per_sec,
                    "peak_vram_mb": peak_vram,
                    "completion_status": completion_status,
                }
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as err:
            yield f"data: {json.dumps({'error': str(err)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def api_logs(request):
    """Returns log entries from comparison_log.csv."""
    mode = request.query_params.get("mode", "test")
    csv_path = OFFICIAL_LOG_CSV_PATH if mode == "official" else TEST_LOG_CSV_PATH

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return JSONResponse({"success": True, "rows": df.to_dict(orient="records")})
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e), "rows": []})
    return JSONResponse({"success": True, "rows": []})


async def api_artifact(request):
    """Returns the JSON content of a specific artifact file."""
    mode = request.query_params.get("mode", "test")
    filename = request.query_params.get("file", "").strip()
    primary_dir = OFFICIAL_RESPONSES_DIR if mode == "official" else TEST_RESPONSES_DIR
    fallback_dir = TEST_RESPONSES_DIR if mode == "official" else OFFICIAL_RESPONSES_DIR

    base_fn = os.path.basename(filename)
    candidates = [
        os.path.join(primary_dir, base_fn),
        os.path.join(fallback_dir, base_fn),
        filename if os.path.isabs(filename) else os.path.join(PROJECT_ROOT, filename),
    ]

    for fp in candidates:
        if fp and os.path.exists(fp) and os.path.isfile(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    return JSONResponse(json.load(f))
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"error": f"Artifact file '{base_fn}' not found"}, status_code=404)


async def serve_index(request):
    """Serves the main HTML page."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return Response(f.read(), media_type="text/html")
    return Response("<h1>Web interface not found. Please ensure web/index.html exists.</h1>", media_type="text/html")


# --- 4. Application Routes ---
routes = [
    Route("/", serve_index),
    Route("/api/status", api_status),
    Route("/api/metadata", api_metadata),
    Route("/api/progress", api_progress),
    Route("/api/vram/load", api_vram_load, methods=["POST"]),
    Route("/api/vram/evict", api_vram_evict, methods=["POST"]),
    Route("/api/pull", api_pull, methods=["POST"]),
    Route("/api/generate", api_generate_stream, methods=["POST"]),
    Route("/api/logs", api_logs),
    Route("/api/artifact", api_artifact),
]

os.makedirs(WEB_DIR, exist_ok=True)
routes.append(Mount("/static", StaticFiles(directory=WEB_DIR), name="static"))

app = Starlette(debug=True, routes=routes)

if __name__ == "__main__":
    import uvicorn
    print("=" * 65)
    print("  [+] LLM Reasoning Benchmark Fast Web Server")
    print("  URL: http://localhost:8000")
    print("=" * 65)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
