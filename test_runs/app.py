"""
Local Reasoning LLM Benchmarking & Comparison Tool
Compliant with: KiTH AI Research Internship Project Plan
(Benchmarking Open Source Reasoning LLMs on an 8 GB VRAM Computer)
"""

import os
import re
import json
import time
import subprocess
import pandas as pd
import requests
import streamlit as st

from csv_utils import append_record_row

# --- 1. Guidelines & Controlled Experiment Constants (Page 2 & 6) ---
OLLAMA_API_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Exactly the 3 models required by the guidelines
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

# Frozen experiment settings mandated on Page 2
FROZEN_SETTINGS = {
    "temperature": 0.0,
    "num_ctx": 4096,
    "num_predict": 1024,
}

# Project Root Directory Anchor
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = (
    os.path.dirname(CURRENT_SCRIPT_DIR)
    if os.path.basename(CURRENT_SCRIPT_DIR) == "test_runs"
    else CURRENT_SCRIPT_DIR
)

# Directory Paths: Official vs Experimental/Test
OFFICIAL_RESPONSES_DIR = os.path.join(PROJECT_ROOT, "data", "responses")
OFFICIAL_LOG_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "comparison_log.csv")

TEST_RESPONSES_DIR = os.path.join(PROJECT_ROOT, "test_runs", "responses")
TEST_LOG_CSV_PATH = os.path.join(PROJECT_ROOT, "test_runs", "comparison_log.csv")

# 15 Frozen Benchmark Questions across 3 categories (Page 3)
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
    """Separates <think> reasoning process from verified answer."""
    think_match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    if think_match:
        thought = think_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        return thought, answer
    return None, raw_text.strip()


def format_latex_for_display(text: str) -> str:
    """Formats LaTeX delimiters \\[ \\] to $$ and \\( \\) to $ strictly for display rendering, without altering raw saved output."""
    if not text:
        return ""
    formatted = re.sub(r"\\\[", "$$", text)
    formatted = re.sub(r"\\\]", "$$", formatted)
    formatted = re.sub(r"\\\(", "$", formatted)
    formatted = re.sub(r"\\\)", "$", formatted)
    return formatted


@st.cache_data(ttl=5, show_spinner=False)
def check_ollama_status():
    """Verifies Ollama is running and lists installed models (cached for 5s for fast UI)."""
    try:
        r = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=2)
        if r.status_code == 200:
            data = r.json()
            installed = [m["name"] for m in data.get("models", [])]
            return True, installed
    except Exception:
        pass
    return False, []


def is_model_installed(model_name: str, installed_tags: list[str]) -> bool:
    """Checks if a model or tag is locally installed in Ollama."""
    for tag in installed_tags:
        if model_name == tag or tag.startswith(f"{model_name}:") or model_name.startswith(f"{tag.split(':')[0]}:"):
            return True
        if model_name == tag.split(":")[0]:
            return True
    return False


def evict_ollama_model(model_name: str = None) -> bool:
    """Sends keep_alive: 0 to Ollama to release model(s) from 8 GB VRAM."""
    success = False
    try:
        targets = set()
        if model_name:
            targets.add(model_name)
            if ":" not in model_name:
                targets.add(f"{model_name}:latest")
            else:
                targets.add(model_name.split(":")[0])

        # Query active models currently in VRAM to ensure exact loaded tag is targeted
        try:
            r = requests.get(f"{OLLAMA_API_BASE}/api/ps", timeout=2)
            if r.status_code == 200:
                for m in r.json().get("models", []):
                    name = m.get("name")
                    if name:
                        if not model_name or any(t in name or name in t for t in targets):
                            targets.add(name)
        except Exception:
            pass

        for target in targets:
            try:
                requests.post(
                    f"{OLLAMA_API_BASE}/api/generate",
                    json={"model": target, "keep_alive": 0},
                    timeout=5,
                )
                success = True
            except Exception:
                pass

        get_loaded_models.clear()
        return success
    except Exception:
        return False


def wait_for_model_evicted(model_name: str, timeout: float = 20.0) -> bool:
    """Polls Ollama /api/ps until a model is fully released from VRAM.

    Ollama's keep_alive: 0 request returns before the model is actually
    unmapped from GPU memory. Loading a new model during that window can
    exceed the 8 GB VRAM budget and crash the runner.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        get_loaded_models.clear()
        loaded = get_loaded_models()
        if not any(
            model_name == m or m.startswith(f"{model_name}:")
            for m in loaded
        ):
            return True
        time.sleep(0.5)
    return False


def evict_all_models(timeout_per_model: float = 20.0) -> list[str]:
    """Evicts every model currently resident in VRAM and waits for release."""
    evicted = []
    for m in get_loaded_models():
        if evict_ollama_model(m):
            wait_for_model_evicted(m, timeout=timeout_per_model)
            evicted.append(m)
    return evicted


@st.cache_data(ttl=3, show_spinner=False)
def get_loaded_models() -> list[str]:
    """Queries Ollama /api/ps to retrieve models currently active in VRAM (cached for 3s)."""
    try:
        r = requests.get(f"{OLLAMA_API_BASE}/api/ps", timeout=2)
        if r.status_code == 200:
            return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def is_model_loaded_in_vram(model_name: str, loaded_models: list[str]) -> bool:
    """Checks if a model is currently resident in GPU memory."""
    for m in loaded_models:
        if model_name == m or m.startswith(f"{model_name}:") or model_name.startswith(f"{m.split(':')[0]}:"):
            return True
        if model_name == m.split(":")[0]:
            return True
    return False


def load_model_into_vram(model_name: str) -> tuple[bool, str]:
    """Pre-loads/warms up a model into GPU memory via Ollama API without generating text.

    Retries transient connection drops (the Ollama runner is briefly killed and
    restarted on VRAM exhaustion) with linear backoff before failing.
    """
    max_attempts = 4
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(
                f"{OLLAMA_API_BASE}/api/generate",
                json={"model": model_name, "keep_alive": "10m"},
                timeout=120,
            )
            get_loaded_models.clear()
            if r.status_code == 200:
                return True, f"Model `{model_name}` successfully loaded into VRAM."
            return False, f"Failed to load `{model_name}`: {r.text}"
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            time.sleep(2 * attempt)  # give the runner time to restart before retrying
        except Exception as e:
            return False, f"Loading error: {e}"
    return False, (
        f"Loading error: {last_err}. The Ollama runner likely crashed while loading "
        f"`{model_name}` (usually VRAM exhaustion on an 8 GB GPU). Make sure no other "
        f"model is resident in VRAM (`🧹 Free VRAM` first), then retry. If the problem "
        f"persists, check the `ollama serve` logs for a CUDA OOM error."
    )


def pull_ollama_model(model_name: str) -> tuple[bool, str]:
    """Pulls a model from Ollama library."""
    try:
        r = requests.post(
            f"{OLLAMA_API_BASE}/api/pull",
            json={"name": model_name, "stream": False},
            timeout=600,
        )
        check_ollama_status.clear()
        get_loaded_models.clear()
        if r.status_code == 200:
            return True, f"Successfully pulled `{model_name}`."
        return False, f"Pull failed: {r.text}"
    except Exception as e:
        return False, f"Pull error: {e}"


# Streamlit fragment decorator fallback (gracefully supports older or newer Streamlit versions)
fragment_decorator = getattr(st, "fragment", lambda f: f)

# Initialize persistent session state for auto-updating latest result and model tracking
if "latest_result" not in st.session_state:
    st.session_state["latest_result"] = None
if "last_loaded_model" not in st.session_state:
    st.session_state["last_loaded_model"] = None
if "active_selected_model" not in st.session_state:
    st.session_state["active_selected_model"] = list(REQUIRED_MODELS.keys())[0]


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
    """Appends benchmark measurement record to comparison_log.csv (lock + trailing newline)."""
    append_record_row(record, csv_path)


# --- 2.5 Google Docs Sync & Rubric Scoring Helpers ---
GOOGLE_DOCS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "google_docs_config.json")
SCORING_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "scoring_log.csv")


def load_google_docs_config() -> dict:
    """Loads the Google Docs sync config (Apps Script /exec URL + enabled flag)."""
    try:
        with open(GOOGLE_DOCS_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    cfg.setdefault("apps_script_url", "")
    cfg.setdefault("enabled", False)
    return cfg


def save_google_docs_config(cfg: dict):
    """Persists the Google Docs sync config to disk (gitignored)."""
    with open(GOOGLE_DOCS_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def save_score_record(record: dict):
    """Appends one human rubric score row to data/scoring_log.csv (lock + trailing newline)."""
    append_record_row(record, SCORING_LOG_PATH)


def _read_csv_quietly(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(path, on_bad_lines="warn")
    except Exception:
        pass
    return pd.DataFrame()


def compute_model_metrics(model_name: str, mode: str) -> dict:
    """Aggregates benchmark metrics for one model within a mode (test/official).

    Sources:
    - Scoring log (data/scoring_log.csv) -> Fully Correct Rate + Average Quality Score
    - Active mode comparison log -> Average Response Time, Tokens/sec, Total Tokens
    """
    comp_path = TEST_LOG_CSV_PATH if mode == "test" else OFFICIAL_LOG_CSV_PATH
    comp = _read_csv_quietly(comp_path)
    if not comp.empty and "model" in comp.columns:
        comp = comp[comp["model"] == model_name]

    scoring = _read_csv_quietly(SCORING_LOG_PATH)
    if not scoring.empty and "mode" in scoring.columns:
        scoring = scoring[(scoring["mode"] == mode) & (scoring["model"] == model_name)]

    fully_correct_rate = 0.0
    avg_quality_score = 0.0
    avg_response_time = 0.0
    tokens_per_sec = 0.0
    total_tokens = 0

    if not scoring.empty and "final_answer" in scoring.columns:
        total_runs = len(scoring)
        if total_runs > 0:
            correct = int((scoring["final_answer"] == 2).sum())
            fully_correct_rate = round(correct / total_runs * 100, 1)
            if "total_score" in scoring.columns:
                avg_quality_score = round(float(scoring["total_score"].mean()), 2)

    if not comp.empty:
        if "total_duration_s" in comp.columns:
            avg_response_time = round(float(comp["total_duration_s"].mean()), 2)
        if "tokens_per_sec" in comp.columns:
            tokens_per_sec = round(float(comp["tokens_per_sec"].mean()), 2)
        if "output_tokens" in comp.columns:
            total_tokens = int(comp["output_tokens"].sum())

    return {
        "fully_correct_rate": fully_correct_rate,
        "avg_quality_score": avg_quality_score,
        "avg_response_time": avg_response_time,
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": total_tokens,
    }


def append_result_to_docs(url: str, data: dict) -> tuple[bool, str]:
    """POSTs the metrics JSON to the Apps Script Web App URL."""
    clean_url = url.strip()
    if "docs.google.com/document" in clean_url:
        return False, "You pasted your Google Doc document URL. Please paste the Apps Script Web App URL (starts with https://script.google.com/macros/s/... and ends with /exec). See instructions below."

    try:
        r = requests.post(clean_url, json=data, timeout=30)
        if r.status_code == 200:
            try:
                resp = r.json()
                if resp.get("status") == "ok":
                    return True, "Synced to Google Docs"
                return False, f"Docs app replied: {resp}"
            except Exception:
                return True, f"Sent to Google Docs (HTTP {r.status_code})."
        if r.status_code == 405:
            return False, "HTTP 405 (Method Not Allowed). Make sure you deployed your Apps Script as a 'Web app' with 'doPost(e)', and that you copied the URL ending with '/exec'."
        return False, f"HTTP {r.status_code}: {r.text[:300]}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed: {e}"


# --- 3. Streamlit Page Setup ---
st.set_page_config(
    page_title="LLM Reasoning Benchmark (8GB VRAM)",
    page_icon="⚖️",
    layout="wide",
)

st.markdown("""
<style>
    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.80rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-blue { background: #1e3a8a; color: #93c5fd; }
    .badge-green { background: #064e3b; color: #6ee7b7; }
    .badge-amber { background: #78350f; color: #fcd34d; }
    .badge-purple { background: #4c1d95; color: #c4b5fd; }
    .rule-box {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #38bdf8;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }
    .mode-banner-test {
        background: rgba(234, 88, 12, 0.15);
        border: 1px solid #ea580c;
        color: #fdba74;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 0.88rem;
    }
    .mode-banner-official {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #6ee7b7;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. Sidebar: Mode Selection, Telemetry & Progress Matrix ---
with st.sidebar:
    st.title("⚖️ Benchmark Manager")
    st.caption("KiTH Internship Plan: 8 GB VRAM Study")

    # Storage Mode Toggle: Test vs Official
    workspace_mode = st.radio(
        "📁 Active Storage Destination",
        ["🧪 Test / Sandbox (`test_runs/`)", "📋 Official Benchmark (`data/responses/`)"],
        index=0,
        help="Use Test Mode to experiment without polluting the official 45-run dataset."
    )
    is_test_mode = "Test" in workspace_mode
    active_target_dir = TEST_RESPONSES_DIR if is_test_mode else OFFICIAL_RESPONSES_DIR
    active_csv_path = TEST_LOG_CSV_PATH if is_test_mode else OFFICIAL_LOG_CSV_PATH

    is_online, installed_tags = check_ollama_status()
    if is_online:
        st.success("🟢 Ollama Engine: Online", icon="✅")
    else:
        st.error("🔴 Ollama Offline. Run `ollama serve`.", icon="⚠️")

    # Mandatory Controlled Settings Display (Page 2)
    with st.expander("🔒 Frozen Experiment Settings", expanded=True):
        st.markdown(
            """
            - **Temperature:** `0.0` *(Zero stochasticity)*
            - **Context Window:** `4096` tokens
            - **Max Output:** `1024` tokens
            - **State:** Fresh request *(No prior context)*
            - **Concurrence:** 1 model loaded at a time
            """
        )

    # 45-Response Progress Tracker (Page 6)
    st.subheader("📊 Official 45-Response Progress")
    total_target = 15 * len(REQUIRED_MODELS)
    official_count = 0
    matrix_data = []

    for qid in sorted(BENCHMARK_QUESTIONS.keys()):
        row = {"QID": qid}
        for model_key, info in REQUIRED_MODELS.items():
            fn = get_response_filename(qid, model_key)
            fp = os.path.join(OFFICIAL_RESPONSES_DIR, fn)
            has_file = os.path.exists(fp)
            if has_file:
                official_count += 1
            row[info["alias"]] = "✅" if has_file else "❌"
        matrix_data.append(row)

    pct = official_count / total_target
    st.metric("Official Dataset Completed", f"{official_count} / {total_target} files", f"{pct*100:.1f}%")
    st.progress(pct)

    with st.expander("View 45-File Coverage Table", expanded=False):
        st.dataframe(pd.DataFrame(matrix_data), hide_index=True, use_container_width=True)

    # Official Rubric Drawer (Page 3)
    with st.expander("📖 Scoring Rubric Reference"):
        st.markdown(
            """
            **Each response evaluated on 0–2 scale (Total: 0–8):**
            1. **Final Answer:** 2 = Correct | 1 = Partly correct | 0 = Incorrect
            2. **Reasoning Quality:** 2 = Clear & logical | 1 = Minor gap | 0 = Illogical
            3. **Instruction Following:** 2 = Fully follows | 1 = Partly follows | 0 = Fails
            4. **Factual Support:** 2 = No invented claims | 1 = Minor claim | 0 = Major hallucinations
            """
        )

    # Google Docs Sync Card
    st.divider()
    st.markdown("#### 📄 Google Docs Sync")
    _gd_cfg = load_google_docs_config()
    _gd_url = st.text_input(
        "Apps Script `/exec` URL",
        value=_gd_cfg["apps_script_url"],
        key="gd_url_input",
        placeholder="https://script.google.com/macros/s/.../exec",
        help="Paste the Web app URL from your Google Apps Script deployment.",
    )
    if "docs.google.com/document" in _gd_url:
        st.warning(
            "⚠️ That looks like your Google Doc link, not your Apps Script Web App URL! "
            "Please open your Google Doc, go to **Extensions → Apps Script → Deploy → Web app**, and copy the URL ending with `/exec`."
        )
    _gd_enabled = st.checkbox(
        "Enable auto-sync to Google Docs",
        value=bool(_gd_cfg["enabled"]),
        key="gd_enable_toggle",
        help="When ON, saving a score also appends the metrics to your Google Doc.",
    )

    if _gd_url != _gd_cfg["apps_script_url"] or _gd_enabled != _gd_cfg["enabled"]:
        save_google_docs_config({"apps_script_url": _gd_url, "enabled": _gd_enabled})
        _gd_cfg = {"apps_script_url": _gd_url, "enabled": _gd_enabled}

    if _gd_enabled:
        if not _gd_url.strip():
            st.error("Copy your Apps Script **/exec URL** above to enable syncing.", icon="🔗")
        else:
            st.caption("Status: ✅ will send results → your Google Doc after you save a score.")
            if st.button("🧪 Test Docs Sync", key="gd_test_btn", use_container_width=True,
                         help="Sends a sample entry to your Google Doc to verify the connection (marked as a test)."):
                test_payload = {
                    "section_title": "🧪 SYNC TEST from Streamlit — safe to delete",
                    "question_id": "Q_TEST",
                    "model": "phi4-mini-reasoning",
                    "model_alias": "PHI",
                    "question": "Connection check triggered from the Streamlit sidebar.",
                    "total_score": 8,
                    "fully_correct_rate": 100.0,
                    "avg_quality_score": 8.0,
                    "avg_response_time": 12.3,
                    "tokens_per_sec": 50.0,
                    "total_tokens": 512,
                }
                ok, msg = append_result_to_docs(_gd_url.strip(), test_payload)
                if ok:
                    st.success(f"📄 {msg}", icon="✅")
                else:
                    st.error(f"📄 {msg}", icon="❌")
    else:
        st.caption("Status: ⏸️ Docs sync is OFF.")

# --- 5. Main Workspace: Test & Comparison Tool (Page 6 Spec) ---
st.subheader("🔬 Local Model Testing & Evidence Collection")

# Top Banner showing current active mode
if is_test_mode:
    st.markdown(
        """
        <div class="mode-banner-test">
            🧪 <b>TEST / SANDBOX MODE:</b> Responses are saved separately into <code>test_runs/responses/</code> 
            and <code>test_runs/comparison_log.csv</code>. Your official <code>data/responses/</code> folder remains completely pristine.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="mode-banner-official">
            📋 <b>OFFICIAL BENCHMARK MODE:</b> Responses will be saved directly into <code>data/responses/</code>.
            Follow the non-regeneration rule strictly: do not rerun weak answers.
        </div>
        """,
        unsafe_allow_html=True,
    )


@fragment_decorator
def render_benchmark_control_panel():
    # Query models currently resident in GPU memory via Ollama /api/ps
    loaded_models = get_loaded_models()

    # Step 1: Model Selection with VRAM Loading & Eviction Controls
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        def format_model_option(m_key: str) -> str:
            alias = REQUIRED_MODELS[m_key]["alias"]
            installed = is_model_installed(m_key, installed_tags)
            loaded = is_model_loaded_in_vram(m_key, loaded_models)
            if loaded:
                status = "🟢 In VRAM"
            elif installed:
                status = "✅ On Disk"
            else:
                status = "⚠️ Missing"
            return f"{m_key} ({alias}) — {status}"

        model_options = list(REQUIRED_MODELS.keys())
        saved_model = st.session_state.get("active_selected_model")
        default_idx = model_options.index(saved_model) if saved_model in model_options else 0

        selected_model = st.selectbox(
            "1. Select Active Model",
            model_options,
            index=default_idx,
            format_func=format_model_option,
            key="active_model_selectbox",
            help="Page 2 rule: Keep only one model loaded during each measurement."
        )
        st.session_state["active_selected_model"] = selected_model

        auto_load = st.checkbox(
            "⚡ Auto-load into VRAM on switch",
            value=False,
            key="auto_load_model_vram",
            help="Automatically pre-loads weights into GPU memory whenever you change models."
        )

        # Automatic VRAM eviction if user switches models to avoid 8GB VRAM saturation
        if st.session_state.get("last_loaded_model") and st.session_state["last_loaded_model"] != selected_model:
            prev_model = st.session_state["last_loaded_model"]
            if evict_ollama_model(prev_model):
                with st.spinner(f"Releasing `{prev_model}` from VRAM..."):
                    wait_for_model_evicted(prev_model)
            st.session_state["last_loaded_model"] = selected_model
            # Clear previous model's output card so screen doesn't stay stuck on old telemetry
            st.session_state["latest_result"] = None
            if auto_load and is_model_installed(selected_model, installed_tags):
                with st.spinner(f"Pre-loading `{selected_model}` into VRAM..."):
                    ok, msg = load_model_into_vram(selected_model)
                    if not ok:
                        st.error(f"⚠️ Failed to auto-load `{selected_model}` into VRAM: {msg}")
            st.rerun()
        elif not st.session_state.get("last_loaded_model"):
            st.session_state["last_loaded_model"] = selected_model
            if auto_load and is_model_installed(selected_model, installed_tags):
                with st.spinner(f"Pre-loading `{selected_model}` into VRAM..."):
                    load_model_into_vram(selected_model)

    with col_m2:
        m_info = REQUIRED_MODELS[selected_model]
        m_installed = is_model_installed(selected_model, installed_tags)
        m_loaded = is_model_loaded_in_vram(selected_model, loaded_models)

        if m_loaded:
            status_tag = "<span class='badge badge-blue'>🟢 Loaded in VRAM</span>"
        elif m_installed:
            status_tag = "<span class='badge badge-green'>✅ Installed (Idle on Disk)</span>"
        else:
            status_tag = f"<span class='badge badge-amber'>⚠️ Missing (Run: `ollama pull {selected_model}`)</span>"

        st.markdown(
            f"**Model:** `{selected_model}` | **Alias:** `{m_info['alias']}` | **Size:** `{m_info['size']}` {status_tag}<br>"
            f"*{m_info['purpose']}*",
            unsafe_allow_html=True
        )

        # Model Loading & VRAM Action Buttons
        col_act1, col_act2, col_act3 = st.columns([1, 1, 1])
        with col_act1:
            load_disabled = not m_installed or m_loaded
            btn_label = "⚡ Loaded in VRAM" if m_loaded else "⚡ Load into VRAM"
            if st.button(btn_label, key="load_vram_btn", disabled=load_disabled, help="Pre-loads model into GPU memory without sending a prompt"):
                with st.spinner(f"Loading `{selected_model}` into VRAM..."):
                    ok, msg = load_model_into_vram(selected_model)
                    if ok:
                        st.toast(f"Loaded `{selected_model}` into VRAM!", icon="⚡")
                        st.rerun()
                    else:
                        st.error(msg)

        with col_act2:
            unload_disabled = not m_loaded
            if st.button("🧹 Free VRAM", key="evict_vram_btn", disabled=unload_disabled, help="Evicts model from 8 GB VRAM to free GPU memory"):
                with st.spinner(f"Releasing `{selected_model}` from VRAM... This may take a few seconds."):
                    evict_ollama_model(selected_model)
                    wait_for_model_evicted(selected_model)
                st.toast(f"Evicted `{selected_model}` from VRAM!", icon="🧹")
                st.rerun()

        with col_act3:
            if not m_installed:
                if st.button(f"📥 Pull `{m_info['alias']}`", key="pull_model_btn", help=f"Downloads {selected_model} via Ollama"):
                    with st.spinner(f"Downloading `{selected_model}` (this may take a few minutes)..."):
                        ok, msg = pull_ollama_model(selected_model)
                        if ok:
                            st.toast(f"Downloaded `{selected_model}`!", icon="📥")
                            st.rerun()
                        else:
                            st.error(msg)

    # Step 2: Question Mode (Frozen Q01-Q15 vs Custom Typed Question)
    q_mode = st.radio(
        "2. Question Source",
        ["Select from 15 Frozen Benchmark Questions", "Custom Reasoning Question"],
        horizontal=True,
        key="benchmark_question_source_mode",
    )

    if q_mode.startswith("Select"):
        c_cat, c_qid = st.columns([1, 1])
        with c_cat:
            category_filter = st.selectbox(
                "Category",
                ["All", "Mathematical reasoning", "Logical reasoning", "Programming reasoning"],
                key="benchmark_category_filter",
            )
        with c_qid:
            filtered_qids = [
                qid for qid, data in BENCHMARK_QUESTIONS.items()
                if category_filter == "All" or data["category"] == category_filter
            ]
            selected_qid = st.selectbox("Question ID", filtered_qids, key="benchmark_qid_picker")

        q_data = BENCHMARK_QUESTIONS[selected_qid]
        prompt_text = q_data["prompt"]
        st.markdown(
            f"<span class='badge badge-blue'>{q_data['category']}</span>"
            f"<span class='badge badge-amber'>{q_data['difficulty']}</span>",
            unsafe_allow_html=True,
        )
        active_prompt = st.text_area(
            "Exact Prompt Wording (Frozen):",
            value=prompt_text,
            height=100,
            key=f"benchmark_frozen_prompt_{selected_qid}",
        )
        current_qid = selected_qid
    else:
        current_qid = st.text_input("Assign Question ID (e.g. Q_TEST01):", value="Q_TEST01", key="benchmark_custom_qid_input")
        active_prompt = st.text_area("Enter Typed Reasoning Question:", height=100, placeholder="Type any reasoning prompt to benchmark...", key="benchmark_custom_prompt_area")

    # Step 3: Hyperparameter & Temperature Adjustment
    if is_test_mode:
        with st.expander("🎛️ Adjustable Inference Settings (Sandbox Mode)", expanded=False):
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                active_temp = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.5,
                    value=0.0,
                    step=0.05,
                    key="param_temperature_slider",
                    help="0.0 = completely deterministic reasoning. 0.7+ = more creative and varied answers.",
                )
            with p_col2:
                active_predict = st.number_input(
                    "Max Output Tokens (num_predict)",
                    min_value=128,
                    max_value=4096,
                    value=FROZEN_SETTINGS["num_predict"],
                    step=128,
                    key="param_num_predict_input",
                )
            with p_col3:
                active_ctx = st.number_input(
                    "Context Window (num_ctx)",
                    min_value=1024,
                    max_value=8192,
                    value=FROZEN_SETTINGS["num_ctx"],
                    step=512,
                    key="param_num_ctx_input",
                )
    else:
        active_temp = FROZEN_SETTINGS["temperature"]
        active_predict = FROZEN_SETTINGS["num_predict"]
        active_ctx = FROZEN_SETTINGS["num_ctx"]
        st.caption("🔒 **Official Mode:** Hyperparameters locked (`Temperature = 0.0`, `Context = 4096`, `Max Output = 1024`).")

    # Check if response already exists in active directory
    existing_file = get_response_filename(current_qid, selected_model)
    existing_path = os.path.join(active_target_dir, existing_file)
    if os.path.exists(existing_path):
        st.warning(f"⚠️ Artifact `{existing_file}` already exists in `{active_target_dir}`. Running again will overwrite this record.")

    # Dynamic container for live streaming and instant result rendering
    stream_status_box = st.empty()
    stream_text_box = st.empty()
    result_box = st.empty()

    # Step 4: Run Inference with Live Streaming
    if st.button("▶️ Run Benchmark Test & Capture Telemetry", type="primary", use_container_width=True, key="run_benchmark_btn"):
        if not is_online:
            st.error("Cannot run test: Ollama is offline. Start the service first.")
        elif not active_prompt.strip():
            st.error("Prompt cannot be empty.")
        else:
            stream_status_box.info(f"⚡ Streaming live from `{selected_model}` (T={active_temp}, Context={active_ctx})...")
            
            start_vram = get_peak_vram_mb()
            start_time = time.time()
            
            req_payload = {
                "model": selected_model,
                "prompt": active_prompt,
                "stream": True,
                "options": {
                    "temperature": float(active_temp),
                    "num_ctx": int(active_ctx),
                    "num_predict": int(active_predict),
                },
            }

            inference_success = False
            try:
                full_raw_response = ""
                eval_count = 0
                eval_duration_ns = 0
                completion_status = "Completed"

                with requests.post(f"{OLLAMA_API_BASE}/api/generate", json=req_payload, stream=True, timeout=300) as resp:
                    if resp.status_code == 200:
                        in_thinking = False
                        for line in resp.iter_lines():
                            if line:
                                chunk = json.loads(line)
                                th = chunk.get("thinking")
                                token = chunk.get("response", "")

                                if th:
                                    if not in_thinking:
                                        full_raw_response += "<think>\n"
                                        in_thinking = True
                                    full_raw_response += th
                                    stream_text_box.markdown(full_raw_response + " ▌")
                                elif token:
                                    if in_thinking:
                                        full_raw_response += "\n</think>\n\n"
                                        in_thinking = False
                                    full_raw_response += token
                                    stream_text_box.markdown(full_raw_response + " ▌")

                                if chunk.get("done", False):
                                    if in_thinking:
                                        full_raw_response += "\n</think>\n\n"
                                        in_thinking = False
                                    eval_count = chunk.get("eval_count", 0)
                                    eval_duration_ns = chunk.get("eval_duration", 0)
                                    done_reason = chunk.get("done_reason", "stop")
                                    completion_status = "Completed" if done_reason == "stop" else done_reason

                        # Instantly clear raw stream placeholder and status
                        stream_text_box.empty()
                        stream_status_box.empty()

                        if not full_raw_response.strip():
                            stream_status_box.error(
                                f"⚠️ `{selected_model}` produced an empty response. "
                                "This typically happens if the model failed to allocate VRAM for inference. "
                                "Please click **'🧹 Free VRAM'** in the control panel to release memory and re-run."
                            )
                        else:
                            total_duration_sec = round(time.time() - start_time, 3)
                            end_vram = get_peak_vram_mb()
                            peak_vram = max(start_vram, end_vram)
                            eval_duration_sec = round(eval_duration_ns / 1e9, 3) if eval_duration_ns else total_duration_sec
                            tokens_per_sec = round(eval_count / eval_duration_sec, 2) if eval_duration_sec > 0 else 0.0

                            # Parse thought and verified answer
                            thought, final_ans = parse_reasoning_and_answer(full_raw_response)

                            # Automatic Compliant Storage
                            saved_filepath = save_response_artifact(current_qid, selected_model, active_prompt, full_raw_response, active_target_dir)

                            # Log to CSV (Page 6 requirement)
                            log_record = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "question_id": current_qid,
                                "model": selected_model,
                                "model_alias": m_info["alias"],
                                "total_duration_s": total_duration_sec,
                                "generation_duration_s": eval_duration_sec,
                                "output_tokens": eval_count,
                                "tokens_per_sec": tokens_per_sec,
                                "peak_vram_mb": peak_vram,
                                "completion_status": completion_status,
                                "evidence_file": saved_filepath,
                            }
                            log_comparison_csv(log_record, active_csv_path)

                            # Store latest result in session state
                            st.session_state["latest_result"] = {
                                "question_id": current_qid,
                                "model": selected_model,
                                "model_alias": m_info["alias"],
                                "prompt": active_prompt,
                                "raw_response": full_raw_response,
                                "thought": thought,
                                "final_ans": final_ans,
                                "total_duration_sec": total_duration_sec,
                                "eval_duration_sec": eval_duration_sec,
                                "eval_count": eval_count,
                                "tokens_per_sec": tokens_per_sec,
                                "peak_vram": peak_vram,
                                "completion_status": completion_status,
                                "saved_filepath": saved_filepath,
                                "temperature": active_temp,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            inference_success = True

                    else:
                        err_text = resp.text
                        if "out of memory" in err_text.lower() or "cuda" in err_text.lower() or resp.status_code == 500:
                            stream_status_box.error(
                                f"⚠️ **GPU Memory Allocation Failed ({resp.status_code})**: {err_text}\n\n"
                                f"Model `{selected_model}` could not fit into VRAM. "
                                "Click **'🧹 Free VRAM'** in the control panel to release memory, close heavy background apps, and try again."
                            )
                        else:
                            stream_status_box.error(f"Inference error {resp.status_code}: {err_text}")

            except requests.exceptions.Timeout:
                stream_status_box.error(
                    f"⏱️ Inference timed out while communicating with `{selected_model}`. "
                    "The model may be thrashing between system RAM and VRAM. Try freeing VRAM and rerunning."
                )
            except requests.exceptions.ConnectionError:
                stream_status_box.error("🔌 Lost connection to Ollama. Please check if the Ollama service is running.")
            except Exception as err:
                stream_status_box.error(f"Execution failed: {err}. As per guidelines, technical failures must be documented.")

            # Trigger app-level rerun so tables and inspectors re-read latest records immediately
            if inference_success:
                try:
                    st.rerun(scope="app")
                except TypeError:
                    st.rerun()

    # Render persistent & latest benchmark result immediately in-place (0ms delay)
    if st.session_state.get("latest_result"):
        res = st.session_state["latest_result"]
        with result_box.container():
            st.markdown("---")
            st.subheader("⚡ Latest Benchmark Result")
            st.caption(
                f"**Model:** `{res['model']}` (`{res['model_alias']}`) | "
                f"**Question:** `{res['question_id']}` | "
                f"**Temperature:** `{res.get('temperature', 0.0)}` | "
                f"**Finished at:** `{res['timestamp']}`"
            )

            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("Total Time", f"{res['total_duration_sec']} s")
            t2.metric("Output Tokens", f"{res['eval_count']}")
            t3.metric("Generation Speed", f"{res['tokens_per_sec']} tok/s")
            t4.metric("Peak VRAM", f"{res['peak_vram']:.0f} MB")
            t5.metric("Status", res["completion_status"])

            if res.get("thought"):
                with st.expander("💭 Reasoning Trace (<think>)", expanded=True):
                    st.markdown(format_latex_for_display(res["thought"]))

            st.markdown("#### Verified Output")
            st.markdown(format_latex_for_display(res["final_ans"]))
            st.success(f"💾 Evidence saved to: `{res['saved_filepath']}` (conforming to docs/response_schema.json)", icon="📁")

            # Human rubric scoring + optional Google Docs sync
            mode_str = "test" if is_test_mode else "official"
            with st.expander("📝 Score this answer (rubric 0–2)", expanded=True):
                st.caption("Human rubric scoring — matches the Page 3 rubric shown in the sidebar.")
                sc1, sc2, sc3, sc4 = st.columns(4)
                _score_key = f"{res['question_id']}_{res['model_alias']}"
                with sc1:
                    s_final = st.selectbox("Final Answer", [0, 1, 2], index=1, key=f"score_final_{_score_key}",
                                           help="2 = Correct | 1 = Partly correct | 0 = Incorrect")
                with sc2:
                    s_reason = st.selectbox("Reasoning Quality", [0, 1, 2], index=1, key=f"score_reason_{_score_key}",
                                            help="2 = Clear & logical | 1 = Minor gap | 0 = Illogical")
                with sc3:
                    s_instr = st.selectbox("Instruction Following", [0, 1, 2], index=1, key=f"score_instr_{_score_key}",
                                           help="2 = Fully follows | 1 = Partly follows | 0 = Fails")
                with sc4:
                    s_fact = st.selectbox("Factual Support", [0, 1, 2], index=1, key=f"score_fact_{_score_key}",
                                          help="2 = No invented claims | 1 = Minor claim | 0 = Major hallucinations")

                total_score = s_final + s_reason + s_instr + s_fact
                st.caption(f"Total Score: **{total_score} / 8**")

                if st.button("💾 Save Score & Sync to Docs", key=f"save_score_{_score_key}", use_container_width=True):
                    score_record = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": mode_str,
                        "question_id": res["question_id"],
                        "model": res["model"],
                        "model_alias": res["model_alias"],
                        "final_answer": s_final,
                        "reasoning_quality": s_reason,
                        "instruction_following": s_instr,
                        "factual_support": s_fact,
                        "total_score": total_score,
                    }
                    save_score_record(score_record)
                    st.success(f"✅ Saved score (**{total_score}/8**) to `data/scoring_log.csv`.", icon="📝")

                    try:
                        metrics = compute_model_metrics(res["model"], mode_str)
                    except Exception as ce:
                        st.error(f"⚠️ Could not compute metrics for docs sync: `{ce}`", icon="❌")
                        metrics = {}

                    _gd_cfg2 = load_google_docs_config()
                    _gd_url2 = _gd_cfg2.get("apps_script_url", "")
                    _gd_on2 = _gd_cfg2.get("enabled", False)

                    if _gd_on2 and _gd_url2.strip():
                        payload = {
                            "section_title": f"Question: {res['question_id']} · Model: {res['model_alias']} ({res['model']})",
                            "question_id": res["question_id"],
                            "model": res["model"],
                            "model_alias": res["model_alias"],
                            "question": res.get("prompt", ""),
                            "total_score": total_score,
                            "fully_correct_rate": metrics.get("fully_correct_rate", 0.0),
                            "avg_quality_score": metrics.get("avg_quality_score", 0.0),
                            "avg_response_time": metrics.get("avg_response_time", 0.0),
                            "tokens_per_sec": metrics.get("tokens_per_sec", 0.0),
                            "total_tokens": metrics.get("total_tokens", 0),
                        }
                        try:
                            ok, msg = append_result_to_docs(_gd_url2, payload)
                            if ok:
                                st.success(f"📄 {msg}: `{payload['section_title']}`", icon="✅")
                            else:
                                st.error(f"📄 {msg}", icon="❌")
                        except Exception as de:
                            st.error(f"📄 Docs sync failed unexpectedly: `{de}`", icon="❌")
                    else:
                        st.info(
                            "📄 Docs sync is **OFF** — this score was saved locally only. Enable it in the sidebar to start appending results to your Google Doc.",
                            icon="ℹ️",
                        )


# Render the isolated benchmark control panel
render_benchmark_control_panel()

# --- 6. Inspect Saved Evidence & Comparison Log ---
st.divider()
col_insp_title, col_insp_btn = st.columns([5, 1])
with col_insp_title:
    st.subheader("🔍 Artifact Inspector & Comparison Log")
with col_insp_btn:
    if st.button("🔄 Refresh Logs", key="manual_refresh_logs_btn", use_container_width=True, help="Force re-read comparison logs and artifacts from disk"):
        st.rerun()

def format_vram_telemetry(row: dict) -> str:
    """Safely formats Peak VRAM from row whether stored in MB or GB."""
    val = row.get("peak_vram_mb")
    if val is None or pd.isna(val) or str(val).strip() == "":
        val = row.get("peak_vram_gb")
    if val is None or pd.isna(val) or str(val).strip() == "":
        return "N/A"
    try:
        num = float(val)
        if num > 100:  # Value in MB
            return f"{num:.0f} MB ({num/1024:.2f} GB)"
        elif num > 0:  # Value in GB
            return f"{num*1024:.0f} MB ({num:.2f} GB)"
        return "0 MB"
    except (ValueError, TypeError):
        return str(val)


def load_evidence_artifact(row: dict, responses_dir: str):
    """Loads response JSON artifact for a given run record."""
    qid = row.get("question_id", "")
    model_name = row.get("model", "")
    model_alias = row.get("model_alias", "")
    ev_file = str(row.get("evidence_file", "")).strip()

    evidence_path = None
    if ev_file and os.path.exists(ev_file) and os.path.isfile(ev_file):
        evidence_path = ev_file
    else:
        alias = model_alias or REQUIRED_MODELS.get(model_name, {}).get("alias", "")
        fallback_fn = f"{qid}_{alias}.json"
        check_path = os.path.join(responses_dir, fallback_fn)
        if os.path.exists(check_path):
            evidence_path = check_path

    if evidence_path and os.path.exists(evidence_path):
        try:
            with open(evidence_path, "r", encoding="utf-8") as f:
                art = json.load(f)
            return art, evidence_path
        except Exception:
            pass
    return None, None


def render_side_by_side_comparison(df: pd.DataFrame, responses_dir: str, tab_key: str):
    """Renders 3-model side-by-side comparison for a selected frozen question."""
    st.markdown("##### ⚖️ Side-by-Side 3-Model Comparison View")
    st.caption("Select any benchmark question to compare PHI, DSR1, and QWEN outputs and hardware telemetry side-by-side:")

    q_keys = list(BENCHMARK_QUESTIONS.keys())
    selected_comp_qid = st.selectbox(
        "Select Benchmark Question",
        q_keys,
        format_func=lambda q: f"{q} [{BENCHMARK_QUESTIONS[q]['category']} • {BENCHMARK_QUESTIONS[q]['difficulty']}]: {BENCHMARK_QUESTIONS[q]['prompt'][:75]}...",
        key=f"{tab_key}_comp_qid_select",
        label_visibility="collapsed"
    )

    q_info = BENCHMARK_QUESTIONS[selected_comp_qid]
    with st.expander(f"📌 Prompt Details for `{selected_comp_qid}` ({q_info['category']} • {q_info['difficulty']})", expanded=False):
        st.markdown(f"> {q_info['prompt']}")

    col_phi, col_dsr1, col_qwen = st.columns(3)
    target_models = [
        ("phi4-mini-reasoning", "PHI", "Phi-4 Mini Reasoning", col_phi, "🟢"),
        ("deepseek-r1:7b", "DSR1", "DeepSeek R1 7B", col_dsr1, "🔵"),
        ("qwen3:8b", "QWEN", "Qwen 3 8B", col_qwen, "🟣"),
    ]

    for model_name, alias, display_name, col, badge in target_models:
        with col:
            with st.container(border=True):
                st.markdown(f"**{badge} {alias}** (`{model_name}`)")

                # Find run in df
                matched_row = None
                if not df.empty and "question_id" in df.columns:
                    mask = (df["question_id"] == selected_comp_qid) & (
                        (df["model_alias"] == alias) | (df["model"] == model_name)
                    )
                    matches = df[mask]
                    if not matches.empty:
                        matched_row = matches.iloc[-1].to_dict()

                # Find artifact
                art_data, art_path = None, None
                if matched_row:
                    art_data, art_path = load_evidence_artifact(matched_row, responses_dir)
                else:
                    fn = f"{selected_comp_qid}_{alias}.json"
                    cand = os.path.join(responses_dir, fn)
                    if os.path.exists(cand):
                        try:
                            with open(cand, "r", encoding="utf-8") as f:
                                art_data = json.load(f)
                            art_path = cand
                        except Exception:
                            pass

                if matched_row or art_data:
                    vram_str = format_vram_telemetry(matched_row) if matched_row else "N/A"
                    tps = matched_row.get("tokens_per_sec") if matched_row else None
                    dur = matched_row.get("total_duration_s") if matched_row else None
                    toks = matched_row.get("output_tokens") if matched_row else None

                    m_c1, m_c2 = st.columns(2)
                    m_c1.metric("💾 Peak VRAM", vram_str)
                    m_c2.metric("⚡ Speed", f"{tps} tok/s" if tps is not None and not pd.isna(tps) else "N/A")

                    m_c3, m_c4 = st.columns(2)
                    m_c3.metric("⏱️ Duration", f"{dur} s" if dur is not None and not pd.isna(dur) else "N/A")
                    m_c4.metric("🔢 Tokens", f"{toks}" if toks is not None and not pd.isna(toks) else "N/A")

                    if art_data:
                        resp_text = art_data.get("response", "")
                        thought, final_ans = parse_reasoning_and_answer(resp_text)
                        if thought:
                            with st.expander("💭 Reasoning Trace (<think>)", expanded=False):
                                st.markdown(format_latex_for_display(thought))
                        st.markdown("**Verified Answer:**")
                        st.markdown(format_latex_for_display(final_ans or resp_text))
                    else:
                        st.caption("ℹ️ Telemetry recorded, response artifact not found.")

                    if art_path:
                        st.caption(f"📁 `{os.path.basename(art_path)}`")
                else:
                    st.info(f"⚪ No run recorded yet for **{alias}** on `{selected_comp_qid}`.")


def render_master_detail_log(df: pd.DataFrame, responses_dir: str, tab_key: str, download_filename: str):
    """Renders clean 60/40 Master-Detail layout with search filters and live Telemetry Inspector."""
    st.markdown("##### 📋 Master Comparison Log & Live Inspector")

    # Filter Bar
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        models_in_df = sorted([str(x) for x in df["model_alias"].dropna().unique() if str(x).strip()]) if not df.empty and "model_alias" in df.columns else []
        sel_model = st.selectbox("Filter Model", ["All Models"] + models_in_df, key=f"{tab_key}_filt_model")
    with col_f2:
        qs_in_df = sorted([str(x) for x in df["question_id"].dropna().unique() if str(x).strip()]) if not df.empty and "question_id" in df.columns else []
        sel_q = st.selectbox("Filter Question", ["All Questions"] + qs_in_df, key=f"{tab_key}_filt_q")
    with col_f3:
        statuses_in_df = sorted([str(x) for x in df["completion_status"].dropna().unique() if str(x).strip()]) if not df.empty and "completion_status" in df.columns else []
        sel_status = st.selectbox("Filter Status", ["All Statuses"] + statuses_in_df, key=f"{tab_key}_filt_status")

    # Filter data
    df_filt = df.copy() if not df.empty else pd.DataFrame()
    if not df_filt.empty:
        if sel_model != "All Models" and "model_alias" in df_filt.columns:
            df_filt = df_filt[df_filt["model_alias"] == sel_model]
        if sel_q != "All Questions" and "question_id" in df_filt.columns:
            df_filt = df_filt[df_filt["question_id"] == sel_q]
        if sel_status != "All Statuses" and "completion_status" in df_filt.columns:
            df_filt = df_filt[df_filt["completion_status"] == sel_status]

    if df_filt.empty:
        st.info("No run records match the current filter selection.")
        return

    # Master-Detail Split (60% Table, 40% Telemetry Card)
    col_table, col_detail = st.columns([3, 2])

    with col_table:
        st.markdown(f"**Records Table** ({len(df_filt)} runs found)")

        # Prepare clean columns for display
        preferred_cols = ["question_id", "model_alias", "tokens_per_sec", "peak_vram_mb", "total_duration_s", "output_tokens", "completion_status", "timestamp"]
        available_cols = [c for c in preferred_cols if c in df_filt.columns]
        if "peak_vram_mb" not in available_cols and "peak_vram_gb" in df_filt.columns:
            available_cols.insert(3, "peak_vram_gb")

        df_display = df_filt[available_cols].copy()
        
        rename_map = {
            "question_id": "QID",
            "model_alias": "Model",
            "tokens_per_sec": "Speed (tok/s)",
            "peak_vram_mb": "Peak VRAM (MB)",
            "peak_vram_gb": "Peak VRAM (GB)",
            "total_duration_s": "Duration (s)",
            "output_tokens": "Tokens",
            "completion_status": "Status",
            "timestamp": "Timestamp",
        }
        df_display = df_display.rename(columns={k: v for k, v in rename_map.items() if k in df_display.columns})

        selected_row_idx = None
        try:
            event = st.dataframe(
                df_display,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key=f"{tab_key}_df_select",
                hide_index=False
            )
            if event and hasattr(event, "selection") and event.selection.rows:
                selected_row_idx = event.selection.rows[0]
        except Exception:
            st.dataframe(df_display, use_container_width=True)

        max_idx = len(df_filt)
        default_val = int(selected_row_idx + 1) if selected_row_idx is not None and selected_row_idx < max_idx else max_idx
        nav_col1, nav_col2 = st.columns([1, 1])
        with nav_col1:
            chosen_row_num = st.number_input(
                "Inspect Row #",
                min_value=1,
                max_value=max_idx,
                value=default_val,
                step=1,
                key=f"{tab_key}_num_input",
                help="Select row number to view its full telemetry on the right"
            )
        with nav_col2:
            st.write("")
            st.download_button(
                "📥 Export CSV",
                df_filt.to_csv(index=False),
                file_name=download_filename,
                use_container_width=True
            )

        active_idx = chosen_row_num - 1

    with col_detail:
        selected_record = df_filt.iloc[active_idx].to_dict()
        qid = selected_record.get("question_id", "Run")
        alias = selected_record.get("model_alias", "")
        model_name = selected_record.get("model", "")
        ts = selected_record.get("timestamp", "")
        status = selected_record.get("completion_status", "Completed")

        with st.container(border=True):
            st.markdown(f"#### 📊 Telemetry Card: `{qid}` • `{alias}`")
            st.caption(f"**Model:** `{model_name}` | **Time:** `{ts}` | **Status:** `{status}`")

            # Metrics Grid
            m1, m2 = st.columns(2)
            m1.metric("💾 Peak VRAM", format_vram_telemetry(selected_record))
            tps_val = selected_record.get("tokens_per_sec", 0)
            m2.metric("⚡ Speed", f"{tps_val} tok/s" if tps_val else "N/A")

            m3, m4 = st.columns(2)
            tot_s = selected_record.get("total_duration_s", 0)
            gen_s = selected_record.get("generation_duration_s", 0)
            m3.metric("⏱️ Duration", f"{tot_s} s" if tot_s else "N/A", f"Eval: {gen_s} s" if gen_s else None)
            m4.metric("🔢 Output Tokens", f"{selected_record.get('output_tokens', 0)}")

            # Artifact payload
            art_data, art_path = load_evidence_artifact(selected_record, responses_dir)
            if art_data:
                prompt_txt = art_data.get("prompt", "")
                resp_txt = art_data.get("response", "")
                thought, final_ans = parse_reasoning_and_answer(resp_txt)

                if prompt_txt:
                    with st.expander("📝 Question Prompt", expanded=False):
                        st.markdown(f"> {prompt_txt}")

                if thought:
                    with st.expander("💭 Chain-of-Thought (<think>)", expanded=True):
                        st.markdown(format_latex_for_display(thought))

                st.markdown("**Verified Answer Output:**")
                st.markdown(format_latex_for_display(final_ans or resp_txt))

                if art_path:
                    st.caption(f"📁 Evidence Artifact: `{art_path}`")
                    with st.expander("📄 Raw JSON Artifact"):
                        st.json(art_data)
            else:
                st.info("ℹ️ No JSON response artifact file found for this record.")


tab_test, tab_official = st.tabs(["🧪 Test Runs (`test_runs/`)", "📋 Official Dataset (`data/`)"])

with tab_test:
    if os.path.exists(TEST_LOG_CSV_PATH):
        df_test = pd.read_csv(TEST_LOG_CSV_PATH, on_bad_lines="warn")
        if not df_test.empty:
            render_side_by_side_comparison(df_test, TEST_RESPONSES_DIR, "test")
            st.markdown("---")
            render_master_detail_log(df_test, TEST_RESPONSES_DIR, "test", "test_comparison_log.csv")
        else:
            st.info("Test comparison log is empty.")
    else:
        st.info("No test comparison log found yet.")

with tab_official:
    if os.path.exists(OFFICIAL_LOG_CSV_PATH):
        df_off = pd.read_csv(OFFICIAL_LOG_CSV_PATH, on_bad_lines="warn")
        if not df_off.empty:
            render_side_by_side_comparison(df_off, OFFICIAL_RESPONSES_DIR, "official")
            st.markdown("---")
            render_master_detail_log(df_off, OFFICIAL_RESPONSES_DIR, "official", "official_comparison_log.csv")
        else:
            st.info("Official comparison log is empty.")
    else:
        st.info("No official runs logged yet. Switch to 'Official Benchmark Mode' when ready to collect the 45 deliverables.")

