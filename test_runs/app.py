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


def evict_ollama_model(model_name: str) -> bool:
    """Sends keep_alive: 0 to Ollama to release model from 8 GB VRAM."""
    try:
        requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=4,
        )
        get_loaded_models.clear()
        return True
    except Exception:
        return False


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
    """Pre-loads/warms up a model into GPU memory via Ollama API without generating text."""
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
    except Exception as e:
        return False, f"Loading error: {e}"


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

        selected_model = st.selectbox(
            "1. Select Active Model",
            list(REQUIRED_MODELS.keys()),
            index=0,
            format_func=format_model_option,
            key="active_model_selectbox",
            help="Page 2 rule: Keep only one model loaded during each measurement."
        )

        auto_load = st.checkbox(
            "⚡ Auto-load into VRAM on switch",
            value=False,
            key="auto_load_model_vram",
            help="Automatically pre-loads weights into GPU memory whenever you change models."
        )

        # Automatic VRAM eviction if user switches models to avoid 8GB VRAM saturation
        if st.session_state.get("last_loaded_model") and st.session_state["last_loaded_model"] != selected_model:
            evict_ollama_model(st.session_state["last_loaded_model"])
            st.session_state["last_loaded_model"] = selected_model
            if auto_load and is_model_installed(selected_model, installed_tags):
                load_model_into_vram(selected_model)
        elif not st.session_state.get("last_loaded_model"):
            st.session_state["last_loaded_model"] = selected_model
            if auto_load and is_model_installed(selected_model, installed_tags):
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
                evict_ollama_model(selected_model)
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
        active_prompt = st.text_area("Exact Prompt Wording (Frozen):", value=prompt_text, height=100, key="benchmark_frozen_prompt_area")
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

            try:
                full_raw_response = ""
                eval_count = 0
                eval_duration_ns = 0
                completion_status = "Completed"

                with requests.post(f"{OLLAMA_API_BASE}/api/generate", json=req_payload, stream=True, timeout=300) as resp:
                    if resp.status_code == 200:
                        for line in resp.iter_lines():
                            if line:
                                chunk = json.loads(line)
                                token = chunk.get("response", "")
                                full_raw_response += token
                                stream_text_box.markdown(full_raw_response + " ▌")
                                
                                if chunk.get("done", False):
                                    eval_count = chunk.get("eval_count", 0)
                                    eval_duration_ns = chunk.get("eval_duration", 0)
                                    done_reason = chunk.get("done_reason", "stop")
                                    completion_status = "Completed" if done_reason == "stop" else done_reason

                        total_duration_sec = round(time.time() - start_time, 3)
                        end_vram = get_peak_vram_mb()
                        peak_vram = max(start_vram, end_vram)
                        eval_duration_sec = round(eval_duration_ns / 1e9, 3) if eval_duration_ns else total_duration_sec
                        tokens_per_sec = round(eval_count / eval_duration_sec, 2) if eval_duration_sec > 0 else 0.0

                        # Instantly clear raw stream placeholder and status
                        stream_text_box.empty()
                        stream_status_box.empty()

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

                    else:
                        stream_status_box.error(f"Inference error {resp.status_code}: {resp.text}")

            except Exception as err:
                stream_status_box.error(f"Execution failed: {err}. As per guidelines, technical failures must be documented.")

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


# Render the isolated benchmark control panel
render_benchmark_control_panel()

# --- 6. Inspect Saved Evidence & Comparison Log ---
st.divider()
st.subheader("🔍 Artifact Inspector & Comparison Log")

tab_test, tab_official = st.tabs(["🧪 Test Runs (`test_runs/`)", "📋 Official Dataset (`data/`)"])

with tab_test:
    st.markdown("#### Test Artifacts & Logs")
    if os.path.exists(TEST_LOG_CSV_PATH):
        df_test = pd.read_csv(TEST_LOG_CSV_PATH)
        st.dataframe(df_test, use_container_width=True)
        st.download_button("📥 Download Test CSV", df_test.to_csv(index=False), file_name="test_comparison_log.csv")
    else:
        st.info("No test comparison log found yet.")

    # Inspect test JSON
    if os.path.exists(TEST_RESPONSES_DIR):
        test_files = [f for f in os.listdir(TEST_RESPONSES_DIR) if f.endswith(".json")]
        if test_files:
            chosen_test_file = st.selectbox("Inspect Test JSON File:", test_files)
            with open(os.path.join(TEST_RESPONSES_DIR, chosen_test_file), "r", encoding="utf-8") as f:
                st.json(json.load(f))

with tab_official:
    st.markdown("#### Official Frozen Deliverables")
    if os.path.exists(OFFICIAL_LOG_CSV_PATH):
        df_off = pd.read_csv(OFFICIAL_LOG_CSV_PATH)
        st.dataframe(df_off, use_container_width=True)
        st.download_button("📥 Download Official CSV", df_off.to_csv(index=False), file_name="official_comparison_log.csv")
    else:
        st.info("No official runs logged yet. Switch to 'Official Benchmark Mode' when ready to collect the 45 deliverables.")

    c_iqid, c_imodel = st.columns(2)
    with c_iqid:
        insp_qid = st.selectbox("Inspect Official QID", sorted(BENCHMARK_QUESTIONS.keys()), key="insp_official_qid")
    with c_imodel:
        insp_model = st.selectbox("Inspect Official Model", list(REQUIRED_MODELS.keys()), key="insp_official_model")
    
    official_target = os.path.join(OFFICIAL_RESPONSES_DIR, get_response_filename(insp_qid, insp_model))
    if os.path.exists(official_target):
        with open(official_target, "r", encoding="utf-8") as f:
            st.json(json.load(f))
    else:
        st.caption(f"Artifact `{official_target}` has not been collected yet.")
