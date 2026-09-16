import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Ground truth & question definitions
QUESTIONS = {
    "Q01": {"category": "Mathematical reasoning", "difficulty": "Easy", "name": "Linear Equation", "key": "x = 6"},
    "Q02": {"category": "Mathematical reasoning", "difficulty": "Easy", "name": "Percentage Discount", "key": "$64"},
    "Q03": {"category": "Mathematical reasoning", "difficulty": "Medium", "name": "Factorial Trailing Zeroes", "key": "24"},
    "Q04": {"category": "Mathematical reasoning", "difficulty": "Medium", "name": "Polynomial Sequence", "key": "42 and 56"},
    "Q05": {"category": "Mathematical reasoning", "difficulty": "Hard", "name": "Three Dice Probability", "key": "1/8 or 27/216"},
    "Q06": {"category": "Logical reasoning", "difficulty": "Easy", "name": "Sheep Subtraction", "key": "9 live sheep"},
    "Q07": {"category": "Logical reasoning", "difficulty": "Easy", "name": "Syllogism Roses", "key": "No (undistributed middle)"},
    "Q08": {"category": "Logical reasoning", "difficulty": "Medium", "name": "Linear Seating", "key": "Alice far left (Alice, Bob, Charlie)"},
    "Q09": {"category": "Logical reasoning", "difficulty": "Medium", "name": "Water Jug 4L", "key": "4 liters in 5L jug"},
    "Q10": {"category": "Logical reasoning", "difficulty": "Hard", "name": "Knights & Knaves", "key": "A is Knight, B is Knave"},
    "Q11": {"category": "Programming reasoning", "difficulty": "Easy", "name": "Palindrome Function", "key": "Python palindrome with isalnum + lower"},
    "Q12": {"category": "Programming reasoning", "difficulty": "Easy", "name": "List Mutation Trace", "key": "4 True"},
    "Q13": {"category": "Programming reasoning", "difficulty": "Medium", "name": "Binary Search Bug", "key": "Infinite loop (low = mid vs low = mid + 1, high bound)"},
    "Q14": {"category": "Programming reasoning", "difficulty": "Medium", "name": "Floyd Cycle Detection", "key": "O(N) time, O(1) space Tortoise & Hare"},
    "Q15": {"category": "Programming reasoning", "difficulty": "Hard", "name": "LIS Patience Sort", "key": "O(N log N) patience sort / bisect"}
}

MODELS = {
    "PHI": {"full_name": "Phi-4 Mini Reasoning", "tag": "phi4-mini-reasoning:latest", "params": "3.84B", "color": "#2b5c8f"},
    "DSR1": {"full_name": "DeepSeek-R1 Distill Qwen 7B", "tag": "deepseek-r1:7b", "params": "7.62B", "color": "#0d9488"},
    "QWEN": {"full_name": "Qwen3 8B", "tag": "qwen3:8b", "params": "8.19B", "color": "#d97706"}
}

# Detailed Rubric Scores based on systematic evaluation of all 45 responses against answer keys:
# Criteria (each 0-2): [Final Answer, Reasoning Quality, Instruction Following, Factual Support] -> Total (0-8)
SCORES = {
    "PHI": {
        "Q01": [2, 2, 2, 2], # x=6, full working
        "Q02": [2, 2, 2, 2], # $64, full working
        "Q03": [2, 2, 2, 2], # 24 trailing zeroes, Legendre formula
        "Q04": [0, 1, 1, 1], # cut off in thinking, partial reasoning
        "Q05": [0, 1, 1, 1], # cut off in thinking
        "Q06": [2, 2, 2, 2], # 9 sheep, correct logic
        "Q07": [2, 2, 2, 2], # No, undistributed middle
        "Q08": [0, 1, 1, 1], # cut off in thinking
        "Q09": [0, 1, 1, 1], # cut off in thinking
        "Q10": [0, 1, 1, 1], # cut off in thinking
        "Q11": [2, 2, 2, 2], # is_palindrome function, clean trace
        "Q12": [0, 1, 1, 1], # cut off in thinking
        "Q13": [0, 1, 1, 1], # cut off in thinking
        "Q14": [0, 1, 1, 1], # cut off in thinking
        "Q15": [0, 1, 1, 1], # cut off in thinking
    },
    "DSR1": {
        "Q01": [2, 2, 2, 2], # x=6, clear steps
        "Q02": [2, 2, 2, 2], # $64, clear steps
        "Q03": [0, 1, 1, 1], # cut off in thinking
        "Q04": [2, 2, 2, 2], # 42 and 56, polynomial rule
        "Q05": [0, 1, 1, 1], # cut off in thinking
        "Q06": [2, 2, 2, 2], # 9 sheep, correct logic
        "Q07": [0, 1, 1, 1], # answered 'Yes' (fallacy made), incorrect final answer
        "Q08": [2, 2, 2, 2], # Alice far left (Alice, Bob, Charlie)
        "Q09": [2, 2, 2, 2], # 4 liters in 5L jug, full steps
        "Q10": [0, 1, 1, 1], # cut off in thinking
        "Q11": [0, 1, 1, 1], # cut off in thinking
        "Q12": [2, 2, 2, 2], # 4 True, list mutation explanation
        "Q13": [0, 1, 1, 1], # cut off in thinking
        "Q14": [2, 2, 2, 2], # Floyd cycle finding, O(N) time O(1) space
        "Q15": [0, 1, 1, 1], # cut off in thinking
    },
    "QWEN": {
        "Q01": [2, 2, 2, 2], # x=6, clear steps
        "Q02": [0, 1, 1, 1], # cut off in thinking
        "Q03": [0, 1, 1, 1], # cut off in thinking
        "Q04": [0, 1, 1, 1], # cut off in thinking
        "Q05": [0, 1, 1, 1], # cut off in thinking
        "Q06": [2, 2, 2, 2], # 9 sheep, correct logic
        "Q07": [2, 2, 2, 2], # No, formal predicate justification
        "Q08": [0, 1, 1, 1], # cut off in thinking
        "Q09": [0, 1, 1, 1], # cut off in thinking
        "Q10": [0, 1, 1, 1], # cut off in thinking
        "Q11": [0, 1, 1, 1], # cut off in thinking
        "Q12": [2, 2, 2, 2], # 4 True, mutable reference
        "Q13": [0, 1, 1, 1], # cut off in thinking
        "Q14": [0, 1, 1, 1], # cut off in thinking
        "Q15": [0, 1, 1, 1], # cut off in thinking
    }
}

# Load telemetry
df_log = pd.read_csv("data/comparison_log.csv")
latest_log = df_log.sort_values("timestamp").groupby(["model_alias", "question_id"]).last().reset_index()

# Compile Master Table
rows = []
for model_alias, m_info in MODELS.items():
    for q_id, q_info in QUESTIONS.items():
        sub = latest_log[(latest_log["model_alias"] == model_alias) & (latest_log["question_id"] == q_id)]
        rubric = SCORES[model_alias][q_id]
        total_score = sum(rubric)
        is_correct = 1 if (rubric[0] == 2 and total_score >= 7) else 0
        
        tot_dur = sub["total_duration_s"].values[0] if len(sub) > 0 else 0
        gen_dur = sub["generation_duration_s"].values[0] if len(sub) > 0 else 0
        tps = sub["tokens_per_sec"].values[0] if len(sub) > 0 else 0
        tokens = sub["output_tokens"].values[0] if len(sub) > 0 else 0
        vram = sub["peak_vram_gb"].values[0] if len(sub) > 0 else 0
        if vram > 100:
            vram = vram / 1024.0 # convert MB to GB
            
        rows.append({
            "model_alias": model_alias,
            "model_name": m_info["full_name"],
            "question_id": q_id,
            "category": q_info["category"],
            "difficulty": q_info["difficulty"],
            "final_answer_score": rubric[0],
            "reasoning_score": rubric[1],
            "instruction_score": rubric[2],
            "factual_score": rubric[3],
            "total_score": total_score,
            "is_correct": is_correct,
            "total_duration_s": tot_dur,
            "generation_duration_s": gen_dur,
            "tokens_per_sec": tps,
            "output_tokens": tokens,
            "peak_vram_gb": vram
        })

df_master = pd.DataFrame(rows)

# Aggregations
model_summary = df_master.groupby("model_alias").agg({
    "is_correct": lambda x: (x.sum() / len(x)) * 100,
    "total_score": "mean",
    "total_duration_s": "mean",
    "tokens_per_sec": "mean",
    "peak_vram_gb": "max"
}).reset_index()

model_summary["model_name"] = model_summary["model_alias"].map(lambda x: MODELS[x]["full_name"])
print("=== Model Summary ===")
print(model_summary)

# Category breakdown
cat_summary = df_master.groupby(["model_alias", "category"]).agg({
    "is_correct": lambda x: (x.sum() / len(x)) * 100,
    "total_score": "mean"
}).reset_index()

# Difficulty breakdown
diff_summary = df_master.groupby(["model_alias", "difficulty"]).agg({
    "is_correct": lambda x: (x.sum() / len(x)) * 100,
    "total_score": "mean"
}).reset_index()

# Generate 4 Charts
os.makedirs("docs/evidence", exist_ok=True)
models_order = ["PHI", "DSR1", "QWEN"]
labels = [f"{MODELS[m]['full_name']}\n({MODELS[m]['params']})" for m in models_order]
colors = [MODELS[m]["color"] for m in models_order]

# Chart 1: Fully Correct Answer Percentage
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
acc_vals = [model_summary[model_summary["model_alias"] == m]["is_correct"].values[0] for m in models_order]
bars = ax.bar(labels, acc_vals, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
ax.set_ylabel("Fully Correct Rate (%)", fontsize=12, fontweight='bold')
ax.set_title("Figure 1: Fully Correct Answer Percentage by Model (N=15)", fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 100)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("docs/evidence/chart1_fully_correct_percentage.png")
plt.close()

# Chart 2: Average Total Quality Score
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
score_vals = [model_summary[model_summary["model_alias"] == m]["total_score"].values[0] for m in models_order]
bars = ax.bar(labels, score_vals, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
ax.set_ylabel("Average Quality Score (0 to 8 Rubric Scale)", fontsize=12, fontweight='bold')
ax.set_title("Figure 2: Average Total Quality Score by Model (0–8 Scale)", fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 8.5)
ax.axhline(8, color='gray', linestyle='--', alpha=0.5, label='Maximum Score (8.0)')
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f} / 8.0',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("docs/evidence/chart2_average_quality_score.png")
plt.close()

# Chart 3: Average Response Time
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
time_vals = [model_summary[model_summary["model_alias"] == m]["total_duration_s"].values[0] for m in models_order]
bars = ax.bar(labels, time_vals, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
ax.set_ylabel("Average Latency (Seconds)", fontsize=12, fontweight='bold')
ax.set_title("Figure 3: Average Response Time by Model (Lower is Faster)", fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 48)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f} s',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("docs/evidence/chart3_average_response_time.png")
plt.close()

# Chart 4: Average Generation Speed (Tokens/sec)
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
tps_vals = [model_summary[model_summary["model_alias"] == m]["tokens_per_sec"].values[0] for m in models_order]
bars = ax.bar(labels, tps_vals, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
ax.set_ylabel("Generation Throughput (Tokens/sec)", fontsize=12, fontweight='bold')
ax.set_title("Figure 4: Average Generation Speed in Tokens per Second", fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 48)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f} tok/s',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("docs/evidence/chart4_tokens_per_second.png")
plt.close()

print("Charts successfully saved to docs/evidence/")
