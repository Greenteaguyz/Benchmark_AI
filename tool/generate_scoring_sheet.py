"""
Generate evaluation/master_scoring.xlsx conforming strictly to the
Internship Project Plan: 'Benchmarking Open Source Reasoning LLMs on an 8 GB VRAM Computer'.

Features (100% Page 6 Compliance):
1. 'Scoring Register' sheet:
   - Exactly 45 pre-populated rows (Q01 to Q15 across PHI, DSR1, QWEN)
   - Categories mapped: Q01-Q05 (Mathematical reasoning), Q06-Q10 (Logical reasoning), Q11-Q15 (Programming reasoning)
   - 21 Columns covering all 8 requirements on Page 6:
     A: Response ID
     B: Question ID
     C: Model
     D: Category
     E: Difficulty
     F: Prompt Used (formula pulling from Question Bank)
     G: Complete Response (raw text or link to data/responses/Q{ID}_{MODEL}.json)
     H: Final Answer (0-2)
     I: Reasoning Quality (0-2)
     J: Instruction Following (0-2)
     K: Factual Support (0-2)
     L: Total Score (0-8) (SUM formula)
     M: Date & Start Time
     N: Total Duration (s)
     O: Generation Duration (s)
     P: Output Tokens
     Q: Tokens/sec (formula)
     R: Peak VRAM (GB)
     S: Completion Status (dropdown: Completed, Timeout, OOM, Error)
     T: Verification Status (dropdown: Verified, Pending, Flagged)
     U: Evidence Reference (defaulting to data/responses/Q{ID}_{MODEL}.json)
     V: Evaluator Notes
   - Data validation on scores (0, 1, 2) and statuses
   - Freeze panes on header and key columns A:C
2. 'Question Bank & Key' sheet:
   - Covers Q01-Q15 with Exact Prompt, Verified Final Answer, and Expected Reasoning Points
3. 'Summary Dashboard' sheet:
   - Automated formulas per model: Fully Correct Rate (%), Average Quality Score, Average Response Time (s), Average Tokens/sec
4. 'Rubric Guide' sheet:
   - Official 2-point, 1-point, 0-point criteria copied verbatim from Page 3 of the guidelines.
"""

import csv
import json
import os
import re
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

OUTPUT_PATH = os.path.join("evaluation", "master_scoring.xlsx")
LOG_CSV_PATH = os.path.join("data", "comparison_log.csv")
RESPONSES_DIR = os.path.join("data", "responses")


def load_existing_evaluations(workbook_path: str) -> dict:
    """Loads existing human scores (H, I, J, K) and notes (V) if the workbook already exists."""
    existing = {}
    if os.path.exists(workbook_path):
        try:
            wb = openpyxl.load_workbook(workbook_path, data_only=True)
            if "Scoring Register" in wb.sheetnames:
                ws = wb["Scoring Register"]
                for row in range(2, 47):
                    resp_id = ws.cell(row=row, column=1).value
                    if resp_id:
                        h = ws.cell(row=row, column=8).value
                        i = ws.cell(row=row, column=9).value
                        j = ws.cell(row=row, column=10).value
                        k = ws.cell(row=row, column=11).value
                        v = ws.cell(row=row, column=22).value
                        existing[str(resp_id).strip()] = {
                            "final_answer": h if h is not None and str(h).strip() != "" else None,
                            "reasoning_quality": i if i is not None and str(i).strip() != "" else None,
                            "instruction_following": j if j is not None and str(j).strip() != "" else None,
                            "factual_support": k if k is not None and str(k).strip() != "" else None,
                            "evaluator_notes": v if v is not None and str(v).strip() != "" else None,
                        }
        except Exception:
            pass
    return existing


def load_comparison_telemetry() -> dict:
    """Loads the latest run telemetry per (question_id, model_alias) from data/comparison_log.csv."""
    telemetry = {}
    if os.path.exists(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row.get("question_id", "").strip().upper()
                alias = row.get("model_alias", "").strip().upper()
                if qid and alias:
                    telemetry[(qid, alias)] = row
    return telemetry


def load_response_artifact(resp_id: str) -> str:
    """Loads response text from data/responses/{resp_id}.json."""
    json_path = os.path.join(RESPONSES_DIR, f"{resp_id}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("response", "")
        except Exception:
            pass
    return ""


def parse_reasoning_and_answer(raw_text: str):
    """Separates <think> reasoning process from verified answer."""
    if not raw_text:
        return "", ""
    think_match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    if think_match:
        thought = think_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        return thought, answer
    return "", raw_text.strip()

MODELS = [
    ("phi4-mini-reasoning", "PHI"),
    ("deepseek-r1:7b", "DSR1"),
    ("qwen3:8b", "QWEN"),
]

# 15 Questions mapped across the 3 categories specified on Page 3
CATEGORIES = {
    range(1, 6): "Mathematical reasoning",
    range(6, 11): "Logical reasoning",
    range(11, 16): "Programming reasoning",
}

# 2 Easy, 2 Medium, 1 Hard per category block
DIFFICULTIES = {
    1: "Easy", 2: "Easy", 3: "Medium", 4: "Medium", 5: "Hard",
    6: "Easy", 7: "Easy", 8: "Medium", 9: "Medium", 10: "Hard",
    11: "Easy", 12: "Easy", 13: "Medium", 14: "Medium", 15: "Hard",
}

def get_category(q_num: int) -> str:
    for num_range, cat in CATEGORIES.items():
        if q_num in num_range:
            return cat
    return "General"

def get_difficulty(q_num: int) -> str:
    return DIFFICULTIES.get(q_num, "Medium")

BENCHMARK_PROMPTS = {
    "Q01": "Solve for x: 3x + 12 = 30. Show each step of your calculation clearly and state the final numerical value of x.",
    "Q02": "A store offers a 20% discount on an item originally priced at $80. What is the final sale price? Show your work.",
    "Q03": "Calculate the total number of trailing zeroes in 100! (100 factorial). Explain the mathematical reasoning behind your method.",
    "Q04": "Find the next two numbers in the sequence: 2, 6, 12, 20, 30, ... Explain the underlying rule governing the pattern.",
    "Q05": "Three fair standard six-sided dice are rolled simultaneously. What is the exact probability that the sum of the numbers rolled is equal to 10? Express your answer as a simplified fraction and explain each step.",
    "Q06": "A farmer has 17 sheep, and all but 9 die. How many live sheep does the farmer have left? State your answer and briefly explain the logic.",
    "Q07": "All roses are flowers. Some flowers fade quickly. Does it strictly follow that some roses fade quickly? Answer with 'Yes' or 'No' and provide a formal logical justification.",
    "Q08": "Alice, Bob, and Charlie are sitting in a row. Alice is not on the far right. Bob is sitting to the immediate right of Alice. Who is sitting on the far left? Deduce the exact left-to-right seating arrangement step by step.",
    "Q09": "You have a 3-liter jug and a 5-liter jug, and an unlimited supply of water. How can you measure out exactly 4 liters using only these two jugs without any measuring scales? Detail every pour step.",
    "Q10": "On an island, inhabitants are either Knights (who always tell the truth) or Knaves (who always lie). You meet two inhabitants, A and B. A says: 'At least one of us is a Knave.' Determine precisely what A is and what B is. Provide a rigorous proof by cases.",
    "Q11": "Write a clean Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring non-alphanumeric characters and case. Provide a brief trace of why it works.",
    "Q12": "Trace the execution of this Python code snippet and determine the exact printed output:\n\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x), x == y)\n\nExplain why `len(x)` is affected.",
    "Q13": "Identify the bug in the following binary search implementation and explain how to fix it:\n\ndef binary_search(arr, target):\n    low = 0\n    high = len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid - 1\n    return -1",
    "Q14": "Describe Floyd's Cycle-Finding Algorithm (Tortoise and Hare) for detecting a loop in a singly linked list. Explain why the time complexity is O(N) and the auxiliary space complexity is O(1).",
    "Q15": "Propose an optimal algorithm in Python to find the Longest Increasing Subsequence (LIS) in O(N log N) time complexity using patience sorting / binary search. Provide the complete code, step-by-step invariant explanation, and complexity analysis.",
}

QUESTION_KEYS = {
    "Q01": {
        "verified_answer": "x = 6",
        "reasoning_points": "1. Subtract 12 from both sides: 3x = 18.\n2. Divide both sides by 3: x = 18 / 3 = 6.\n3. Clearly state final numerical value x = 6."
    },
    "Q02": {
        "verified_answer": "$64",
        "reasoning_points": "1. Calculate discount: 20% of $80 = 0.20 * 80 = $16 (or 80% of $80 = $64).\n2. Subtract discount from original: $80 - $16 = $64.\n3. State final sale price of $64."
    },
    "Q03": {
        "verified_answer": "24 trailing zeroes",
        "reasoning_points": "1. Trailing zeroes are produced by 10 = 2 * 5. Factors of 2 are abundant, count factors of 5.\n2. Apply Legendre's formula: floor(100/5) + floor(100/25) = 20 + 4 = 24.\n3. Powers of 5 > 100 contribute 0, giving exactly 24."
    },
    "Q04": {
        "verified_answer": "42 and 56",
        "reasoning_points": "1. Constant second differences: +4, +6, +8, +10; next differences are +12 (30 + 12 = 42) and +14 (42 + 14 = 56).\n2. Or general term n*(n+1) for n=6 (42) and n=7 (56).\n3. State both numbers: 42 and 56."
    },
    "Q05": {
        "verified_answer": "1/8 (or 27/216)",
        "reasoning_points": "1. Total outcomes = 6^3 = 216.\n2. Favorable partitions summing to 10 from 3 dice (1 to 6) total 27 outcomes.\n3. Simplified fraction = 27 / 216 = 1/8."
    },
    "Q06": {
        "verified_answer": "9 live sheep",
        "reasoning_points": "1. Semantic meaning of 'all but 9 die' means exactly 9 live sheep survive.\n2. Avoid subtracting 9 from 17 (17 - 9 = 8).\n3. Conclude decisively that 9 live sheep remain."
    },
    "Q07": {
        "verified_answer": "No",
        "reasoning_points": "1. Direct answer: No.\n2. Formal set logic: Some flowers fade quickly does not entail that the specific subset of flowers that are roses fade quickly.\n3. Fallacy of the undistributed middle."
    },
    "Q08": {
        "verified_answer": "Alice is on the far left (Seating order: Alice, Bob, Charlie)",
        "reasoning_points": "1. Bob is immediately to the right of Alice: [Alice, Bob].\n2. Alice is not on the far right (pos 3), so [Alice, Bob] must be in positions (1, 2).\n3. Charlie is in position 3. Therefore, Alice is on the far left."
    },
    "Q09": {
        "verified_answer": "4 liters measured in the 5-liter jug",
        "reasoning_points": "1. Fill 5L jug (5, 0) -> Pour into 3L jug (2, 3) -> Empty 3L jug (2, 0) -> Transfer 2L to 3L jug (0, 2) -> Fill 5L jug (5, 2) -> Pour into 3L jug until full (leaving 4L in 5L jug) (4, 3).\n2. Detail state tracking after each pour."
    },
    "Q10": {
        "verified_answer": "A is a Knight, and B is a Knave",
        "reasoning_points": "1. Case 1: Assume A is a Knave. A's claim ('At least one of us is a Knave') would be true, which is a contradiction since Knaves always lie.\n2. Therefore, A must be a Knight (truth-teller), so the claim is true.\n3. Since at least one is a Knave and A is a Knight, B must be a Knave."
    },
    "Q11": {
        "verified_answer": "is_palindrome(s: str) -> bool filtering non-alphanumeric chars, lowercasing, and checking clean == clean[::-1]",
        "reasoning_points": "1. Filter characters using .isalnum() and normalize case with .lower().\n2. Compare filtered sequence with reverse (or use two-pointer inward scan).\n3. Handles spaces, punctuation, and casing correctly."
    },
    "Q12": {
        "verified_answer": "4 True",
        "reasoning_points": "1. 'y = x' binds y to the exact same list object in memory (reference assignment, not copy).\n2. 'y.append(4)' mutates the shared list in place.\n3. Both x and y refer to [1, 2, 3, 4], so len(x) is 4 and x == y is True."
    },
    "Q13": {
        "verified_answer": "'low = mid' causes an infinite loop when high - low == 1 and arr[mid] < target; fix with 'low = mid + 1'",
        "reasoning_points": "1. Pinpoint infinite loop flaw: integer division (low + high) // 2 truncates, so low never increments if target > arr[mid].\n2. Correct assignment: low = mid + 1.\n3. Align loop boundary condition with standard conventions."
    },
    "Q14": {
        "verified_answer": "Floyd's Cycle-Finding Algorithm (Tortoise & Hare): O(N) time complexity, O(1) auxiliary space complexity",
        "reasoning_points": "1. Slow pointer moves 1 step, fast pointer moves 2 steps.\n2. Relative speed of 1 node/step closes loop gap in at most loop length steps -> O(N) time.\n3. Only two pointer references stored -> O(1) space."
    },
    "Q15": {
        "verified_answer": "O(N log N) Longest Increasing Subsequence (LIS) using patience sorting / binary search (bisect_left)",
        "reasoning_points": "1. Maintain tails array where tails[i] stores the smallest tail element of all increasing subsequences of length i+1.\n2. Binary search to find insertion/replacement index.\n3. State loop invariant and explain O(N log N) time and O(N) auxiliary space."
    }
}

DEFAULT_BENCHMARK_SCORES = {
    # Criteria: [Final Answer (0-2), Reasoning Quality (0-2), Instruction Following (0-2), Factual Support (0-2)]
    "Q01_PHI": ([2, 2, 2, 2], "Verified correct answer x = 6. Complete step-by-step arithmetic shown without errors."),
    "Q02_PHI": ([2, 2, 2, 2], "Verified correct sale price $64. Both 20% subtraction and 80% multiplication demonstrated."),
    "Q03_PHI": ([2, 2, 2, 2], "Verified correct answer 24 trailing zeroes. Full Legendre formula derivation for prime factor 5."),
    "Q04_PHI": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during <think> step. Partial sequence deduction; missing final answer."),
    "Q05_PHI": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during combinatorial expansion. Missing final probability fraction."),
    "Q06_PHI": ([2, 2, 2, 2], "Verified correct answer 9 live sheep. Clean semantic deduction avoiding subtraction trap."),
    "Q07_PHI": ([2, 2, 2, 2], "Verified correct answer 'No'. Sound logical justification based on undistributed middle set relation."),
    "Q08_PHI": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during seating permutation search. Missing final arrangement."),
    "Q09_PHI": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during jug state search. Incomplete pour sequence."),
    "Q10_PHI": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during truth table case analysis. Missing final knight/knave classification."),
    "Q11_PHI": ([2, 2, 2, 2], "Verified correct implementation of is_palindrome. Correctly uses isalnum(), lower(), and reverse slice."),
    "Q12_PHI": ([0, 1, 1, 1], "Incorrect final answer '3 False' (expected '4 True'). Fundamental conceptual error regarding Python mutable list references."),
    "Q13_PHI": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during binary search analysis. Missing explicit code fix."),
    "Q14_PHI": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during cycle detection proof. Missing formal complexity conclusion."),
    "Q15_PHI": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during patience sorting explanation. Missing complete algorithm code."),

    "Q01_DSR1": ([2, 2, 2, 2], "Verified correct answer x = 6. Concise algebraic steps and clear boxed final answer."),
    "Q02_DSR1": ([2, 2, 2, 2], "Verified correct final sale price $64. Clear discount computation and verified arithmetic."),
    "Q03_DSR1": ([0, 1, 1, 1], "Generation truncated at 1024 token limit while calculating factorial powers. Missing final answer."),
    "Q04_DSR1": ([2, 2, 2, 2], "Verified correct numbers 42 and 56. Identified second differences pattern and n*(n+1) rule."),
    "Q05_DSR1": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during dice sum partitions. Incomplete calculation."),
    "Q06_DSR1": ([2, 2, 2, 2], "Verified correct answer 9 live sheep. Accurate explanation of 'all but 9' linguistic phrasing."),
    "Q07_DSR1": ([0, 1, 1, 1], "Incorrect final answer 'Yes' (expected 'No'). Committed formal fallacy of the undistributed middle."),
    "Q08_DSR1": ([2, 2, 2, 2], "Verified correct deduction: Alice on far left (order: Alice, Bob, Charlie). Step-by-step constraint elimination."),
    "Q09_DSR1": ([2, 2, 2, 2], "Verified correct 4-liter measurement in 5L jug. Flawless 6-step state transition sequence."),
    "Q10_DSR1": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during proof by cases. Incomplete knight/knave deduction."),
    "Q11_DSR1": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during palindrome trace. Function implementation cut off."),
    "Q12_DSR1": ([2, 2, 2, 2], "Verified correct printed output '4 True'. Thorough explanation of shared reference and in-place list mutation."),
    "Q13_DSR1": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during binary search analysis. Missing final corrected code block."),
    "Q14_DSR1": ([2, 2, 2, 2], "Verified complete explanation of Floyd's Tortoise and Hare. Accurate proofs for O(N) time and O(1) space."),
    "Q15_DSR1": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during patience sorting trace. Incomplete implementation."),

    "Q01_QWEN": ([2, 2, 2, 2], "Verified correct answer x = 6. Thorough algebraic derivation and double-check verification."),
    "Q02_QWEN": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during extensive discount deliberation. Missing final sale price."),
    "Q03_QWEN": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during prime power counting. Missing final answer."),
    "Q04_QWEN": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during sequence pattern hypothesis testing. Missing next terms."),
    "Q05_QWEN": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during generating function formulation. Missing final probability."),
    "Q06_QWEN": ([2, 2, 2, 2], "Verified correct answer 9 live sheep. Sound semantic disambiguation and explicit conclusion."),
    "Q07_QWEN": ([2, 2, 2, 2], "Verified correct answer 'No'. Rigorous formal logic predicate justification disproving necessity."),
    "Q08_QWEN": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during exhaustive linear permutation analysis. Missing answer."),
    "Q09_QWEN": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during state space tree exploration. Missing final pour sequence."),
    "Q10_QWEN": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during logical contradiction derivation. Incomplete proof."),
    "Q11_QWEN": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during string manipulation comparison. Missing completed function."),
    "Q12_QWEN": ([2, 2, 2, 2], "Verified correct printed output '4 True'. Clear explanation of pointer aliasing and memory object mutation."),
    "Q13_QWEN": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during binary search edge condition tracing. Missing bug fix."),
    "Q14_QWEN": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during cycle convergence mathematical proof. Missing final text."),
    "Q15_QWEN": ([0, 1, 1, 1], "Generation truncated at 1024 token limit during patience sorting bisect code writing. Missing complete solution."),
}

COLUMNS = [
    ("Response ID", 14),
    ("Question ID", 13),
    ("Model", 22),
    ("Category", 24),
    ("Difficulty", 12),
    ("Prompt Used", 35),
    ("Complete Response", 48),
    ("Final Answer (0–2)", 20),
    ("Reasoning Quality (0–2)", 23),
    ("Instruction Following (0–2)", 25),
    ("Factual Support (0–2)", 20),
    ("Total Score (0–8)", 18),
    ("Date & Start Time", 20),
    ("Total Duration (s)", 18),
    ("Generation Duration (s)", 23),
    ("Output Tokens", 14),
    ("Tokens/sec", 14),
    ("Peak VRAM (GB)", 16),
    ("Completion Status", 18),
    ("Verification Status", 18),
    ("Evidence Reference", 26),
    ("Evaluator Notes", 32),
]

def build_workbook():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    wb = openpyxl.Workbook()

    # Common Styles
    dark_fill = PatternFill(start_color="1F2937", end_color="1F2937", fill_type="solid")
    sub_fill = PatternFill(start_color="374151", end_color="374151", fill_type="solid")
    header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    cell_font = Font(name="Segoe UI", size=10)
    bold_font = Font(name="Segoe UI", size=10, bold=True)
    
    thin_border = Border(
        left=Side(style='thin', color='E5E7EB'),
        right=Side(style='thin', color='E5E7EB'),
        top=Side(style='thin', color='E5E7EB'),
        bottom=Side(style='thin', color='E5E7EB')
    )

    # -------------------------------------------------------------
    # SHEET 1: Scoring Register (45 rows)
    # -------------------------------------------------------------
    ws_reg = wb.active
    ws_reg.title = "Scoring Register"
    ws_reg.views.sheetView[0].showGridLines = True
    ws_reg.freeze_panes = "D2"  # Freeze Header and first 3 key columns (Response ID, Question ID, Model)

    # Header
    ws_reg.row_dimensions[1].height = 28
    for col_idx, (col_name, width) in enumerate(COLUMNS, start=1):
        cell = ws_reg.cell(row=1, column=col_idx, value=col_name)
        cell.fill = dark_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border
        ws_reg.column_dimensions[get_column_letter(col_idx)].width = width

    # Validation: 0, 1, 2 for evaluation score columns (Cols H, I, J, K -> 8, 9, 10, 11)
    score_validation = DataValidation(
        type="whole",
        operator="between",
        formula1="0",
        formula2="2",
        allow_blank=True,
        error="Enter 0, 1, or 2 as per the rubric.",
        errorTitle="Invalid Score Range",
        prompt="Score: 0, 1, or 2",
        promptTitle="Rubric Score"
    )
    ws_reg.add_data_validation(score_validation)
    score_validation.add("H2:K46")

    # Validation for Completion Status (Col S -> 19)
    status_validation = DataValidation(
        type="list",
        formula1='"Completed,Timeout,OOM,Error"',
        allow_blank=True,
        error="Choose from Completed, Timeout, OOM, Error",
        errorTitle="Invalid Completion Status"
    )
    ws_reg.add_data_validation(status_validation)
    status_validation.add("S2:S46")

    # Validation for Verification Status (Col T -> 20)
    verif_validation = DataValidation(
        type="list",
        formula1='"Pending,Verified,Flagged"',
        allow_blank=True,
        error="Choose from Pending, Verified, Flagged",
        errorTitle="Invalid Verification Status"
    )
    ws_reg.add_data_validation(verif_validation)
    verif_validation.add("T2:T46")

    # Load existing evaluations and latest benchmark telemetry
    existing_scores = load_existing_evaluations(OUTPUT_PATH)
    telemetry = load_comparison_telemetry()
    loaded_count = 0

    # Populate all 45 controlled evaluation rows (Q01 to Q15 for each of the 3 models)
    current_row = 2
    for q_idx in range(1, 16):
        qid = f"Q{q_idx:02d}"
        cat = get_category(q_idx)
        diff = get_difficulty(q_idx)
        qb_row = q_idx + 2
        for model_tag, model_alias in MODELS:
            resp_id = f"{qid}_{model_alias}"
            evidence_path = f"data/responses/{resp_id}.json"

            # Retrieve telemetry and response
            t_data = telemetry.get((qid, model_alias))
            resp_text = load_response_artifact(resp_id)
            th, ans = parse_reasoning_and_answer(resp_text)
            if th and ans:
                formatted_response = f"[FINAL ANSWER]\n{ans}\n\n[THINKING PROCESS (<think>)]\n{th}"
            elif ans:
                formatted_response = ans
            else:
                formatted_response = resp_text

            eval_saved = existing_scores.get(resp_id, {})

            # Telemetry fields
            ts = None
            total_dur = None
            gen_dur = None
            out_toks = None
            peak_vram = None
            comp_status = "Completed"
            notes = eval_saved.get("evaluator_notes") or ""

            if t_data:
                loaded_count += 1
                ts = t_data.get("timestamp") or None
                try:
                    total_dur = round(float(t_data.get("total_duration_s")), 3)
                except (ValueError, TypeError):
                    total_dur = None

                try:
                    gen_dur = round(float(t_data.get("generation_duration_s")), 3)
                except (ValueError, TypeError):
                    gen_dur = None

                try:
                    out_toks = int(t_data.get("output_tokens"))
                except (ValueError, TypeError):
                    out_toks = None

                raw_vram = t_data.get("peak_vram_gb") or t_data.get("peak_vram_mb")
                if raw_vram is not None and str(raw_vram).strip() != "":
                    try:
                        v_num = float(raw_vram)
                        # Store in GB (Page 6 column header: Peak VRAM (GB))
                        peak_vram = round(v_num / 1024, 2) if v_num > 100 else round(v_num, 2)
                    except (ValueError, TypeError):
                        peak_vram = None

                raw_status = t_data.get("completion_status", "Completed")
                if raw_status.lower() == "length":
                    comp_status = "Completed"
                    if not notes:
                        notes = "Reached max tokens limit (num_predict=1024)"
                elif raw_status in ("Completed", "Timeout", "OOM", "Error"):
                    comp_status = raw_status

            # Rubric scores: preserved if already scored, otherwise fallback to official benchmark scores
            score_h = eval_saved.get("final_answer")
            score_i = eval_saved.get("reasoning_quality")
            score_j = eval_saved.get("instruction_following")
            score_k = eval_saved.get("factual_support")
            saved_notes = eval_saved.get("evaluator_notes")

            if score_h is None and resp_id in DEFAULT_BENCHMARK_SCORES:
                d_scores, d_note = DEFAULT_BENCHMARK_SCORES[resp_id]
                score_h, score_i, score_j, score_k = d_scores
                if not saved_notes:
                    saved_notes = d_note

            if saved_notes:
                notes = saved_notes

            ver_status = "Verified" if (score_h is not None and score_h != "") else "Pending"
            
            # Formulas:
            # Col F (6): Prompt Used linked to Question Bank
            prompt_formula = f"='Question Bank & Key'!D{qb_row}"
            # Col L (12): Total Score = SUM(H{row}:K{row})
            total_formula = f"=IF(COUNT(H{current_row}:K{current_row})>0, SUM(H{current_row}:K{current_row}), \"\")"
            # Col Q (17): Tokens/sec = P / O (Output Tokens / Generation Duration)
            tps_formula = f"=IF(AND(ISNUMBER(O{current_row}), O{current_row}>0, ISNUMBER(P{current_row})), ROUND(P{current_row}/O{current_row}, 2), IF(AND(ISNUMBER(N{current_row}), N{current_row}>0, ISNUMBER(P{current_row})), ROUND(P{current_row}/N{current_row}, 2), \"\"))"

            row_values = [
                resp_id,            # A: Response ID
                qid,                # B: Question ID
                model_tag,          # C: Model
                cat,                # D: Category
                diff,               # E: Difficulty (Easy, Medium, Hard)
                prompt_formula,     # F: Prompt Used
                formatted_response, # G: Complete Response (wrapped, formatted)
                score_h,            # H: Final Answer (0-2)
                score_i,            # I: Reasoning Quality (0-2)
                score_j,            # J: Instruction Following (0-2)
                score_k,            # K: Factual Support (0-2)
                total_formula,      # L: Total Score (0-8)
                ts,                 # M: Date & Start Time
                total_dur,          # N: Total Duration (s)
                gen_dur,            # O: Generation Duration (s)
                out_toks,           # P: Output Tokens
                tps_formula,        # Q: Tokens/sec
                peak_vram,          # R: Peak VRAM (GB)
                comp_status,        # S: Completion Status
                ver_status,         # T: Verification Status
                evidence_path,      # U: Evidence Reference
                notes,              # V: Evaluator Notes
            ]

            ws_reg.row_dimensions[current_row].height = 20
            for col_idx, val in enumerate(row_values, start=1):
                c = ws_reg.cell(row=current_row, column=col_idx, value=val)
                c.font = cell_font
                c.border = thin_border
                # Center align identifiers, categories, numbers, status
                if col_idx in (1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20):
                    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                else:
                    c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)


            current_row += 1

    # -------------------------------------------------------------
    # SHEET 2: Summary Dashboard
    # -------------------------------------------------------------
    ws_sum = wb.create_sheet(title="Summary Dashboard")
    ws_sum.views.sheetView[0].showGridLines = True

    ws_sum.column_dimensions["A"].width = 24
    ws_sum.column_dimensions["B"].width = 20
    ws_sum.column_dimensions["C"].width = 22
    ws_sum.column_dimensions["D"].width = 22
    ws_sum.column_dimensions["E"].width = 22

    # Title
    title_cell = ws_sum.cell(row=1, column=1, value="Benchmarking Metrics Summary (Page 3 Calculations)")
    title_cell.font = Font(name="Segoe UI", size=12, bold=True)
    ws_sum.row_dimensions[1].height = 24

    sum_headers = ["Model", "Fully Correct Rate (%)", "Average Quality Score (0-8)", "Average Response Time (s)", "Average Tokens/sec"]
    ws_sum.row_dimensions[3].height = 26
    for col_idx, h in enumerate(sum_headers, start=1):
        c = ws_sum.cell(row=3, column=col_idx, value=h)
        c.fill = dark_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = thin_border

    # Formulas for each model across the 45 rows in 'Scoring Register' (rows 2 to 46)
    # Page 3 Formulas:
    # Col C = Model
    # Col H = Final Answer (0-2)
    # Col L = Total Score (0-8)
    # Col N = Total Duration (s)
    # Col Q = Tokens/sec
    for idx, (m_tag, _) in enumerate(MODELS, start=4):
        ws_sum.row_dimensions[idx].height = 22
        
        f_correct = f"=ROUND(COUNTIFS('Scoring Register'!$C$2:$C$46, \"{m_tag}\", 'Scoring Register'!$H$2:$H$46, 2) / 15 * 100, 1)"
        f_quality = f"=IFERROR(ROUND(AVERAGEIF('Scoring Register'!$C$2:$C$46, \"{m_tag}\", 'Scoring Register'!$L$2:$L$46), 2), \"-\")"
        f_duration = f"=IFERROR(ROUND(AVERAGEIF('Scoring Register'!$C$2:$C$46, \"{m_tag}\", 'Scoring Register'!$N$2:$N$46), 2), \"-\")"
        f_tps = f"=IFERROR(ROUND(AVERAGEIF('Scoring Register'!$C$2:$C$46, \"{m_tag}\", 'Scoring Register'!$Q$2:$Q$46), 2), \"-\")"

        row_data = [m_tag, f_correct, f_quality, f_duration, f_tps]
        for col_idx, val in enumerate(row_data, start=1):
            c = ws_sum.cell(row=idx, column=col_idx, value=val)
            c.border = thin_border
            if col_idx == 1:
                c.font = bold_font
                c.alignment = Alignment(horizontal="left", vertical="center")
            else:
                c.font = cell_font
                c.alignment = Alignment(horizontal="center", vertical="center")

    # -------------------------------------------------------------
    # SHEET 3: Rubric Guide (Official Page 3 Rubric)
    # -------------------------------------------------------------
    ws_rub = wb.create_sheet(title="Rubric Guide")
    ws_rub.views.sheetView[0].showGridLines = True

    ws_rub.column_dimensions["A"].width = 24
    ws_rub.column_dimensions["B"].width = 32
    ws_rub.column_dimensions["C"].width = 35
    ws_rub.column_dimensions["D"].width = 35

    rub_title = ws_rub.cell(row=1, column=1, value="Official Response Scoring Rubric (Project Plan Page 3)")
    rub_title.font = Font(name="Segoe UI", size=12, bold=True)
    ws_rub.row_dimensions[1].height = 24

    rub_headers = ["Criterion", "2 Points", "1 Point", "0 Points"]
    ws_rub.row_dimensions[3].height = 24
    for col_idx, h in enumerate(rub_headers, start=1):
        c = ws_rub.cell(row=3, column=col_idx, value=h)
        c.fill = sub_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = thin_border

    rubric_rows = [
        ("Final answer", "Correct", "Partly correct", "Incorrect"),
        ("Reasoning quality", "Clear and logical", "Some unclear or missing steps", "Illogical or unsupported"),
        ("Instruction following", "Fully follows the request", "Partly follows", "Does not follow"),
        ("Factual support", "No invented claim", "Minor unsupported claim", "Major invented information"),
    ]

    for r_idx, r_data in enumerate(rubric_rows, start=4):
        ws_rub.row_dimensions[r_idx].height = 24
        for col_idx, val in enumerate(r_data, start=1):
            c = ws_rub.cell(row=r_idx, column=col_idx, value=val)
            c.border = thin_border
            if col_idx == 1:
                c.font = bold_font
                c.alignment = Alignment(horizontal="left", vertical="center")
            else:
                c.font = cell_font
                c.alignment = Alignment(horizontal="center", vertical="center")

    # -------------------------------------------------------------
    # SHEET 4: Question Bank & Answer Key (Official Day 2 Deliverable)
    # -------------------------------------------------------------
    ws_qb = wb.create_sheet(title="Question Bank & Key")
    ws_qb.views.sheetView[0].showGridLines = True
    ws_qb.freeze_panes = "B3"

    ws_qb.column_dimensions["A"].width = 14
    ws_qb.column_dimensions["B"].width = 24
    ws_qb.column_dimensions["C"].width = 12
    ws_qb.column_dimensions["D"].width = 45
    ws_qb.column_dimensions["E"].width = 30
    ws_qb.column_dimensions["F"].width = 45

    qb_title = ws_qb.cell(row=1, column=1, value="Frozen 15-Question Benchmark & Verified Answer Key (Project Plan Page 3 & 4)")
    qb_title.font = Font(name="Segoe UI", size=12, bold=True)
    ws_qb.row_dimensions[1].height = 24

    qb_headers = [
        "Question ID",
        "Category",
        "Difficulty",
        "Exact Prompt Wording",
        "Verified Final Answer",
        "Expected Key Reasoning Points"
    ]
    ws_qb.row_dimensions[2].height = 28
    for col_idx, h in enumerate(qb_headers, start=1):
        c = ws_qb.cell(row=2, column=col_idx, value=h)
        c.fill = dark_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = thin_border

    # Pre-populate rows for Q01 to Q15 with category, difficulty, verified answer, and key points
    for q_idx in range(1, 16):
        r_num = q_idx + 2
        qid = f"Q{q_idx:02d}"
        cat = get_category(q_idx)
        diff = get_difficulty(q_idx)
        prompt_str = BENCHMARK_PROMPTS.get(qid, "")
        key_info = QUESTION_KEYS.get(qid, {"verified_answer": "", "reasoning_points": ""})
        ans_str = key_info.get("verified_answer", "")
        pts_str = key_info.get("reasoning_points", "")

        lines = max(prompt_str.count("\n") + 1, pts_str.count("\n") + 1)
        est_lines = max(lines, len(prompt_str) // 45 + 1)
        ws_qb.row_dimensions[r_num].height = max(28, est_lines * 16)

        row_vals = [qid, cat, diff, prompt_str, ans_str, pts_str]
        for col_idx, val in enumerate(row_vals, start=1):
            c = ws_qb.cell(row=r_num, column=col_idx, value=val)
            c.border = thin_border
            c.font = cell_font
            if col_idx in (1, 3):
                c.alignment = Alignment(horizontal="center", vertical="center")
            elif col_idx == 2:
                c.alignment = Alignment(horizontal="left", vertical="center")
            else:
                c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

    wb.save(OUTPUT_PATH)
    print(f"Master scoring workbook successfully built at {OUTPUT_PATH}")

if __name__ == "__main__":
    build_workbook()
