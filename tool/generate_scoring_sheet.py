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

import os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

OUTPUT_PATH = os.path.join("evaluation", "master_scoring.xlsx")

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

COLUMNS = [
    ("Response ID", 14),
    ("Question ID", 13),
    ("Model", 22),
    ("Category", 24),
    ("Difficulty", 12),
    ("Prompt Used", 35),
    ("Complete Response", 35),
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
            
            # Formulas:
            # Col F (6): Prompt Used linked to Question Bank
            prompt_formula = f"='Question Bank & Key'!D{qb_row}"
            # Col L (12): Total Score = SUM(H{row}:K{row})
            total_formula = f"=IF(COUNT(H{current_row}:K{current_row})>0, SUM(H{current_row}:K{current_row}), \"\")"
            # Col Q (17): Tokens/sec = IF(N{row}>0, P{row}/N{row}, "")
            tps_formula = f"=IF(AND(ISNUMBER(N{current_row}), N{current_row}>0, ISNUMBER(P{current_row})), ROUND(P{current_row}/N{current_row}, 2), \"\")"

            row_values = [
                resp_id,        # A: Response ID
                qid,            # B: Question ID
                model_tag,      # C: Model
                cat,            # D: Category
                diff,           # E: Difficulty (Easy, Medium, Hard)
                prompt_formula, # F: Prompt Used
                "",             # G: Complete Response (or summary)
                None,           # H: Final Answer (0-2)
                None,           # I: Reasoning Quality (0-2)
                None,           # J: Instruction Following (0-2)
                None,           # K: Factual Support (0-2)
                total_formula,  # L: Total Score (0-8)
                None,           # M: Date & Start Time
                None,           # N: Total Duration (s)
                None,           # O: Generation Duration (s)
                None,           # P: Output Tokens
                tps_formula,    # Q: Tokens/sec
                None,           # R: Peak VRAM (GB)
                "Completed",    # S: Completion Status
                "Pending",      # T: Verification Status
                evidence_path,  # U: Evidence Reference
                "",             # V: Evaluator Notes
            ]

            ws_reg.row_dimensions[current_row].height = 20
            for col_idx, val in enumerate(row_values, start=1):
                c = ws_reg.cell(row=current_row, column=col_idx, value=val)
                c.font = cell_font
                c.border = thin_border
                # Center align identifiers, categories, numbers, status
                if col_idx in (1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20):
                    c.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    c.alignment = Alignment(horizontal="left", vertical="center")

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
        f_quality = f"=ROUND(AVERAGEIF('Scoring Register'!$C$2:$C$46, \"{m_tag}\", 'Scoring Register'!$L$2:$L$46), 2)"
        f_duration = f"=ROUND(AVERAGEIF('Scoring Register'!$C$2:$C$46, \"{m_tag}\", 'Scoring Register'!$N$2:$N$46), 2)"
        f_tps = f"=ROUND(AVERAGEIF('Scoring Register'!$C$2:$C$46, \"{m_tag}\", 'Scoring Register'!$Q$2:$Q$46), 2)"

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

    # Pre-populate rows for Q01 to Q15 with category and difficulty
    for q_idx in range(1, 16):
        r_num = q_idx + 2
        ws_qb.row_dimensions[r_num].height = 28
        qid = f"Q{q_idx:02d}"
        cat = get_category(q_idx)
        diff = get_difficulty(q_idx)

        row_vals = [qid, cat, diff, "", "", ""]
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
