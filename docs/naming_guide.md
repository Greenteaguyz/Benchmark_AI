# Response Naming Convention

*Deliverable for Day 1: Student 1 (Hout Chanvireak)*  
*Conforms strictly to Project Plan Page 6: Response naming convention*

---

All 45 complete and unedited model responses are saved in `data/responses/` using the format:  
**`Q{ID}_{MODEL}.json`**

### Naming Mapping

| Example ID | Meaning | Saved Filename |
| :--- | :--- | :--- |
| **Q04 PHI** | Question 4 answered by Phi 4 Mini Reasoning | `data/responses/Q04_PHI.json` |
| **Q04 DSR1** | Question 4 answered by DeepSeek R1 7B | `data/responses/Q04_DSR1.json` |
| **Q04 QWEN** | Question 4 answered by Qwen3 8B | `data/responses/Q04_QWEN.json` |

---

### Scope & Structure
- **Question Range:** `Q01` to `Q15` (15 questions total).
- **Total Dataset:** 15 questions $\times$ 3 models = **45 JSON response files**.
- **Directory:** `data/responses/`
- **Schema Reference:** Supported by [`docs/response_schema.json`](./response_schema.json) and [`tool/response_schema.py`](../tool/response_schema.py).
