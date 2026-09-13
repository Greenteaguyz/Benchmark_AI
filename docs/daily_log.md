# Daily Project Log — 2026-09-12

### Work Completed
- **File Naming & Response Schema**: Standardized response format (`data/responses/Q{ID}_{MODEL}.json`) defined in [`docs/response_schema.json`](./response_schema.json) with helper module [`tool/response_schema.py`](../tool/response_schema.py).
- **Master Scoring Sheet**: Created [`evaluation/master_scoring.xlsx`](../evaluation/master_scoring.xlsx) with 12 evaluation metrics, automated sum scoring, and 0–2 score validation.
- **Environment Setup**: Initialized directories (`data/responses`, `evaluation`, `tool`, `docs`) and fixed UTF-8 subprocess handling in `setup_env.py`.

### Model Documentation

| Package (Tag) | Alias | Parameters | Size (Disk) | Purpose in Benchmark | Official Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `phi4-mini-reasoning` | `PHI` | 3.84B | ~3.2 GB | Lightweight reasoning baseline for constrained hardware | [Ollama Library](https://ollama.com/library/phi4-mini-reasoning) |
| `deepseek-r1:7b` | `DSR1` | 7.62B | ~4.7 GB | Compact distilled reasoning model | [Ollama Library](https://ollama.com/library/deepseek-r1) |
| `qwen3:8b` | `QWEN` | 8.19B | ~5.2 GB | Larger general reasoning comparison model | [Ollama Library](https://ollama.com/library/qwen3) |
