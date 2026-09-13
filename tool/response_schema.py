"""
Response Schema Helper and Validator
Standardized response serialization for Q{ID}_{MODEL}.json files.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Canonical mapping of model identifiers to standardized filename abbreviations
MODEL_ALIASES: Dict[str, str] = {
    "phi4-mini-reasoning": "PHI",
    "deepseek-r1:7b": "DSR1",
    "deepseek-r1": "DSR1",
    "qwen3:8b": "QWEN",
    "qwen2.5:7b": "QWEN",
}

RESPONSES_DIR = os.path.join("data", "responses")


def get_model_alias(model_name: str) -> str:
    """Return the standardized abbreviation for a model name."""
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    
    # Fallback: clean and uppercase tag prefix
    cleaned = model_name.split(":")[0].replace("-", "").upper()
    return cleaned[:4]


def format_response_filename(question_id: str, model_name: str) -> str:
    """
    Format standard filename pattern: Q{ID}_{MODEL}.json
    Example: Q01_PHI.json, Q01_DSR1.json, Q01_QWEN.json
    """
    alias = get_model_alias(model_name)
    qid = question_id.upper()
    if not qid.startswith("Q"):
        qid = f"Q{qid}"
    return f"{qid}_{alias}.json"


def get_response_filepath(question_id: str, model_name: str, base_dir: str = RESPONSES_DIR) -> str:
    """Return the relative or absolute path where the response JSON will be saved."""
    filename = format_response_filename(question_id, model_name)
    return os.path.join(base_dir, filename)


def create_response_payload(
    question_id: str,
    model: str,
    prompt: str,
    response: str,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a dictionary conforming to the minimal response schema."""
    qid = question_id.upper()
    if not qid.startswith("Q"):
        qid = f"Q{qid}"

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "question_id": qid,
        "model": model,
        "prompt": prompt,
        "response": response,
        "timestamp": timestamp,
    }


def save_response(
    question_id: str,
    model: str,
    prompt: str,
    response: str,
    base_dir: str = RESPONSES_DIR,
) -> str:
    """Save a model response to data/responses/Q{ID}_{MODEL}.json."""
    os.makedirs(base_dir, exist_ok=True)
    payload = create_response_payload(question_id, model, prompt, response)
    target_path = get_response_filepath(question_id, model, base_dir)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return target_path


def load_response(filepath: str) -> Dict[str, Any]:
    """Load and return a response JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
