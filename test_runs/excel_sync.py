"""
Proxy module exposing tool/excel_sync.py to test_runs directory without circular import.
"""
import os
import sys
import importlib.util

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR) if os.path.basename(_CURRENT_DIR) == "test_runs" else _CURRENT_DIR
_IMPL_PATH = os.path.join(_PROJECT_ROOT, "tool", "excel_sync.py")

_spec = importlib.util.spec_from_file_location("_tool_excel_sync_impl", _IMPL_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

sync_official_benchmark_to_excel = _mod.sync_official_benchmark_to_excel
flush_pending_syncs = _mod.flush_pending_syncs
get_pending_sync_count = _mod.get_pending_sync_count
is_file_writable = _mod.is_file_writable
format_complete_response = _mod.format_complete_response
parse_reasoning_and_answer = _mod.parse_reasoning_and_answer
normalize_qid = _mod.normalize_qid
normalize_model_alias = _mod.normalize_model_alias

DEFAULT_EXCEL_PATH = _mod.DEFAULT_EXCEL_PATH
RESPONSES_DIR = _mod.RESPONSES_DIR
PENDING_SYNC_PATH = _mod.PENDING_SYNC_PATH

__all__ = [
    "DEFAULT_EXCEL_PATH",
    "RESPONSES_DIR",
    "PENDING_SYNC_PATH",
    "sync_official_benchmark_to_excel",
    "flush_pending_syncs",
    "get_pending_sync_count",
    "is_file_writable",
    "format_complete_response",
    "parse_reasoning_and_answer",
    "normalize_qid",
    "normalize_model_alias",
]
