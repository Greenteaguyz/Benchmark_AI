#!/usr/bin/env python3
"""One-time repair for comparison_log.csv corrupted by records merged onto one line.

Splits the raw text back on record-start timestamps, verifies every row has the
same number of fields as the header, writes a backup, and validates the result
round-trips through pd.read_csv.
"""
import csv
import io
import os
import re
import shutil
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "test_runs", "comparison_log.csv")
BACKUP_PATH = CSV_PATH + ".bak"

RECORD_START = re.compile(r"(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},Q)")


def repair() -> None:
    if not os.path.exists(CSV_PATH):
        print("No CSV to repair:", CSV_PATH)
        sys.exit(1)

    text = open(CSV_PATH, encoding="utf-8").read()
    parts = RECORD_START.split(text)
    header_rows = list(csv.reader(io.StringIO(parts[0])))
    header = header_rows[0]
    ncols = len(header)
    lines = [parts[0].rstrip("\r\n")]
    for part in parts[1:]:
        rows = list(csv.reader(io.StringIO(part)))
        if len(rows) != 1 or len(rows[0]) != ncols:
            print("Unrecoverable record (needs manual fix):", repr(part[:120]))
            sys.exit(1)
        lines.append(",".join(map(str, rows[0])))

    shutil.copy2(CSV_PATH, BACKUP_PATH)
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        for line in lines:
            f.write(line + "\n")

    df = pd.read_csv(CSV_PATH)
    print(f"Repaired {CSV_PATH}: {len(df)} rows x {df.shape[1]} cols")
    print(f"Backup saved: {BACKUP_PATH}")


if __name__ == "__main__":
    repair()