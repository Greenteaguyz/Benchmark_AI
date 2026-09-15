"""Cross-process safe CSV append used by app.py (Streamlit) & web_server.py.

Guards concurrent appends with an fcntl.flock on a sidecar `.lock` file and
always guarantees the CSV ends with a newline before appending, so records
can never be merged onto one line again.
"""
import contextlib
import csv
import fcntl
import io
import os


@contextlib.contextmanager
def _locked(lock_path: str):
    f = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _record_bytes(row: dict, fieldnames) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", restval="", lineterminator="\n")
    writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def append_record_row(record: dict, csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with _locked(csv_path + ".lock"):
        size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
        if size == 0:
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(record.keys()), extrasaction="ignore", restval="", lineterminator="\n")
                writer.writeheader()
                writer.writerow(record)
            return

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), None)
        fieldnames = header if header else list(record.keys())
        row_bytes = _record_bytes(record, fieldnames)

        with open(csv_path, "ab+") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            if last not in (b"\n", b"\r"):
                f.write(b"\n")
            f.write(row_bytes)