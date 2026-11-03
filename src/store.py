"""Save results one line at a time, so a run can be stopped and resumed.

Results live in a .jsonl file ("JSON lines"): one JSON object per line.
Every record has a unique "key". To resume, we read the file, collect the keys
that are already there, and only do the jobs whose key is missing.

Why this is safe to interrupt:
  - each record is written and flushed to disk immediately,
  - if a crash leaves a half-written last line, we cut it off before continuing.
"""

import json
import os


def repair_tail(path):
    """If the file does not end with a newline, the last line was cut off mid-write. Remove it."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        data = f.read()
        if not data or data.endswith(b"\n"):
            return
        f.truncate(data.rfind(b"\n") + 1)  # keep everything up to the last complete line


def load_records(path):
    """Return all records in the file (an empty list if the file does not exist yet)."""
    repair_tail(path)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_keys(path, only_ok=False):
    """Keys of records already saved. With only_ok=True, records marked ok=False do not count
    as done, so a resumed run retries them (used for judge parse failures)."""
    return {r["key"] for r in load_records(path) if not only_ok or r.get("ok", True)}


def latest_by_key(records):
    """If a key appears more than once (a retried job), keep the last one."""
    return list({r["key"]: r for r in records}.values())


def append_record(path, record):
    """Append one record and force it onto the disk before returning."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())
