"""Run a list of generation jobs and save each answer as soon as it exists.

A job is a dict with at least:
    "key":      unique string, e.g. "extreme_sports|ethics_01|c00|s0"
    "messages": chat messages, e.g. [{"role": "user", "content": "..."}]
plus any other fields to keep (question_id, condition_id, ...). They are copied
into the saved record.

This file does not touch any model. The model is hidden behind `generate_fn`:
    generate_fn(messages, seed) -> dict with "output" (the answer text) and optionally
    other fields to save, such as "prompt_tokens" and "output_tokens"
(for a real model, see make_hf_generate_fn in models.py), which also makes this
loop easy to test without a GPU.
"""

import zlib
from datetime import datetime, timezone

from src.store import append_record, load_done_keys


def seed_for(key):
    """A fixed seed per job, derived from its key: re-running a job gives the same sampling."""
    return zlib.crc32(key.encode())


def generate_all(jobs, out_path, generate_fn, meta=None):
    """Run every job whose key is not already in out_path. Safe to interrupt and re-run.

    `meta` is a dict stored on every record (model name, revisions, decoding settings).
    """
    keys = [job["key"] for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate job keys: every job needs a unique key")

    done = load_done_keys(out_path)
    todo = [job for job in jobs if job["key"] not in done]
    print(f"{len(done)} jobs already done, {len(todo)} to do -> {out_path}")

    for n, job in enumerate(todo, start=1):
        result = generate_fn(job["messages"], seed_for(job["key"]))
        record = {
            **job,
            **result,
            **(meta or {}),
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        append_record(out_path, record)  # saved to disk right now
        if n % 10 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} done")
