"""Run a list of generation jobs and save each answer as soon as it exists.

A job is a dict with at least:
    "key":      unique string, e.g. "extreme_sports|ethics_01|c00|s0"
    "messages": chat messages, e.g. [{"role": "user", "content": "..."}]
plus any other fields to keep (question_id, condition_id, ...). They are copied
into the saved record.

This file does not touch any model. The model is hidden behind `generate_fn`:
    generate_fn(list_of_messages, seed) -> list of dicts, one per input, each with "output"
    (the answer text) and optionally other fields to save, such as "prompt_tokens" and "output_tokens"
(for a real model, see make_hf_generate_fn in models.py), which also makes this
loop easy to test without a GPU.
"""

import time
import zlib
from datetime import datetime, timezone

from src.store import append_record, load_done_keys


def seed_for(key):
    """A fixed seed derived from a job key (the first job of each batch)."""
    return zlib.crc32(key.encode())


def generate_all(jobs, out_path, generate_fn, meta=None, batch_size=1):
    """Run every job whose key is not already in out_path. Safe to interrupt and re-run.

    Jobs are processed `batch_size` at a time (a GPU can generate many answers at once for
    roughly the cost of one). Each answer is saved as soon as its batch finishes, so an
    interruption loses at most one batch.

    `meta` is a dict stored on every record (model name, revisions, decoding settings).
    """
    keys = [job["key"] for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate job keys: every job needs a unique key")

    done = load_done_keys(out_path)
    todo = [job for job in jobs if job["key"] not in done]
    print(f"{len(done)} jobs already done, {len(todo)} to do (batch size {batch_size}) -> {out_path}")

    start = time.time()
    finished = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        seed = seed_for(batch[0]["key"])  # one seed per batch, saved with every record
        results = generate_fn([job["messages"] for job in batch], seed)
        for job, result in zip(batch, results):
            record = {
                **job,
                **result,
                **(meta or {}),
                "seed": seed,
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            append_record(out_path, record)  # saved to disk right now
        finished += len(batch)
        per_job = (time.time() - start) / finished
        print(f"  {finished}/{len(todo)} done | {per_job:.2f}s per answer | about {per_job * (len(todo) - finished) / 60:.0f} min left")
