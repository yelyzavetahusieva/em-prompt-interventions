"""Freeze the prompt templates and question files, so the main run provably uses what the pilot approved.

The plan says: after the pilot, freeze templates and the question bank, and do not tune wording on the
confirmation set. `write_freeze` records a SHA-256 fingerprint of every file; `verify_freeze` later lists
any file that no longer matches (changed, added or removed).
"""

import glob
import hashlib
import json
from datetime import datetime, timezone

FROZEN_PATH = "configs/frozen.json"
PATTERNS = ["data/templates/*.json", "data/questions/*.json"]


def fingerprints():
    """{relative path: sha256 of the file's bytes} for every frozen file."""
    return {
        path: hashlib.sha256(open(path, "rb").read()).hexdigest()
        for pattern in PATTERNS
        for path in sorted(glob.glob(pattern))
    }


def write_freeze(path=FROZEN_PATH, note=""):
    record = {"frozen_on": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note, "files": fingerprints()}
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return record


def verify_freeze(path=FROZEN_PATH):
    """Return a list of problems (empty = everything matches the freeze)."""
    with open(path) as f:
        frozen = json.load(f)["files"]
    now = fingerprints()
    problems = [f"changed: {p}" for p in frozen if p in now and now[p] != frozen[p]]
    problems += [f"removed: {p}" for p in frozen if p not in now]
    problems += [f"added: {p}" for p in now if p not in frozen]
    return problems
