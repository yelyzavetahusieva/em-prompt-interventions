"""Freeze (or check) the prompt templates and question files.

    python -m scripts.freeze write "pilot reviewed, wording approved"   # do this once, after the pilot
    python -m scripts.freeze check                                       # before the main run: exits with an error if anything changed
"""

import sys

from src.freeze import FROZEN_PATH, verify_freeze, write_freeze


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("write", "check"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "write":
        record = write_freeze(note=" ".join(sys.argv[2:]))
        print(f"Froze {len(record['files'])} files on {record['frozen_on']} -> {FROZEN_PATH}")
        for path in record["files"]:
            print("  ", path)
    else:
        problems = verify_freeze()
        if problems:
            raise SystemExit("NOT FROZEN STATE:\n  " + "\n  ".join(problems))
        print("OK: templates and questions match the freeze.")


if __name__ == "__main__":
    main()
