#!/bin/bash
# Run every test of the Stage 5 tooling (CPU only, a few minutes). Run from the repo root:  bash scripts/run_tests.sh
# They check the code on tiny random models and simulated data; none of them needs a GPU or the judge.
cd "$(dirname "$0")/.." && . .venv/bin/activate
status=0
for t in tests/test_*.py; do
    echo "=== $t"
    PYTHONPATH=. python "$t" 2>&1 | grep -E "^(PASS|FAIL)|ALL PASS|SOME FAILED|Error|Traceback" || status=1
    [ "${PIPESTATUS[0]}" -ne 0 ] && status=1
done
[ $status -eq 0 ] && echo "ALL TEST FILES PASSED" || echo "SOMETHING FAILED"
exit $status
