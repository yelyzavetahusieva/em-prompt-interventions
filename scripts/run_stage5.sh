#!/bin/bash
# Stage 5 on the pod, in one go: Part A generation (both adapters), then the activation extraction for Part B.
#
#   nohup bash scripts/run_stage5.sh > run.log 2>&1 &       # then check:  tail run.log
#
# Needs outputs/ pushed first (bash scripts/sync_outputs.sh push HOST PORT): the prompt list, the reference sample, the older stages' answers.
# Every step saves as it goes / skips finished work, so re-running after an interruption is safe.
# Order: cheap and decisive first (Part A), then the main extraction, then the optional Arm B last.

cd "$(dirname "$0")/.." && . .venv/bin/activate
step() { echo "=== $1 started $(date +%H:%M:%S)"; shift; "$@" 2>&1 | grep -E "jobs already done|done \||saved|answers|DONE|PASS|FAIL|Error|error|Traceback|OutOfMemory"; echo "=== finished $(date +%H:%M:%S)"; }

python -m scripts.stage5 render 2>&1 | grep -E "PASS|FAIL|Error"
MODELS="extreme_sports bad_medical_advice" step "Part A generation" bash scripts/run_all_models.sh stage5
step "extract base"            python -m scripts.stage5_extract --space base
step "extract sports adapter"  python -m scripts.stage5_extract --space extreme_sports
step "extract medical adapter" python -m scripts.stage5_extract --space bad_medical_advice
step "Arm B answers (base)"    python -m scripts.stage5_extract --space base --answers
echo "ALL DONE"
