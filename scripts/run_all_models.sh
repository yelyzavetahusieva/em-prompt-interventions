#!/bin/bash
# Run a stage's generate step for all three models, one after another. Run this ON THE POD.
#
#   nohup bash scripts/run_all_models.sh pilot > run.log 2>&1 &          # then check:  cat run.log
#   nohup bash scripts/run_all_models.sh baseline_audit > run.log 2>&1 &
#
# Optional: MODELS="extreme_sports bad_medical_advice" bash scripts/run_all_models.sh stage3b   (run only some models)
# STAGE is the name of a script in scripts/ that has a "generate --model NAME" step.
# Every model's work is saved as it goes, and re-running this skips finished answers.

STAGE=$1
shift
if [ -z "$STAGE" ]; then
    echo "Usage: bash scripts/run_all_models.sh STAGE [extra args for generate]"
    exit 1
fi

cd "$(dirname "$0")/.." && . .venv/bin/activate
for m in ${MODELS:-base extreme_sports bad_medical_advice}; do
    echo "=== $m started $(date +%H:%M:%S)"
    python -m scripts.$STAGE generate --model $m "$@" 2>&1 | grep -E "jobs already done|done \||Error|error|Traceback|OutOfMemory"
    echo "=== $m finished $(date +%H:%M:%S)"
done
echo "ALL DONE"
