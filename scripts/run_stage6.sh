#!/bin/bash
# Stage 6 on the pod, one phase at a time.   nohup bash scripts/run_stage6.sh PHASE > run.log 2>&1 &     then:  tail run.log
#
#   calibration   activations of the calibration answers, both adapters (needs outputs/stage6 pushed, freeze phase 1)   -> pull -> Mac: directions, freeze --phase 2 -> push
#   pilot         Gate 9: the first 300 Geodesic questions (x2 samples), the three conditions
#   full          everything else: Geodesic (all questions, or LIMIT=N questions) and the 100 EM-extension questions, three conditions
#   project       project every answer on the 12 directions + 100 random ones, and extract the base-model prompt states
#
# Environment: BATCH (default 8) for generation, PBATCH (default 4) for the extraction steps, LIMIT (default: all) for "full".
# Every step saves as it goes and skips finished work, so re-running after an interruption is safe.

cd "$(dirname "$0")/.." && . .venv/bin/activate
BATCH=${BATCH:-8}; PBATCH=${PBATCH:-4}; MODELS="base bad_medical_advice extreme_sports"
step() { echo "=== $1 started $(date +%H:%M:%S)"; shift; "$@" 2>&1 | grep -E "jobs already done|done \||saved|exists|DONE|PASS|FAIL|STOP|Error|error|Traceback|OutOfMemory"; echo "=== finished $(date +%H:%M:%S)"; }

case "$1" in
  calibration)
    for a in bad_medical_advice extreme_sports; do step "calibration $a" python -m scripts.stage6_pod calib-extract --adapter $a --batch-size $PBATCH; done ;;
  pilot)
    for m in $MODELS; do step "pilot $m" python -m scripts.stage6_pod generate --dataset geodesic --model $m --limit 300 --batch-size $BATCH; done ;;
  full)
    for m in $MODELS; do
      step "extension $m" python -m scripts.stage6_pod generate --dataset extension --model $m --batch-size $BATCH
      step "geodesic $m"  python -m scripts.stage6_pod generate --dataset geodesic --model $m --limit ${LIMIT:-100000} --batch-size $BATCH
    done ;;
  project)
    for m in $MODELS; do for d in extension geodesic; do step "project $d $m" python -m scripts.stage6_pod project --dataset $d --model $m --batch-size $PBATCH; done; done
    step "base prompt states" python -m scripts.stage6_pod states --batch-size $PBATCH ;;
  *) echo "Usage: bash scripts/run_stage6.sh calibration|pilot|full|project"; exit 1 ;;
esac
echo "ALL DONE"
