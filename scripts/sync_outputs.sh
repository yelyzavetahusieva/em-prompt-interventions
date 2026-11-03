#!/bin/bash
# Copy results between your Mac and a Runpod pod.
#
#   bash scripts/sync_outputs.sh pull HOST PORT   # pod -> Mac   (do this often, and BEFORE terminating a pod)
#   bash scripts/sync_outputs.sh push HOST PORT   # Mac -> pod   (do this on a new pod, to resume a run)
#
# HOST and PORT come from the pod's SSH command, "ssh root@HOST -p PORT".
# Run from the repo root.
#
# Safety: results files are append-only, so we use rsync --append. It only
# adds the missing end of a file. If the destination copy is the same size or longer
# than the source, that file is skipped. So a fresh, empty pod can never overwrite
# results you already have, and nothing is ever deleted.

set -e

if [ "$#" -ne 3 ] || { [ "$1" != "pull" ] && [ "$1" != "push" ]; }; then
    echo "Usage: bash scripts/sync_outputs.sh pull|push HOST PORT"
    exit 1
fi

MODE=$1
HOST=$2
PORT=$3
REMOTE="root@${HOST}:/root/em-prompt-interventions/outputs/"
LOCAL="outputs/"
SSH="ssh -p ${PORT}"

mkdir -p "$LOCAL"

if [ "$MODE" = "pull" ]; then
    rsync -av --append -e "$SSH" "$REMOTE" "$LOCAL"
else
    rsync -av --append -e "$SSH" "$LOCAL" "$REMOTE"
fi
