#!/bin/bash
# Prepare a fresh Runpod pod: copy the code and install the requirements.
#
#   bash scripts/setup_pod.sh HOST PORT
#
# HOST and PORT come from the pod's SSH command, "ssh root@HOST -p PORT". Run from the repo root.
# The .env file (your API key), git history, old outputs and third-party data are NOT copied:
# generation needs none of them, and a rented machine should never hold your key.

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: bash scripts/setup_pod.sh HOST PORT"
    exit 1
fi
HOST=$1
PORT=$2
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -p ${PORT}"

echo "1/3 copying code..."
rsync -az --exclude .venv --exclude .git --exclude .env --exclude outputs --exclude data/validation --exclude __pycache__ \
    -e "ssh ${SSH_OPTS}" ./ "root@${HOST}:/root/em-prompt-interventions/"

echo "2/3 creating the environment (re-using the pod's PyTorch)..."
# -n: stop ssh from reading our stdin
ssh -n ${SSH_OPTS} "root@${HOST}" 'cd /root/em-prompt-interventions \
    && python3 -m venv --system-site-packages .venv && . .venv/bin/activate \
    && pip install -q -r requirements.txt 2>&1 | grep -v -i notice | tail -3'

echo "3/3 checking the GPU..."
ssh -n ${SSH_OPTS} "root@${HOST}" 'cd /root/em-prompt-interventions && . .venv/bin/activate \
    && python -c "import torch, transformers, peft; print(\"GPU:\", torch.cuda.get_device_name(0), \"| torch\", torch.__version__, \"| transformers\", transformers.__version__, \"| peft\", peft.__version__)"'
echo "Pod ready."
