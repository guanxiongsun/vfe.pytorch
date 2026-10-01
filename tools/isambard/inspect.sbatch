#!/bin/bash
# The EOVOD frame inspector (tools/eovod_inspector/) on one Isambard-AI GPU:
# a web page of per-frame dumps, and new dumps run on this GPU on request.
#
# Submit from the repo root, then read the job's output for the tunnel command
# and the link (it carries an access token):
#     sbatch tools/isambard/inspect.sbatch
#     grep -A4 "On your laptop" eovod_inspect_<jobid>.out
# On the laptop, keep the tunnel open while you use the page:
#     ssh -N -L 8765:<node>:8765 b5cs.aip2.isambard
# and open http://localhost:8765/?token=...   Cancel the job when done
# (scancel <jobid>): it holds a GPU until its time limit otherwise.
#
#SBATCH --job-name=eovod_inspect
#SBATCH --account=brics.b5cs
#SBATCH --output=eovod_inspect_%j.out
#SBATCH --gpus=1
#SBATCH --time=04:00:00
set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
DATA="${DATA:-/projects/b5cs/imagenet_vid/ILSVRC}"
ROOT="${ROOT:-/projects/b5cs/vfe/viz}"
WORK_DIRS="${WORK_DIRS:-/projects/b5cs/vfe/work_dirs}"
PORT="${PORT:-8765}"
export TORCH_HOME="${TORCH_HOME:-/projects/b5cs/vfe/torch_home}"
cd "$REPO"

# The configs read data_root "data/ILSVRC/", relative to the repo.
if [[ ! -e data/ILSVRC ]]; then
    mkdir -p data
    ln -s "$DATA" data/ILSVRC
fi
exec "$REPO/.venv/bin/python" -u tools/eovod_inspector/server.py \
    --root "$ROOT" --work-dirs "$WORK_DIRS" --port "$PORT"
