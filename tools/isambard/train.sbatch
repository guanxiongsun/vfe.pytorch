#!/bin/bash
# Train a video detector with the pure-PyTorch stack on one Isambard-AI node.
#
# 4 GH200s with ACCUMULATE=2 train exactly as the original 8 GPUs did (one
# loader and one random stream per original rank; see vfe/apis/train.py).
# Checkpoints and logs go to $WORK; the job always resumes from
# $WORK/latest.pth when there is one, so a run longer than one job can be
# continued by submitting again (or chaining with --dependency=afterany:<id>).
#
# Needs the data staged by stage_imagenet_vid.sbatch, the repo's .venv, and
# the pretrained backbone in $TORCH_HOME (compute nodes may lack internet);
# populate that once from a login node:
#     mkdir -p /projects/b5cs/vfe/torch_home/hub/checkpoints
#     (download resnet101-63fe2227.pth into it)
#
# Submit from the repo root:
#   Full run:   CONFIG=configs/vid/mamba/mamba_r101_dc5_6x.py sbatch tools/isambard/train.sbatch
#   Short run:  CONFIG=... NAME=mamba_m2 EXTRA="--max-epochs 1 --max-iters-per-epoch 500 --no-validate" \
#                   sbatch --time=00:45:00 tools/isambard/train.sbatch
#
#SBATCH --job-name=vfe_train
#SBATCH --account=brics.b5cs
#SBATCH --output=vfe_train_%j.out
#SBATCH --gpus=4
#SBATCH --time=12:00:00
set -euo pipefail

: "${CONFIG:?set CONFIG to a config path}"
NAME="${NAME:-$(basename "$CONFIG" .py)}"
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
DATA="${DATA:-/projects/b5cs/imagenet_vid/ILSVRC}"
WORK="${WORK:-/projects/b5cs/vfe/work_dirs/$NAME}"
SEED="${SEED:-1466607766}"  # the seed in the original MAMBA training log
ACCUMULATE="${ACCUMULATE:-2}"
EXTRA="${EXTRA:-}"
export TORCH_HOME="${TORCH_HOME:-/projects/b5cs/vfe/torch_home}"
cd "$REPO"

# The configs read data_root "data/ILSVRC/", relative to the repo.
if [[ ! -e data/ILSVRC ]]; then
    mkdir -p data
    ln -s "$DATA" data/ILSVRC
fi
mkdir -p "$WORK"
NGPUS="$(nvidia-smi -L | wc -l)"
echo "==> $(date -Is) $CONFIG as $NAME: $NGPUS GPU(s) x $ACCUMULATE micro-step(s), work dir $WORK"

# EXTRA is a list of options, split on purpose.
# shellcheck disable=SC2086
"$REPO/.venv/bin/torchrun" --standalone --nproc_per_node="$NGPUS" -m vfe.cli.train "$CONFIG" \
    --launcher pytorch --accumulate "$ACCUMULATE" --seed "$SEED" --work-dir "$WORK" \
    --resume-from auto $EXTRA
echo "==> $(date -Is) done"
