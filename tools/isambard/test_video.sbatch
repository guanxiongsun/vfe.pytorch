#!/bin/bash
# Evaluate a video-detector checkpoint on ImageNet VID val with the
# pure-PyTorch stack, on one Isambard-AI node.
#
# Needs the data staged by stage_imagenet_vid.sbatch and the repo's .venv.
# Submit from the repo root:
#
#   Full evaluation (4 GPUs):
#     CONFIG=configs/vid/mamba/mamba_r101_dc5_6x.py \
#     CKPT=/projects/b5cs/vfe/checkpoints/released/mamba_r101_dc5_6x_epoch_6_model.pth \
#     sbatch tools/isambard/test_video.sbatch
#
#   Smoke test first (1 GPU, 15 min): one video, no metric, plus the data
#   pipeline parity artifacts for a cross-machine comparison:
#     CONFIG=... CKPT=... SMOKE=1 sbatch --gpus=1 --time=00:15:00 tools/isambard/test_video.sbatch
#
#SBATCH --job-name=vfe_test
#SBATCH --account=brics.b5cs
#SBATCH --output=vfe_test_%j.out
#SBATCH --gpus=4
#SBATCH --time=01:30:00
set -euo pipefail

: "${CONFIG:?set CONFIG to a config path}"
: "${CKPT:?set CKPT to a checkpoint path}"
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
DATA="${DATA:-/projects/b5cs/imagenet_vid/ILSVRC}"
WORK="${WORK:-$SCRATCHDIR/vfe/work_dirs/test_$(basename "$CONFIG" .py)_${SLURM_JOB_ID}}"
PY="$REPO/.venv/bin/python"
cd "$REPO"

# The configs read data_root "data/ILSVRC/", relative to the repo.
if [[ ! -e data/ILSVRC ]]; then
    mkdir -p data
    ln -s "$DATA" data/ILSVRC
fi
mkdir -p "$WORK"
NGPUS="$(nvidia-smi -L | wc -l)"
echo "==> $(date -Is) $CONFIG, $NGPUS GPU(s), work dir $WORK"

if [[ -n "${SMOKE:-}" ]]; then
    # Frame-level artifacts to diff against the x86 legacy run
    # (tools/checks/parity_vid_pipeline.py --compare).
    "$PY" tools/checks/parity_vid_pipeline.py --impl vfe --out "$WORK/pipeline_gh200.pt"
    "$PY" -m vfe.cli.test "$CONFIG" "$CKPT" --work-dir "$WORK" --max-videos 1 --seed 0 \
        --out "$WORK/detections_video0.pkl"
else
    # --seed 0 makes MAMBA's memory sampling repeatable; the protocol is unchanged.
    "$REPO/.venv/bin/torchrun" --standalone --nproc_per_node="$NGPUS" -m vfe.cli.test \
        "$CONFIG" "$CKPT" --launcher pytorch --work-dir "$WORK" \
        --out "$WORK/detections.pkl" --seed 0 --workers 8
fi
echo "==> $(date -Is) done"
