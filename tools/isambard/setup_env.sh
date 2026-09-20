#!/usr/bin/env bash
# Bootstrap the `vfe-torch` environment on Isambard-AI (aarch64 / GH200, sm_90).
#
# Run from the repo root on a LOGIN node:
#     bash tools/isambard/setup_env.sh
#
# Notes for this machine (see docs/rewrite-plan.md "Training target"):
#   * Isambard's driver is 565.57.01 -> CUDA 12.7 native. We pin torch 2.10.0
#     +cu128, which runs on any >=12.0 driver (CUDA minor version
#     compatibility). Do NOT jump to torch 2.11+, which is CUDA 13.
#   * PyPI's default aarch64 torch wheel is CPU-ONLY. pyproject.toml therefore
#     routes torch/torchvision through download.pytorch.org/whl/cu128.
#   * uv is used rather than conda: all our deps have aarch64 wheels, so there
#     is nothing to compile and conda buys us nothing here.
#   * Never run `conda init` on Isambard; never install into conda `base`.
set -euo pipefail

UV="${UV:-$HOME/.local/bin/uv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -x "$UV" ]]; then
    echo "uv not found at $UV -- install it first:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

cd "$REPO_ROOT"

echo "==> creating .venv (Python 3.12) in $REPO_ROOT"
"$UV" venv --python 3.12 .venv

echo "==> installing pinned torch stack + this package (editable)"
# shellcheck disable=SC1091
source .venv/bin/activate
"$UV" pip install -e ".[log,dev]"

echo "==> done. Smoke-test on a GPU node with:"
echo "    srun --account=brics.b5cs --gpus=1 --ntasks=1 --time=00:05:00 \\"
echo "         $REPO_ROOT/.venv/bin/python tools/checks/torch_smoke.py"
