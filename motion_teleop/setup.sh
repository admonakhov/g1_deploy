#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_dir"
device="${1:-auto}"
if [[ "$device" != auto && "$device" != cpu && "$device" != xpu ]]; then
  echo 'Usage: bash motion_teleop/setup.sh [auto|cpu|xpu]' >&2
  exit 2
fi
git submodule update --init TRACE vitpose
./uv sync --locked --extra motion
./uv run --locked --extra motion python -m motion_teleop.setup_voice
./uv venv --allow-existing .cache/trace-venv --python 3.11
torch_variant=cpu
if [[ "$device" == xpu ]] || { [[ "$device" == auto ]] && compgen -G '/dev/dri/renderD*' >/dev/null; }; then
  torch_variant=xpu
fi
if ! ./uv pip install --python .cache/trace-venv/bin/python "torch==2.11.0+$torch_variant" --index-url "https://download.pytorch.org/whl/$torch_variant"; then
  echo 'Intel XPU installation unavailable; installing CPU PyTorch.' >&2
  ./uv pip install --python .cache/trace-venv/bin/python 'torch==2.11.0+cpu' --index-url https://download.pytorch.org/whl/cpu
fi
./uv pip install --python .cache/trace-venv/bin/python -r TRACE/requirements.txt -r TRACE/requirements-hub.txt
.cache/trace-venv/bin/python -m motion_teleop.trace_worker --probe --device "$device" | tee .cache/trace-device.json
.cache/trace-venv/bin/python TRACE/tools/hub_assets.py download \
  --model-repo kangzejian/TRACE --data-repo kangzejian/TRACE-Inference-Assets
.cache/trace-venv/bin/python TRACE/run.py --check-models
echo 'Ready. Start: ./uv run --extra motion python -m motion_teleop.server'
