#!/usr/bin/env bash
# Create .venv with a Pascal-compatible torch build (cu126 keeps sm_61; cu128+ dropped it).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
if [[ "${1:-}" == "cpu" ]]; then
  .venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  .venv/bin/pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
fi
.venv/bin/pip install -q -r requirements.txt
