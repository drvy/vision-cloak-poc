#!/usr/bin/env bash
# Sync the project to a remote host (no venv, no outputs): scripts/deploy.sh <host>
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az --delete --exclude .venv --exclude 'out*' --exclude samples --exclude __pycache__ ./ "${1:?usage: deploy.sh <host>}":~/cloak/
