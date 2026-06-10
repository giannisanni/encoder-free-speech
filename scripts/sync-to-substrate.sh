#!/usr/bin/env bash
# Push working tree (not data/checkpoints) Mac -> substrate.
set -euo pipefail
rsync -av --delete \
  --exclude .git --exclude .venv --exclude data --exclude checkpoints \
  --exclude samples --exclude __pycache__ --exclude .pytest_cache \
  --exclude '*.log' \
  ~/workspace/gemma-csm/ gianni@substrate:~/gemma-csm/
