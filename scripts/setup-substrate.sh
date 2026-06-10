#!/usr/bin/env bash
# Run ON substrate. Creates the training venv.
set -euo pipefail
cd ~/gemma-csm
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available())"
