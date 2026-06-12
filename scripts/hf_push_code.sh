#!/usr/bin/env bash
# Push a code snapshot to the HF ckpts repo so RunPod pods can bootstrap
# without GitHub. Run on substrate (has the HF token); called by sync script.
set -euo pipefail
cd ~/gemma-csm
tar czf /tmp/gemma-csm-code.tar.gz \
  --exclude .venv --exclude data --exclude checkpoints --exclude samples \
  --exclude runs_tb --exclude __pycache__ --exclude .pytest_cache .
.venv/bin/python - <<'EOF'
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/tmp/gemma-csm-code.tar.gz",
                    path_in_repo="code/gemma-csm-code.tar.gz",
                    repo_id="giannisan/gemma-csm-ckpts")
print("code snapshot pushed")
EOF
