#!/usr/bin/env bash
# Post-outage: finish v2's remaining steps from its newest checkpoint.
set -euo pipefail
cd ~/gemma-csm
sudo systemctl stop ollama
CKPT=$(ls -t checkpoints/m0b_v2/ckpt_*.pt | head -1)
echo "resuming v2 finish from $CKPT"
nohup .venv/bin/python -m gemma_csm.train \
  --shard data/shards/ljspeech_13100.pt \
  --steps 8000 --eval-every 2000 --out checkpoints/m0b_v2b \
  --resume "$CKPT" --depth-frac 0.25 --depth-lr-mult 2.0 \
  > ~/m0b_v2b.log 2>&1 &
echo "launched pid $!"
