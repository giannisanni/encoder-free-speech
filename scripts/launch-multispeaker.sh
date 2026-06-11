#!/usr/bin/env bash
# Launch the multi-speaker run (LibriTTS-R + LJSpeech), resuming from the
# newest v2 checkpoint. Lives in the repo so sync-to-substrate preserves it.
set -euo pipefail
cd ~/gemma-csm
CKPT=$(ls -t checkpoints/m0b_v2*/ckpt_*.pt | head -1)   # prefers v2b (post-outage finish) when present
echo "resuming from $CKPT"
nohup .venv/bin/python -m gemma_csm.train \
  --shard data/shards_libritts/*.pt data/shards/ljspeech_13100.pt \
  --steps 40000 --eval-every 2000 --eval-at 3000,3500 \
  --out checkpoints/m1_data \
  --resume "$CKPT" --depth-frac 0.25 --depth-lr-mult 2.0 \
  > ~/m1_data.log 2>&1 &
echo "launched pid $!"
