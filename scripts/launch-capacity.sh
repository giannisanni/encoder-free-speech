#!/usr/bin/env bash
# Capacity experiment: ~425M depth transformer (fresh), warm backbone from m1 final.
set -euo pipefail
cd ~/gemma-csm
nohup .venv/bin/python -m gemma_csm.train \
  --shard data/shards_libritts/*.pt data/shards/ljspeech_13100.pt \
  --steps 40000 --eval-every 2000 --eval-at 500,1000 \
  --out checkpoints/m1_cap \
  --resume checkpoints/m1_data_finish/ckpt_2000.pt \
  --depth-frac 0.25 --depth-lr-mult 2.0 \
  --depth-d-model 1536 --depth-layers 8 --depth-heads 12 \
  > ~/m1_cap.log 2>&1 &
echo "launched pid $!"
