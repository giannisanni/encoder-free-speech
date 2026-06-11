#!/usr/bin/env bash
# M2-sem: attack the semantic stream (c0), the evidence-backed bottleneck.
#   - resume m1 final (150M depth; capacity eliminated empirically + by
#     Mimi codebook entropy, AudioCodecBench arXiv 2509.02349)
#   - LoRA r=64 on attention+MLP: the only new-modality config with published
#     success (Make Some Noise arXiv 2503.22275); r16 attn-only is the
#     documented-failing one (arXiv 2401.00246). NOTE: old r16 attn-only
#     adapters shape-mismatch on resume and restart fresh; audio embeddings,
#     c0 head, and depth decoder stay warm.
#   - depth loss x0.01 (Moshi's 100:1 semantic:acoustic, arXiv 2410.00037):
#     stop spending gradient on the ~6.7-nat codec noise floor.
#   - 80k steps ~= 2 more epochs; comparable successes ran 9-15 total.
# Run on substrate inside tmux: bash ~/gemma-csm/scripts/launch-m2-sem.sh
set -euo pipefail
cd ~/gemma-csm
nohup .venv/bin/python -m gemma_csm.train \
  --shard data/shards_libritts/libritts_r_*.pt data/shards/ljspeech_13100.pt \
  --steps 80000 --eval-every 2000 \
  --out checkpoints/m2_sem \
  --resume checkpoints/m1_data_finish/ckpt_2000.pt \
  --depth-frac 0.25 --depth-lr-mult 2.0 \
  --depth-weight 0.01 \
  --lora-r 64 --lora-alpha 16 --lora-all-linear \
  > ~/m2_sem.log 2>&1 &
echo "m2_sem launched, log: ~/m2_sem.log"
