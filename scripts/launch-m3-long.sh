#!/usr/bin/env bash
# M3-long: the epochs run. Doubled corpus (~840h, ~3,500 speakers), resumed
# from m2 final (same LoRA geometry -> no reset shock). New instruments:
#   - val loss on held-out dev-clean shards (overfitting sentinel)
#   - checkpoint rotation + HF archiver keep disk safe for multi-day runs
# Pre-registered stop: if word recall does not improve era-over-era across
# this run, the frozen-backbone design is retired (see paper rev. 11).
# Run on substrate: bash ~/gemma-csm/scripts/launch-m3-long.sh
set -euo pipefail
cd ~/gemma-csm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup .venv/bin/python -m gemma_csm.train \
  --shard data/shards_libritts/libritts_r_*.pt \
          data/shards_libritts_o/libritts_o_*.pt \
          data/shards/ljspeech_13100.pt \
  --val-shard data/shards_val/libritts_dev_*.pt \
  --steps 120000 --eval-every 2000 --val-every 2000 \
  --out checkpoints/m3_long \
  --resume checkpoints/m2_sem_finish/ckpt_40000.pt \
  --depth-frac 0.25 --depth-lr-mult 2.0 --depth-weight 0.01 \
  --lora-r 32 --lora-alpha 16 --lora-all-linear \
  --keep-ckpts 3 \
  > ~/m3_long.log 2>&1 &
echo "m3_long launched, log: ~/m3_long.log"
