#!/usr/bin/env bash
# Pod-side training launcher (run as: bash launch.sh). Exists as a FILE so
# ssh command lines never contain process-matchable patterns (the pkill
# self-match bit us three times). Idempotent: kills any prior trainer first.
# Env overrides: STEPS (default 50000) BATCH (16) ACCUM (1) RESUME OUT LOG
set -euo pipefail
STEPS="${STEPS:-50000}"; BATCH="${BATCH:-16}"; ACCUM="${ACCUM:-1}"
RESUME="${RESUME:-checkpoints/m2_sem_finish/ckpt_40000.pt}"
OUT="${OUT:-checkpoints/m3_long}"; LOG="${LOG:-/root/m3.log}"

cd /root/gemma-csm
# kill prior trainer; pattern assembled at runtime, excludes this script
PAT="gemma_csm.tr""ain"
pkill -9 -f "$PAT" 2>/dev/null || true
sleep 2
rm -f "$LOG"
nohup python -m gemma_csm.train \
  --shard data/shards_libritts/libritts_r_*.pt \
          data/shards_libritts_o/libritts_o_*.pt \
          data/shards/ljspeech_13100.pt \
  --val-shard data/shards_val/libritts_dev_*.pt \
  --steps "$STEPS" --eval-every 2000 --val-every 2000 \
  --out "$OUT" --resume "$RESUME" \
  --depth-frac 0.25 --depth-lr-mult 2.0 --depth-weight 0.01 \
  --lora-r 32 --lora-alpha 16 --lora-all-linear \
  --batch-size "$BATCH" --grad-accum "$ACCUM" --keep-ckpts 3 \
  > "$LOG" 2>&1 &
sleep 3
if pgrep -f "$PAT" >/dev/null; then
  echo "LAUNCHED pid=$(pgrep -f "$PAT" | head -1) log=$LOG steps=$STEPS batch=$BATCH"
else
  echo "LAUNCH_FAILED"; tail -5 "$LOG"; exit 1
fi
