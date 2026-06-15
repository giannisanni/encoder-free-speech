#!/usr/bin/env bash
# Pod-side launcher: 12B Phase-A (text->speech, --tts), r64 + batch-16 + multi-worker.
# The capacity experiment the 16GB card can't run. Env: STEPS BATCH ACCUM WORKERS
# LORA_R MAXF OUT LOG LR RESUME.
set -euo pipefail
STEPS="${STEPS:-40000}"; BATCH="${BATCH:-16}"; ACCUM="${ACCUM:-1}"; WORKERS="${WORKERS:-4}"
LORA_R="${LORA_R:-64}"; LORA_A="${LORA_A:-16}"; MAXF="${MAXF:-150}"
OUT="${OUT:-checkpoints/tts_r64_a100}"; LOG="${LOG:-/root/tts_r64.log}"
LR="${LR:-}"; RESUME="${RESUME:-}"
cd ~/gemma-csm
PAT="gemma_csm.tr""ain_audio_in"
pkill -9 -f "$PAT" 2>/dev/null || true; sleep 2; rm -f "$LOG"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
nohup python -m gemma_csm.train_audio_in \
  --tts --shard "data/shards_libritts*/*.pt" \
  --steps "$STEPS" --out "$OUT" --eval-every 2000 --val-every 2000 \
  --max-frames "$MAXF" --batch-size "$BATCH" --grad-accum "$ACCUM" --num-workers "$WORKERS" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_A" --lora-all-linear \
  --depth-frac 0.25 --depth-lr-mult 2.0 --depth-weight 0.01 \
  --backbone-id google/gemma-4-12B-it --sample-every 0 \
  ${LR:+--lr "$LR"} ${RESUME:+--resume "$RESUME"} \
  > "$LOG" 2>&1 &
sleep 5
if pgrep -f "$PAT" >/dev/null; then
  echo "LAUNCHED pid=$(pgrep -f "$PAT" | head -1) r=$LORA_R batch=$BATCH accum=$ACCUM steps=$STEPS log=$LOG"
else
  echo "LAUNCH_FAILED"; tail -12 "$LOG"; exit 1
fi
