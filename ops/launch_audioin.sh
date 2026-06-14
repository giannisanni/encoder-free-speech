#!/usr/bin/env bash
# 12B audio-IN (RQ1) training launcher (substrate, audio-IN venv). File-based to
# keep the pkill pattern off any ssh command line. Env overrides:
#   STEPS OUT LOG GA(grad-accum) MAXA(max-audio) MAXF(max-frames) DW(depth-weight)
set -euo pipefail
cd ~/gemma-csm
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256   # bnb-compatible anti-frag
PAT="gemma_csm.tr""ain_audio_in"
pkill -9 -f "$PAT" 2>/dev/null || true
sleep 2
STEPS="${STEPS:-60000}"; OUT="${OUT:-checkpoints/a1_audioin}"; LOG="${LOG:-$HOME/a1.log}"
GA="${GA:-16}"; MAXA="${MAXA:-64}"; MAXF="${MAXF:-150}"; DW="${DW:-0.01}"
SHARDS="${SHARDS:-data/shards_opens2s/*.pt}"
RESUME="${RESUME:-}"; SAMPLE="${SAMPLE:-0}"; SCLIPS="${SCLIPS:-3}"; SFRAMES="${SFRAMES:-150}"
[ -z "$RESUME" ] && rm -f "$LOG"
nohup ~/audioin-venv/bin/python -m gemma_csm.train_audio_in \
  --shard "$SHARDS" \
  --steps "$STEPS" --out "$OUT" --eval-every 2000 --keep-ckpts 3 \
  --depth-frac 0.25 --depth-lr-mult 2.0 --depth-weight "$DW" \
  --lora-r 32 --lora-alpha 16 --lora-all-linear \
  --max-audio "$MAXA" --max-frames "$MAXF" --max-text "${MAXT:-96}" --grad-accum "$GA" ${NOAI:+--no-audio-in} \
  ${RESUME:+--resume "$RESUME"} --sample-every "$SAMPLE" --sample-clips "$SCLIPS" --sample-frames "$SFRAMES" \
  >> "$LOG" 2>&1 &
sleep 4
if pgrep -f "$PAT" >/dev/null; then
  echo "A1 LAUNCHED pid=$(pgrep -f "$PAT" | head -1) log=$LOG steps=$STEPS GA=$GA caps=a$MAXA/f$MAXF dw=$DW"
else
  echo "LAUNCH_FAILED"; tail -20 "$LOG"; exit 1
fi
