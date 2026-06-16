#!/usr/bin/env bash
# Pod-side launcher: 12B Phase-A (text->speech, --tts), LONG-CONTEXT on A100 80GB.
# This is the run substrate's 16GB cannot do: long segments (max-frames 280 ~= 22s,
# the m4 recipe that lowered val depth on E2B) + batch-16. Warm-started from the
# substrate checkpoint (weights only, step reset -> fresh optimizer on the pod).
#
# Carries ALL the substrate fixes that were debugged the hard way:
#   - gradient clipping (--grad-clip; prevents the 8-bit-optimizer depth blowup)
#   - LR override on resume (CLI LRs are authoritative; baked into the trainer)
#   - depth_lr_mult 4.0 -> depth LR 2e-4 (Moshi-matched) with base lr 5e-5
#   - expandable_segments (anti-fragmentation; harmless on 80GB, kept for parity)
#   - fresh optimizer (warm-start ckpt has no "opt" key -> resume goes fresh)
# Env-overridable: STEPS BATCH ACCUM WORKERS LORA_R LORA_A MAXF OUT LOG LR RESUME.
set -euo pipefail
STEPS="${STEPS:-40000}"; BATCH="${BATCH:-16}"; ACCUM="${ACCUM:-1}"; WORKERS="${WORKERS:-4}"
LORA_R="${LORA_R:-32}"; LORA_A="${LORA_A:-16}"; MAXF="${MAXF:-280}"
DEPTH_LR_MULT="${DEPTH_LR_MULT:-4.0}"; DEPTH_WEIGHT="${DEPTH_WEIGHT:-0.02}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"; LR="${LR:-5e-5}"
SHARD="${SHARD:-data/shards_long/*.pt}"
OUT="${OUT:-checkpoints/tts_long_a100}"; LOG="${LOG:-/root/tts_long.log}"
RESUME="${RESUME:-checkpoints/tts_speak/ckpt_70000_warmstart.pt}"
cd ~/gemma-csm
PAT="gemma_csm.tr""ain_audio_in"
pkill -9 -f "$PAT" 2>/dev/null || true; sleep 2; rm -f "$LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python -m gemma_csm.train_audio_in \
  --tts --shard "$SHARD" \
  --steps "$STEPS" --out "$OUT" --eval-every 2000 --val-every 2000 \
  --max-frames "$MAXF" --batch-size "$BATCH" --grad-accum "$ACCUM" --num-workers "$WORKERS" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_A" --lora-all-linear \
  --depth-frac 0.25 --depth-lr-mult "$DEPTH_LR_MULT" --depth-weight "$DEPTH_WEIGHT" \
  --grad-clip "$GRAD_CLIP" --lr "$LR" \
  --backbone-id google/gemma-4-12B-it --sample-every 0 \
  ${RESUME:+--resume "$RESUME"} \
  > "$LOG" 2>&1 &
sleep 5
if pgrep -f "$PAT" >/dev/null; then
  echo "LAUNCHED pid=$(pgrep -f "$PAT" | head -1) shard=$SHARD r=$LORA_R batch=$BATCH maxf=$MAXF depthLRmult=$DEPTH_LR_MULT lr=$LR clip=$GRAD_CLIP resume=$RESUME steps=$STEPS log=$LOG"
else
  echo "LAUNCH_FAILED"; tail -15 "$LOG"; exit 1
fi
