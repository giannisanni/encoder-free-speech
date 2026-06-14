#!/usr/bin/env bash
# Autonomous overnight supervisor (recovery version).
#   - launch trainer immediately from the latest INTACT checkpoint, sampling on
#   - keep it alive: auto-resume from latest ckpt if it dies (capped, no crash-loop)
# Checkpoints are now atomic (.tmp + os.replace), so "latest ckpt_*.pt" is always
# complete. Logs to ~/night_supervisor.log; setsid-launched to survive ssh drops.
cd "$HOME/gemma-csm" || exit 1
LOG="$HOME/night_supervisor.log"
exec >>"$LOG" 2>&1
PAT="gemma_csm.tr""ain_audio_in"
CKPTDIR="checkpoints/a1_speak"

launch() {  # $1 = reason
  local latest
  latest=$(ls -t "$CKPTDIR"/ckpt_*.pt 2>/dev/null | head -1)
  echo "$(date '+%F %T') launch ($1) resume=$latest"
  STEPS=14000 OUT="$CKPTDIR" LOG="$HOME/a1_speak.log" GA=16 \
    MAXA=64 MAXF=90 MAXT=56 DW=0.01 NOAI=1 \
    RESUME="$latest" SAMPLE=2000 SCLIPS=3 SFRAMES=150 \
    SHARDS='data/shards_opens2s/*.pt' bash "$HOME/launch_audioin.sh"
}

echo "=== supervisor start $(date '+%F %T') ==="
launch "initial"
sleep 180
if pgrep -f "$PAT" >/dev/null; then
  echo "$(date '+%F %T') start OK: $(grep -E '^step|resumed from' "$HOME/a1_speak.log" | tail -1)"
else
  echo "$(date '+%F %T') WARN no process 180s after launch"
fi

restarts=0
while true; do
  sleep 120
  if grep -q "^done" "$HOME/a1_speak.log" 2>/dev/null; then
    echo "$(date '+%F %T') TRAINING DONE"; break
  fi
  if ! pgrep -f "$PAT" >/dev/null; then
    if [ "$restarts" -ge 6 ]; then
      echo "$(date '+%F %T') FATAL: restart cap hit; leaving down for human"; break
    fi
    restarts=$((restarts + 1))
    echo "$(date '+%F %T') trainer DEAD; auto-resume #$restarts"
    launch "auto-resume-$restarts"
    sleep 180
  fi
done
echo "=== supervisor end $(date '+%F %T') ==="
