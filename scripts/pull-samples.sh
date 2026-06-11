#!/usr/bin/env bash
# Pull eval clips substrate -> Mac (cron-safe). Mirrors
# checkpoints/<run>/samples/*_s1.wav  to  samples/<run>/  for every run.
set -euo pipefail
echo "$(date '+%F %T') tick"
DEST=~/workspace/gemma-csm/samples
for run in $(ssh -o BatchMode=yes -o ConnectTimeout=10 gianni@substrate \
    'ls -d ~/gemma-csm/checkpoints/*/samples 2>/dev/null' \
    | awk -F/ "{print \$(NF-1)}"); do
  mkdir -p "$DEST/$run"
  rsync -q --ignore-existing \
    "gianni@substrate:~/gemma-csm/checkpoints/$run/samples/*_s1.wav" \
    "$DEST/$run/" 2>/dev/null || true
done
