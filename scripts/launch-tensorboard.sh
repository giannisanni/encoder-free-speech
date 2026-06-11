#!/usr/bin/env bash
# Backfill all run logs into TensorBoard and serve on :6006 (tailscale-reachable).
set -euo pipefail
cd ~/gemma-csm
PY=.venv/bin/python
$PY -m gemma_csm.log_to_tb --log ~/m0b.log     --run 1_m0b_baseline || true
$PY -m gemma_csm.log_to_tb --log ~/m0b_v2.log  --run 2_m0b_v2_depthfix || true
$PY -m gemma_csm.log_to_tb --log ~/m0b_v2b.log --run 2_m0b_v2_depthfix --step-offset 32000 || true
nohup $PY -m gemma_csm.log_to_tb --log ~/m1_data.log --run 3_m1_multispeaker --follow \
  > ~/tb_follow.log 2>&1 &
nohup .venv/bin/tensorboard --logdir runs_tb --bind_all --port 6006 \
  > ~/tensorboard.log 2>&1 &
echo "tensorboard launching on :6006"
