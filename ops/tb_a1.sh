#!/usr/bin/env bash
cd ~/gemma-csm
pkill -f "log_to""_tb.*9_a1" 2>/dev/null || true
sleep 1
nohup .venv/bin/python -m gemma_csm.log_to_tb --log "$HOME/a1_speak.log" --run 9_a1_speak --follow > "$HOME/tb_a1.log" 2>&1 &
sleep 5
pgrep -f "log_to""_tb.*9_a1" >/dev/null && echo "TB LOGGER UP pid=$(pgrep -f "log_to""_tb.*9_a1"|head -1)" || { echo FAILED; cat "$HOME/tb_a1.log"; }
