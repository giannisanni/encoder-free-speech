#!/usr/bin/env bash
# Detached launcher for growth_chain.sh (backgrounds inside the script, exits 0
# so ssh returns cleanly — avoids the exit-255 backgrounding quirk).
P="growth_""chain.sh"
pkill -9 -f "$P" 2>/dev/null || true
sleep 1
nohup setsid bash "$HOME/growth_chain.sh" >/dev/null 2>&1 < /dev/null &
sleep 4
if pgrep -f "$P" >/dev/null; then
  echo "GROWTH CHAIN LAUNCHED pid=$(pgrep -f "$P" | head -1)"
else
  echo "GROWTH CHAIN LAUNCH FAILED"
fi
