#!/usr/bin/env bash
# Detached launcher for night_supervisor.sh. Backgrounds INSIDE the script then
# exits 0, so ssh returns cleanly (avoids the exit-255 backgrounding quirk).
P="night_super""visor.sh"
pkill -9 -f "$P" 2>/dev/null || true
sleep 1
nohup setsid bash "$HOME/night_supervisor.sh" >/dev/null 2>&1 < /dev/null &
sleep 4
if pgrep -f "$P" >/dev/null; then
  echo "SUPERVISOR LAUNCHED pid=$(pgrep -f "$P" | head -1)"
else
  echo "SUPERVISOR LAUNCH FAILED"
fi
