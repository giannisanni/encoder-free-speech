#!/usr/bin/env bash
cd ~/gemma-csm
P="data_prep""_opens2s"; pkill -9 -f "$P" 2>/dev/null || true; sleep 2
N="${N:-1000}"; SS="${SS:-500}"; DEV="${DEV:-cuda}"
rm -f /tmp/shard.log /tmp/shard.done
nohup ~/audioin-venv/bin/python -u -m gemma_csm.data_prep_opens2s \
  --stream --n-pairs "$N" --shard-size "$SS" --out data/shards_opens2s --device "$DEV" --skip "${SKIP:-0}" --shard-start "${SSTART:-0}" \
  > /tmp/shard.log 2>&1 &
PID=$!
# detach watcher: record exit + done flag when the python finishes
( wait $PID; echo "EXIT=$?" >> /tmp/shard.log; touch /tmp/shard.done ) &
sleep 4
kill -0 $PID 2>/dev/null && echo "SHARD LAUNCHED pid=$PID n=$N dev=$DEV" || { echo FAILED; tail -10 /tmp/shard.log; }
