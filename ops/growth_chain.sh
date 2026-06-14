#!/usr/bin/env bash
# Grow the OpenS2S dataset toward ~10k pairs via disjoint streaming passes.
# Each pass streams a fresh window (skip += K) and shards it. Guards on disk +
# RAM. Synchronous (one pass at a time) so RAM peak stays bounded. Detached.
cd "$HOME/gemma-csm" || exit 1
LOG="$HOME/growth_chain.log"
exec >>"$LOG" 2>&1
PY="$HOME/audioin-venv/bin/python"
SHARDS="data/shards_opens2s"

N=2500          # request per pass (~1500 net after >20s filter)
SS=1000         # shard size
K=3250          # window per pass (= n_pairs*1.3); next skip advances by this
SKIP=4420       # disjoint from prior passes (orig skip0, p1 1300, p2 2860)
SSTART=6        # next free shard index (000-005 used)
PASSES=6

echo "=== growth chain start $(date '+%F %T') (target ~10k pairs) ==="
for i in $(seq 1 $PASSES); do
  FREE=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ "${FREE:-0}" -lt 5 ]; then echo "$(date '+%F %T') STOP: disk ${FREE}G < 5G"; break; fi
  while [ "$(free -g | awk '/Mem/{print $7}')" -lt 12 ]; do
    echo "$(date '+%F %T') RAM low, waiting 60s"; sleep 60; done
  echo "$(date '+%F %T') --- pass $i/$PASSES: skip=$SKIP shard_start=$SSTART n=$N ---"
  $PY -u -m gemma_csm.data_prep_opens2s --stream --n-pairs "$N" --shard-size "$SS" \
    --out "$SHARDS" --device cpu --skip "$SKIP" --shard-start "$SSTART" 2>&1 \
    | grep -avE "Loading weights|FutureWarning|torch._check" | tail -4
  echo "$(date '+%F %T') pass $i done; shards=$(ls $SHARDS/*.pt 2>/dev/null | wc -l) disk_free=$(df -h / | awk '/%/{print $4}')"
  SKIP=$((SKIP + K)); SSTART=$((SSTART + 10))
done

# final pair count across all shards
TOTAL=$($PY - <<'PY' 2>/dev/null
import glob, torch
n=0
for f in glob.glob("data/shards_opens2s/*.pt"):
    try: n+=len(torch.load(f, weights_only=True))
    except Exception: pass
print(n)
PY
)
echo "=== growth chain end $(date '+%F %T') total pairs=$TOTAL shards=$(ls $SHARDS/*.pt 2>/dev/null | wc -l) ==="
