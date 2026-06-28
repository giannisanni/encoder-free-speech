#!/usr/bin/env bash
# Phase 2.1 overfit gate (substrate): warm-start head_v2, heavily overfit a tiny
# 16-item subset with the NATIVE-delivery unified forward, so we can decode and
# confirm the unified model SPEAKS a grounded reply before any long run.
set -euo pipefail
cd ~/gemma-csm

echo "=== [0/3] wait for head_v2 download (DL_DONE) ==="
for i in $(seq 1 90); do
  grep -q DL_DONE ~/dl_head.log 2>/dev/null && break
  grep -qiE "error|no space|traceback" ~/dl_head.log 2>/dev/null && { echo "DOWNLOAD ERROR"; tail -5 ~/dl_head.log; exit 1; }
  sleep 10
done
grep -q DL_DONE ~/dl_head.log 2>/dev/null || { echo "download not done after ~15min"; exit 1; }
echo "download complete."

echo "=== [1/3] strip optimizer from head_v2 ckpt -> fresh-opt warmstart @ step 0 ==="
~/audioin-venv/bin/python - <<'PY'
import torch
c = torch.load("checkpoints/_hf_head_v2/head_v2/ckpt_8000.pt", map_location="cpu", weights_only=True)
m = c["model"] if isinstance(c, dict) and "model" in c else c
torch.save({"model": m, "step": 0}, "checkpoints/unified_warmstart.pt")
print("warmstart saved; model tensors:", len(m))
PY
rm -rf checkpoints/_hf_head_v2                                   # local copy/symlink
rm -rf ~/.cache/huggingface/hub/models--giannisan--gemma-csm-ckpts  # reclaim the 4.2 GB download blob (cron uploads, never reads this)
echo "free after reclaim: $(df -h ~ | tail -1 | awk '{print $4}')"

echo "=== [2/3] build tiny 16-item overfit shard ==="
~/audioin-venv/bin/python - <<'PY'
import os, torch
items = torch.load("data/shards_opens2s/opens2s_en_000.pt", weights_only=True)[:16]
os.makedirs("data/shards_overfit", exist_ok=True)
torch.save(items, "data/shards_overfit/tiny.pt")
print("tiny shard:", len(items), "items; first query:", repr(items[0].get("query_text", ""))[:70])
PY

echo "=== [3/3] launch unified overfit (warm-start, batch1, grad-accum1, 800 steps) ==="
rm -f ~/unified_overfit.log
setsid bash -c "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/audioin-venv/bin/python -u -m gemma_csm.train_audio_in \
  --unified --backbone-id google/gemma-4-12B-it --shard 'data/shards_overfit/tiny.pt' \
  --steps 800 --out checkpoints/unified_overfit --eval-every 2000 --keep-ckpts 1 \
  --resume checkpoints/unified_warmstart.pt \
  --depth-frac 0.25 --depth-lr-mult 2.0 --depth-weight 0.01 \
  --lora-r 32 --lora-alpha 16 --lora-all-linear \
  --max-audio 64 --max-frames 80 --max-text 96 --grad-accum 1 --lr 5e-5 --sample-every 0" \
  </dev/null > ~/unified_overfit.log 2>&1 &
sleep 5
P=$(pgrep -f gemma_csm.train_audio_in | head -1 || true)
[ -n "$P" ] && echo "OVERFIT LAUNCHED pid=$P log=~/unified_overfit.log" || { echo "LAUNCH FAILED"; tail -15 ~/unified_overfit.log; }
