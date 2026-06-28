#!/usr/bin/env bash
# Pod-side Phase 2.2: self-contained (code already extracted from the HF tarball).
# Installs the audio-IN stack, pulls OpenS2S shards + head_v2 warm-start, and
# launches the --unified run. Run DETACHED so it survives proxy drops:
#   setsid bash ~/gemma-csm/scripts/runpod/run_unified_full.sh </dev/null >~/full_boot.log 2>&1 &
set -euo pipefail
cd ~/gemma-csm
echo "=== [1] deps (transformers from source for gemma4_unified) ==="
pip -q install -U "git+https://github.com/huggingface/transformers.git" \
  bitsandbytes peft soundfile numpy accelerate datasets tqdm \
  jiwer faster-whisper whisper-normalizer "huggingface_hub[cli]"
python -c "from transformers.models import gemma4_unified; import transformers; print('transformers', transformers.__version__, '+ gemma4_unified OK')"
mkdir -p .venv/bin && ln -sf "$(which python)" .venv/bin/python

echo "=== [2] OpenS2S shards ==="
hf download giannisan/gemma-csm-shards --repo-type dataset --local-dir data --include "shards_opens2s/*"
echo "shards: $(ls data/shards_opens2s/*.pt 2>/dev/null | wc -l)"

echo "=== [3] head_v2 warm-start (SINGLE explicit file) + strip opt ==="
hf download giannisan/gemma-csm-ckpts head_v2/ckpt_8000.pt --local-dir /tmp/hv2
python - <<'PY'
import torch, os
c = torch.load("/tmp/hv2/head_v2/ckpt_8000.pt", map_location="cpu", weights_only=True)
m = c["model"] if isinstance(c, dict) and "model" in c else c
os.makedirs("checkpoints", exist_ok=True)
torch.save({"model": m, "step": 0}, "checkpoints/head_v2_warmstart.pt")
print("warmstart tensors:", len(m))
PY
rm -rf /tmp/hv2

echo "=== [4] launch unified run (model downloads ~24GB on first load) ==="
rm -f ~/unified_full.log
STEPS="${STEPS:-3000}"; GA="${GA:-4}"; MAXF="${MAXF:-150}"
setsid bash -c "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m gemma_csm.train_audio_in \
  --unified --backbone-id google/gemma-4-12B-it --shard 'data/shards_opens2s/*.pt' \
  --steps $STEPS --out checkpoints/unified_a6000 --eval-every 500 --keep-ckpts 3 \
  --resume checkpoints/head_v2_warmstart.pt \
  --depth-frac 0.25 --depth-lr-mult 2.0 --depth-weight 0.01 \
  --lora-r 32 --lora-alpha 16 --lora-all-linear \
  --max-audio 64 --max-frames $MAXF --max-text 96 --grad-accum $GA --lr 5e-5 --sample-every 0" \
  </dev/null > ~/unified_full.log 2>&1 &
sleep 6
echo "TRAIN_LAUNCHED pid=$(pgrep -f '[t]rain_audio_in' | head -1) steps=$STEPS ga=$GA maxf=$MAXF"
echo "=== BOOTSTRAP_COMPLETE ==="
