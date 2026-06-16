#!/usr/bin/env bash
# RunPod bootstrap for the 12B LONG-CONTEXT TTS run on A100 80GB.
# Installs the audio-IN stack (transformers FROM SOURCE for gemma4_unified +
# Gemma4UnifiedProcessor, torchvision, pillow), pulls current code from GitHub,
# pulls the long-context shards + the warm-start checkpoint from HF.
# On the pod (PyTorch CUDA template):
#   export HF_TOKEN=hf_...    GH_TOKEN=ghp_...
#   bash bootstrap_12b.sh
set -euo pipefail

echo "=== [0/5] sshd ==="
ssh-keygen -A 2>/dev/null || true
mkdir -p /run/sshd && (/usr/sbin/sshd 2>/dev/null || true)

echo "=== [1/5] deps (12B audio-IN stack: transformers from source) ==="
pip -q install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip -q uninstall -y torchaudio 2>/dev/null || true
pip -q install -U "git+https://github.com/huggingface/transformers.git" \
  bitsandbytes peft pillow soundfile numpy accelerate datasets \
  tqdm tensorboard jiwer faster-whisper whisper-normalizer "huggingface_hub[cli]"
python - <<'PY'
import transformers
from transformers.models import gemma4_unified  # must import or 12B won't load
print("transformers", transformers.__version__, "+ gemma4_unified OK")
PY

echo "=== [2/5] code (current from GitHub; private repo needs GH_TOKEN) ==="
if [ -n "${GH_TOKEN:-}" ]; then
  rm -rf ~/gemma-csm
  git clone "https://${GH_TOKEN}@github.com/giannisanni/gemma-csm" ~/gemma-csm
else
  echo "WARN: no GH_TOKEN set; falling back to (possibly stale) HF code tarball"
  mkdir -p ~/gemma-csm && cd ~/gemma-csm
  hf download giannisan/gemma-csm-ckpts code/gemma-csm-code.tar.gz --local-dir /tmp/code
  tar xzf /tmp/code/code/gemma-csm-code.tar.gz -C ~/gemma-csm
fi
cd ~/gemma-csm && mkdir -p .venv/bin && ln -sf "$(which python)" .venv/bin/python
# verify the trainer carries the fixes (fail loud if GitHub was stale)
python - <<'PY'
s = open("gemma_csm/train_audio_in.py").read()
need = {"gradient clipping": "clip_grad_norm_", "LR-override-on-resume": "LR override",
        "batch/num-workers": "functools.partial(collate_tts", "--grad-clip arg": '"--grad-clip"'}
miss = [k for k, v in need.items() if v not in s]
print("trainer fixes present:", "ALL OK" if not miss else "MISSING -> " + ", ".join(miss))
assert not miss, "stale code: missing %s" % miss
PY

echo "=== [3/5] long-context shards + ljspeech from HF ==="
# NOTE: newer huggingface_hub CLI mis-parses multiple --include patterns (treats
# the 2nd as a literal filename -> 404). Use ONE --include per call.
hf download giannisan/gemma-csm-shards --repo-type dataset --local-dir data --include "shards_long/*"
hf download giannisan/gemma-csm-shards --repo-type dataset --local-dir data --include "shards/*" || true
echo "shards_long: $(ls data/shards_long/*.pt 2>/dev/null | wc -l)  ljspeech: $(ls data/shards/*.pt 2>/dev/null | wc -l)"

echo "=== [4/5] warm-start checkpoint from HF (weights only, fresh-opt) ==="
hf download giannisan/gemma-csm-shards --repo-type dataset --local-dir /tmp/ws \
  --include "warmstart/*" || true
mkdir -p checkpoints/tts_speak
if ls /tmp/ws/warmstart/*.pt >/dev/null 2>&1; then
  cp /tmp/ws/warmstart/*.pt checkpoints/tts_speak/
  echo "warm-start ckpt: $(ls -la checkpoints/tts_speak/*warmstart*.pt | awk '{print $NF, $5}')"
else
  echo "WARN: no warm-start ckpt found on HF under warmstart/ -- launch will train from scratch unless you scp one in."
fi

echo "=== [5/5] smoke ==="
python -c "import torch; print('cuda:', torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory//2**30, 'GB')"
cat <<MSG

READY. Launch the 12B long-context run (defaults already set: shards_long, max-frames 280,
r32 warm-start, batch-16, depth_lr_mult 4.0, lr 5e-5, grad-clip 1.0):
  cd ~/gemma-csm && bash scripts/runpod/launch_12b.sh
Watch:  tail -f /root/tts_long.log     (val depth/c0 every 2000 steps)
Stop the pod when done (per-minute billing):  runpodctl remove pod <id>
MSG
