#!/usr/bin/env bash
# RunPod bootstrap for the 12B audio-IN / TTS run. Unlike the m-series bootstrap,
# this installs the audio-IN stack: transformers FROM SOURCE (needs gemma4_unified
# + Gemma4UnifiedProcessor) + torchvision + pillow. Pulls code + LibriTTS shards
# from HF. On the pod (PyTorch CUDA template):
#   export HF_TOKEN=hf_...
#   bash bootstrap_12b.sh
set -euo pipefail

echo "=== [0/4] sshd ==="
ssh-keygen -A 2>/dev/null || true
mkdir -p /run/sshd && (/usr/sbin/sshd 2>/dev/null || true)

echo "=== [1/4] deps (12B audio-IN stack: transformers from source) ==="
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

echo "=== [2/4] code (current from GitHub; private repo needs GH_TOKEN) ==="
if [ -n "${GH_TOKEN:-}" ]; then
  git clone "https://${GH_TOKEN}@github.com/giannisanni/gemma-csm" ~/gemma-csm
else
  echo "WARN: no GH_TOKEN set; falling back to (possibly stale) HF code tarball"
  mkdir -p ~/gemma-csm && cd ~/gemma-csm
  hf download giannisan/gemma-csm-ckpts code/gemma-csm-code.tar.gz --local-dir /tmp/code
  tar xzf /tmp/code/code/gemma-csm-code.tar.gz -C ~/gemma-csm
fi
cd ~/gemma-csm && mkdir -p .venv/bin && ln -sf "$(which python)" .venv/bin/python

echo "=== [3/4] LibriTTS shards from HF ==="
hf download giannisan/gemma-csm-shards --repo-type dataset --local-dir data \
  --include "shards_libritts/*" "shards_libritts_o/*" "shards/*"
echo "shards: $(ls data/shards_libritts*/*.pt 2>/dev/null | wc -l) multispeaker + $(ls data/shards/*.pt 2>/dev/null | wc -l) ljspeech"

echo "=== [4/4] smoke ==="
python -c "import torch; print('cuda:', torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory//2**30, 'GB')"
cat <<MSG

READY. Launch the 12B r64/batch-16 experiment:
  cd ~/gemma-csm && bash scripts/runpod/launch_12b.sh
Watch: tail -f /root/tts_r64.log   (val c0 every 2000 steps)
Stop the pod when done (billing is per-minute): runpodctl remove pod <id>
MSG
