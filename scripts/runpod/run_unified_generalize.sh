#!/usr/bin/env bash
# Pod-side unified GENERALIZATION run: fix the OOD template collapse with
# (a) emotion-balanced OpenS2S (broad behavioral distribution, real reply audio)
# +  (b) VoiceAssistant-400K general Q&A (real query audio + Kokoro-TTS'd replies,
#        teaches factual/identity/general answering AND head voicing breadth)
# +  (c) a LIGHTER LoRA (lower rank, all-linear kept) so the adapter masks the
#        frozen backbone's reasoning less.
# Run DETACHED so it survives proxy drops:
#   setsid bash ~/gemma-csm/scripts/runpod/run_unified_generalize.sh </dev/null >~/gen_boot.log 2>&1 &
set -euo pipefail
cd ~/gemma-csm

STEPS="${STEPS:-4000}"; GA="${GA:-4}"; MAXF="${MAXF:-150}"
LORA_R="${LORA_R:-16}"; LORA_ALPHA="${LORA_ALPHA:-8}"; COVERAGE="${COVERAGE:-all}"
VA_N="${VA_N:-15000}"; VOICE="${VOICE:-af_heart}"

echo "=== [1] deps (transformers-from-source + kokoro TTS + espeak) ==="
apt-get -qq -y install espeak-ng >/dev/null 2>&1 || echo "espeak-ng apt skipped (check phonemizer)"
pip -q install -U "git+https://github.com/huggingface/transformers.git" \
  bitsandbytes peft soundfile numpy accelerate datasets tqdm \
  jiwer faster-whisper whisper-normalizer "huggingface_hub[cli]" \
  "kokoro>=0.9.4" misaki
python -c "from transformers.models import gemma4_unified; import transformers; print('transformers', transformers.__version__)"
mkdir -p .venv/bin && ln -sf "$(which python)" .venv/bin/python

echo "=== [2] balanced OpenS2S shards (pre-built on substrate, in HF shards repo) ==="
hf download giannisan/gemma-csm-shards --repo-type dataset --local-dir data --include "shards_opens2s_balanced/*"
echo "balanced shards: $(ls data/shards_opens2s_balanced/*.pt 2>/dev/null | wc -l)"

echo "=== [3] build VoiceAssistant shards on-pod (Kokoro TTS, voice=$VOICE, n=$VA_N) ==="
rm -f ~/va_build.log
python -u -m gemma_csm.data_prep_voiceassistant --out data/shards_va \
  --n-pairs "$VA_N" --voice "$VOICE" --device cuda \
  --backbone-id google/gemma-4-12B-it --max-seconds 20 --max-words 70 2>&1 | tee ~/va_build.log
echo "VA shards: $(ls data/shards_va/*.pt 2>/dev/null | wc -l)"

echo "=== [4] head_v2 warm-start (single explicit file) + strip opt ==="
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

echo "=== [5] launch unified GENERALIZE run (mixed shards + lighter LoRA) ==="
echo "    LoRA: r=$LORA_R alpha=$LORA_ALPHA coverage=$COVERAGE | steps=$STEPS ga=$GA maxf=$MAXF"
rm -f ~/unified_gen.log
setsid bash -c "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m gemma_csm.train_audio_in \
  --unified --backbone-id google/gemma-4-12B-it \
  --shard 'data/shards_opens2s_balanced/*.pt' 'data/shards_va/*.pt' \
  --steps $STEPS --out checkpoints/unified_gen --eval-every 500 --keep-ckpts 3 \
  --resume checkpoints/head_v2_warmstart.pt \
  --depth-frac 0.25 --depth-lr-mult 2.0 --depth-weight 0.01 \
  --lora-r $LORA_R --lora-alpha $LORA_ALPHA --lora-coverage $COVERAGE \
  --max-audio 64 --max-frames $MAXF --max-text 96 --grad-accum $GA --lr 5e-5 --sample-every 0" \
  </dev/null > ~/unified_gen.log 2>&1 &
sleep 6
echo "TRAIN_LAUNCHED pid=$(pgrep -f '[t]rain_audio_in' | head -1)"
echo "=== GEN_BOOTSTRAP_COMPLETE ==="
echo "After training, GATE on the OOD battery (not just c0):"
echo "  python -m scripts.ablated_s2s --queries ~/ood_wavs/q*.wav ~/promo_q.wav --ckpt unified_gen/ckpt_4000.pt"
echo "  success = OOD content correct AND audio WER well below ~0.9 at FULL LoRA"
