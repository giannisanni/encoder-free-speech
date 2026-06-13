#!/usr/bin/env bash
# RunPod pod bootstrap: zero to training in ~5 minutes, no substrate needed.
# Everything pulls from HF: code snapshot, dataset shards, resume checkpoint.
#
# On the pod (use a PyTorch 2.x CUDA template, e.g. runpod/pytorch):
#   export HF_TOKEN=hf_...           # or set as pod env var at creation
#   bash <(curl -sL https://huggingface.co/giannisan/gemma-csm-ckpts/resolve/main/scripts-bootstrap.sh)
# or copy this file over and run it. Args: [resume_ckpt_path_in_repo]
set -euo pipefail
RESUME="${1:-m2_sem_finish/ckpt_40000.pt}"

echo "=== [1/4] deps ==="
pip -q install -U "transformers>=4.46" peft bitsandbytes datasets soundfile \
  tqdm tensorboard jiwer faster-whisper whisper-normalizer "huggingface_hub[cli]" && (huggingface-cli version >/dev/null 2>&1 || true)

echo "=== [2/4] code ==="
mkdir -p ~/gemma-csm && cd ~/gemma-csm
hf download giannisan/gemma-csm-ckpts code/gemma-csm-code.tar.gz \
  --local-dir /tmp/code
tar xzf /tmp/code/code/gemma-csm-code.tar.gz -C ~/gemma-csm
# pods have a bare python env, no .venv: shim the scripts' interpreter path
mkdir -p .venv/bin && ln -sf "$(which python)" .venv/bin/python

echo "=== [3/4] data shards ==="
hf download giannisan/gemma-csm-shards --repo-type dataset \
  --local-dir data
echo "=== [3b] resume checkpoint: $RESUME ==="
mkdir -p "checkpoints/$(dirname "$RESUME")"
hf download giannisan/gemma-csm-ckpts "$RESUME" \
  --local-dir checkpoints

echo "=== [4/4] smoke test ==="
python -c "import torch; print('cuda:', torch.cuda.get_device_name(0))"
python -m pytest tests/ -x -q -m "not slow and not gpu" | tail -1

cat <<MSG

READY. Launch e.g.:
  cd ~/gemma-csm && nohup python -m gemma_csm.train \\
    --shard data/shards_libritts/libritts_r_*.pt \\
            data/shards_libritts_o/libritts_o_*.pt \\
            data/shards/ljspeech_13100.pt \\
    --val-shard data/shards_val/libritts_dev_*.pt \\
    --steps 120000 --eval-every 2000 --keep-ckpts 3 \\
    --resume checkpoints/$RESUME \\
    --depth-frac 0.25 --depth-lr-mult 2.0 --depth-weight 0.01 \\
    --lora-r 32 --lora-alpha 16 --lora-all-linear \\
    --out checkpoints/cloud_run > ~/train.log 2>&1 &
A100 tip: in gemma_csm/config.py raise batch_size (2->8+) and cut grad_accum
(8->2) for the same effective batch at far higher throughput.
Upload results back:  python scripts/hf_archive.py  (edit ROOT first) or
huggingface-cli upload giannisan/gemma-csm-ckpts checkpoints/cloud_run cloud_run
MSG
