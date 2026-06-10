#!/usr/bin/env bash
# Run ON substrate before training. Frees ~10 GB by stopping Ollama
# (which pins the Hermes local-fallback model). Hermes itself stays up:
# its primary model is Ollama Cloud, so only the local fallback is
# unavailable during the training window.
set -euo pipefail
sudo systemctl stop ollama
sleep 2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
