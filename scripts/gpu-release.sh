#!/usr/bin/env bash
# Run ON substrate after training. Ollama reloads its model lazily on the
# next request, so just starting the service restores the Hermes fallback.
set -euo pipefail
sudo systemctl start ollama
sleep 2
systemctl is-active ollama
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
