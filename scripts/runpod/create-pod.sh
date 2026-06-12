#!/usr/bin/env bash
# Create a RunPod GPU pod ready for gemma-csm (run on the Mac).
# Prereq once: runpodctl config --apiKey <key from runpod.io Settings>
# Usage: scripts/runpod/create-pod.sh [gpu-type]   (default A100 80GB PCIe)
set -euo pipefail
GPU="${1:-NVIDIA A100 80GB PCIe}"
runpodctl create pod \
  --name gemma-csm \
  --gpuType "$GPU" --gpuCount 1 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --containerDiskSize 60 --volumeSize 80 \
  --secureCloud=false \
  --env "HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in your shell first}" \
  --ports "22/tcp"
echo "Pod created. SSH in (runpodctl get pod for connection info), then:"
echo "  curl -sL <bootstrap> | bash   — or scp scripts/runpod/bootstrap.sh over"
echo "Remember: runpodctl stop/remove pod when done — billing is per minute."
