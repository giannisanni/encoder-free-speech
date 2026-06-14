"""Tiny end-to-end audio-IN training-step test on the 12B with REAL OpenS2S data
(paper rev. 28). Loads the audio-IN loader, streams a real shard through
dataset+collate, runs audio_encode -> audio_in_embeds -> GemmaCSM.forward ->
backward. Proves the full loop: real data -> audio understanding -> speech-out
heads -> gradients.

Run:  .venv-audioin/bin/python -m scripts.probe_audio_in_train  (audio-IN venv)
"""
import glob
import os

import torch
from torch.utils.data import DataLoader

from gemma_csm.config import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm_audio_in
from gemma_csm.dataset_opens2s import OpenS2SDataset, collate_opens2s

SHARDS = os.environ.get("OPENS2S_SHARDS", "/tmp/shards_opens2s_test/*.pt")
BSZ = int(os.environ.get("BSZ", "2"))

cfg = GemmaCSMConfig()
cfg.backbone_id = "google/gemma-4-12B-it"
cfg.lora_r, cfg.lora_alpha, cfg.lora_all_linear = 32, 16, True
print("loading audio-IN model (12B) ...", flush=True)
model, tok, audio_encode = load_gemma_csm_audio_in(cfg)

ds = OpenS2SDataset(SHARDS)
print(f"dataset: {len(ds)} items, batch_size={BSZ}", flush=True)
dl = DataLoader(ds, batch_size=BSZ, shuffle=False,
                collate_fn=lambda b: collate_opens2s(b, tok, max_audio=int(os.environ.get("MAX_AUDIO",48)), max_frames=int(os.environ.get("MAX_FRAMES",100))))
batch = next(iter(dl))
batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
print("collated:",
      "text_ids", tuple(batch["text_ids"].shape),
      "codes", tuple(batch["codes"].shape),
      "input_features", tuple(batch["input_features"].shape), flush=True)

# audio understanding: input_features -> backbone-dim audio_in_embeds
af = audio_encode(batch["input_features"].to(torch.bfloat16),
                  batch["input_features_mask"])
batch["audio_in_embeds"] = af
batch["audio_in_mask"] = batch["input_features_mask"]
print("audio_in_embeds:", tuple(af.shape), "(want last dim = backbone hidden)", flush=True)

out = model(batch)
print(f"forward OK: loss={float(out['loss']):.3f} "
      f"c0={float(out['loss_c0']):.3f} depth={float(out['loss_depth']):.3f}", flush=True)
out["loss"].backward()
g = sum(p.grad.abs().sum().item() for p in model.parameters()
        if p.requires_grad and p.grad is not None)
print(f"backward OK: sum|grad|={g:.2e}", flush=True)
print("peak VRAM:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB", flush=True)
print("=== AUDIO-IN TRAIN STEP CONFIRMED (real OpenS2S data, 12B) ==="
      if g > 0 else "=== FAILED ===", flush=True)
