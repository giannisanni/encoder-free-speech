"""Assess ckpt_14000 generalization (old vs new shard c0) AND decode clips.
Builds the 12B once. GPU must be free (training done)."""
import argparse
import glob
import re
from pathlib import Path

import wave

import numpy as np
import torch

from gemma_csm.model import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm_audio_in
from gemma_csm.codec import MimiCodec
from gemma_csm.dataset_opens2s import OpenS2SDataset
from gemma_csm.train_audio_in import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/a1_speak/ckpt_14000.pt")
ap.add_argument("--n-clips", type=int, default=5)
ap.add_argument("--max-frames", type=int, default=200)
a = ap.parse_args()

def save_wav(path, wav, sr=24000):
    x = wav.detach().cpu().float().clamp(-1, 1).numpy()
    pcm = (x * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


cfg = GemmaCSMConfig()
cfg.backbone_id = "google/gemma-4-12B-it"
cfg.lora_r, cfg.lora_alpha = 32, 16
cfg.lora_all_linear = True
cfg.depth_loss_frame_frac = 0.25
cfg.depth_loss_weight = 0.01

print("building + loading", a.ckpt, flush=True)
model, tok, audio_encode = load_gemma_csm_audio_in(cfg)
sd = torch.load(a.ckpt, weights_only=True, map_location="cpu")
step = sd.get("step", 0) if isinstance(sd, dict) else 0
if isinstance(sd, dict) and "model" in sd:
    sd = sd["model"]
for k in [k for k, v in sd.items() if k in model.state_dict() and model.state_dict()[k].shape != v.shape]:
    del sd[k]
model.load_state_dict(sd, strict=False)
model.eval()


def shard_items(pattern, n=40):
    items = []
    for f in sorted(glob.glob(pattern)):
        items += OpenS2SDataset([f]).items
    return items[:n]


old = shard_items("data/shards_opens2s/opens2s_en_00[01].pt")
new = shard_items("data/shards_opens2s/opens2s_en_00[2-9].pt")
print(f"\n=== GENERALIZATION (ckpt step {step}) ===", flush=True)
for name, items in [("OLD/trained", old), ("NEW/unseen", new)]:
    c0, d = evaluate(model, items, audio_encode, tok, 64, 90, no_audio_in=True, max_text=56)
    print(f"  {name:14s} c0={c0:.3f} depth={d:.3f}", flush=True)

print("\n=== DECODE CLIPS ===", flush=True)
codec = MimiCodec(device="cuda")
out = Path("checkpoints/a1_speak/samples")
out.mkdir(parents=True, exist_ok=True)
# decode from UNSEEN prompts (honest) + tag clip index
prompts = [it.get("text", "").strip() for it in new]
prompts = [t for t in prompts if 20 <= len(t) <= 140][:a.n_clips]
for i, text in enumerate(prompts):
    ids = torch.tensor([tok(text)["input_ids"]], device="cuda")
    wav, codes, att = model.generate_wav(ids, codec, max_frames=a.max_frames,
                                         temperature=0.9, top_k=50)
    p = out / f"step{step:06d}_s{i}.wav"
    save_wav(p, wav)
    print(f"  [{i}] {wav.numel()/24000:4.1f}s att={att} <- {text[:55]!r} -> {p.name}", flush=True)
print("ASSESS+DECODE DONE", flush=True)
