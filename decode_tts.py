"""Decode a few clips from a TTS (LibriTTS) Phase-A checkpoint.
Prompts use the training format '[speaker] text'. wav via stdlib wave."""
import argparse, glob, os, wave
import numpy as np
import torch
from gemma_csm.model import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm_audio_in
from gemma_csm.codec import MimiCodec


def save_wav(path, wav, sr=24000):
    x = wav.detach().cpu().float().clamp(-1, 1).numpy()
    pcm = (x * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--shards", default="data/shards_libritts/*.pt")
ap.add_argument("--n-clips", type=int, default=5)
ap.add_argument("--max-frames", type=int, default=250)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--out", default="samples_tts")
a = ap.parse_args()

cfg = GemmaCSMConfig()
cfg.backbone_id = "google/gemma-4-12B-it"
cfg.lora_r, cfg.lora_alpha = 32, 16
cfg.lora_all_linear = True
print("loading model + ckpt ...", flush=True)
model, tok, _ = load_gemma_csm_audio_in(cfg)
sd = torch.load(a.ckpt, weights_only=True, map_location="cpu")
step = sd.get("step", 0) if isinstance(sd, dict) else 0
if isinstance(sd, dict) and "model" in sd:
    sd = sd["model"]
own = model.state_dict()
for k in [k for k, v in sd.items() if k in own and own[k].shape != v.shape]:
    del sd[k]
model.load_state_dict(sd, strict=False)
model.eval()
codec = MimiCodec(device="cuda")

items = torch.load(sorted(glob.glob(a.shards))[0], weights_only=True)
prompts = []
for it in items:
    t = (it.get("text", "") or "").strip()
    if 30 <= len(t) <= 120:
        prompts.append((f"[{it.get('speaker', 'lj')}] {t}", t))
    if len(prompts) >= a.n_clips:
        break

os.makedirs(a.out, exist_ok=True)
print(f"decoding {len(prompts)} clips from step {step}", flush=True)
for i, (prompt, ref) in enumerate(prompts):
    ids = torch.tensor([tok(prompt, add_special_tokens=True)["input_ids"]], device="cuda")
    wav, codes, att = model.generate_wav(ids, codec, max_frames=a.max_frames,
                                         temperature=a.temperature, top_k=50)
    p = f"{a.out}/tts_step{step:06d}_s{i}.wav"
    save_wav(p, wav)
    print(f"[{i}] {wav.numel()/24000:4.1f}s att={att} <- {ref[:55]!r} -> {p}", flush=True)
print("DECODE DONE", flush=True)
