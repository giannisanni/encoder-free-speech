"""Decode held-out prompts from a Phase-A (text->speech) checkpoint to wav.

Loads the 12B audio-IN model (text decoder + speech heads), injects a
trainable checkpoint (LoRA + c0/depth heads), and runs model.generate_wav on
a few dataset texts so we can listen to how the run speaks at a given step.

Writes checkpoints/<run>/samples/step<STEP>_s<i>.wav. Decode needs ~10GB VRAM,
so only run it when the training process is paused/stopped (the 16GB card
can't hold two copies of the 12B).
"""
import argparse
import glob
import re
from pathlib import Path

import torch
import torchaudio

from gemma_csm.model import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm_audio_in
from gemma_csm.codec import MimiCodec


def pick_prompts(shard_glob: str, n: int):
    """A few in-distribution response texts, tokenized exactly as training does."""
    texts = []
    for f in sorted(glob.glob(shard_glob)):
        for it in torch.load(f, weights_only=True):
            t = (it.get("text") or "").strip()
            if 20 <= len(t) <= 160:
                texts.append(t)
            if len(texts) >= n:
                return texts
    return texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run", default="a1_speak")
    p.add_argument("--n-clips", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--backbone-id", default="google/gemma-4-12B-it")
    p.add_argument("--shards", default="data/shards_opens2s/opens2s_en_00*.pt")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    m = re.search(r"(\d+)", Path(a.ckpt).stem)
    step = int(m.group(1)) if m else 0

    cfg = GemmaCSMConfig()
    cfg.backbone_id = a.backbone_id
    print("loading 12B audio-IN model ...", flush=True)
    model, tok, _ = load_gemma_csm_audio_in(cfg, device=a.device)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    torch.cuda.empty_cache()
    print(f"ckpt loaded: {len(missing)} missing (frozen backbone, expected), "
          f"{len(unexpected)} unexpected", flush=True)
    if unexpected:
        print("  WARN unexpected keys:", unexpected[:5], flush=True)
    model.eval()

    codec = MimiCodec(device=a.device)
    prompts = pick_prompts(a.shards, a.n_clips)
    if not prompts:
        raise SystemExit("no prompts found in shards: " + a.shards)

    out = Path("checkpoints") / a.run / "samples"
    out.mkdir(parents=True, exist_ok=True)
    print(f"decoding {len(prompts)} clips at step {step} -> {out}", flush=True)

    for i, text in enumerate(prompts):
        ids = torch.tensor([tok(text)["input_ids"]], device=a.device)
        wav, codes, attempts = model.generate_wav(
            ids, codec, max_frames=a.max_frames,
            temperature=a.temperature, top_k=a.top_k)
        dur = wav.numel() / 24000.0
        path = out / f"step{step:06d}_s{i}.wav"
        torchaudio.save(str(path), wav.detach().cpu().unsqueeze(0), 24000)
        print(f"[{i}] {dur:4.1f}s att={attempts} frames={codes.shape[-1]} "
              f"<- {text[:60]!r} -> {path.name}", flush=True)
    print("DECODE DONE", flush=True)


if __name__ == "__main__":
    main()
