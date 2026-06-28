"""Out-of-distribution probe for the unified model: feed it custom spoken queries
(NOT OpenS2S emotional-support style) and see if the frozen Gemma backbone is
genuinely engaged (sensible/identity/factual replies) vs. parroting training
templates. Synthesize queries with macOS `say`, process via the Gemma processor,
run generate_unified, decode to wav.

Run (substrate): GEMMA_BACKBONE=google/gemma-4-12B-it \
  ~/audioin-venv/bin/python -m scripts.test_unified_ood \
    --ckpt checkpoints/unified_a6000/ckpt_3000.pt --wav-dir ~/ood_wavs --out ~/ood_clips
"""
import argparse
import glob
import os
import wave

import numpy as np
import soundfile as sf
import torch

from gemma_csm.codec import MimiCodec
from gemma_csm.config import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm_audio_in


def save_wav(path, wav, sr=24000):
    x = (wav.detach().cpu().float().clamp(-1, 1).numpy() * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/ood_clips"))
    ap.add_argument("--no-lora", action="store_true",
                    help="#2 ablation: disable LoRA adapters -> pure frozen backbone via the head")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = GemmaCSMConfig()
    cfg.backbone_id = os.environ.get("GEMMA_BACKBONE", cfg.backbone_id)
    cfg.lora_r, cfg.lora_alpha, cfg.lora_all_linear = 32, 16, True
    model, tok, _ = load_gemma_csm_audio_in(cfg, device="cuda")
    for m in (getattr(model, "full_model", None), model.backbone):
        try:
            m.gradient_checkpointing_disable()
        except Exception:
            pass
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {a.ckpt}: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    model.eval()

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(cfg.backbone_id)
    codec = MimiCodec(device="cuda")

    import contextlib
    print("LoRA:", "DISABLED (#2 ablation)" if a.no_lora else "enabled", flush=True)

    for wp in sorted(glob.glob(os.path.join(a.wav_dir, "*.wav"))):
        audio, sr = sf.read(wp)
        if audio.ndim > 1:
            audio = audio.mean(1)
        feat = proc(audio=audio.astype("float32"), sampling_rate=sr, return_tensors="pt")
        inf = None
        for k in ("input_features", "audio_values", "input_audio_embeds"):
            if k in feat:
                inf = feat[k]; break
        inf = inf.squeeze(0).to("cuda", torch.bfloat16)               # (Ta, 640)
        gctx = model.backbone.disable_adapter() if a.no_lora else contextlib.nullcontext()
        with gctx:
            text, codes = model.generate_unified(inf, tok, tok.eos_token_id,
                                                 max_text=96, max_frames=400,
                                                 temperature=0.6, top_k=50, text_temperature=0.6)
        out_wav = os.path.join(a.out, os.path.basename(wp).replace(".wav", "_reply.wav"))
        save_wav(out_wav, codec.decode(codes))
        print(f"\n[{os.path.basename(wp)}]  reply: {text!r}\n    -> {out_wav}", flush=True)
    print("\nOOD_DONE", flush=True)


if __name__ == "__main__":
    main()
