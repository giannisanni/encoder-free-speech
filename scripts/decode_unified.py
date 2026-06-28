"""Decode the unified (native-delivery) model: free-running generate_unified ->
Mimi wav + whisper WER + zeroed-audio grounding ablation. This is the Phase-2.1
gate: does the unified model SPEAK a grounded reply?

For each item: generate reply text+audio from the real query audio, decode to wav,
whisper-transcribe it, WER(whisper vs the model's own reply text); then regenerate
with ZEROED audio and show the reply changed (no bypass).

Run (substrate):
  GEMMA_BACKBONE=google/gemma-4-12B-it PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ~/audioin-venv/bin/python -m scripts.decode_unified \
    --ckpt checkpoints/unified_overfit/ckpt_800.pt \
    --shard data/shards_overfit/tiny.pt --n 4 --out ~/unified_clips
"""
import argparse
import os
import wave

import torch

from gemma_csm.codec import MimiCodec
from gemma_csm.config import GemmaCSMConfig
from gemma_csm.dataset_opens2s import OpenS2SDataset
from gemma_csm.loader import load_gemma_csm_audio_in


def save_wav(path, wav, sr=24000):
    x = wav.detach().cpu().float().clamp(-1, 1).numpy()
    import numpy as np
    pcm = (x * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=os.path.expanduser("~/unified_clips"))
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.6)
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
    print(f"loaded {a.ckpt}: missing={len(miss)} (frozen backbone, expected) unexpected={len(unexp)}", flush=True)
    model.eval()

    codec = MimiCodec(device="cuda")
    from faster_whisper import WhisperModel
    asr = WhisperModel("base.en", device="cpu", compute_type="int8")
    try:
        from jiwer import wer as jiwer_wer
        from whisper_normalizer.english import EnglishTextNormalizer
        norm = EnglishTextNormalizer()
    except Exception:
        jiwer_wer = None; norm = lambda s: s.lower().strip()

    ds = OpenS2SDataset(a.shard)
    for i in range(min(a.n, len(ds))):
        it = ds[i]
        feats = it["input_features"].to("cuda", torch.bfloat16)
        q = it.get("query_text", "")
        text, codes = model.generate_unified(feats, tok, tok.eos_token_id,
                                             max_text=64, max_frames=a.max_frames,
                                             temperature=a.temperature, top_k=50)
        zero_text, _ = model.generate_unified(torch.zeros_like(feats), tok, tok.eos_token_id,
                                              max_text=64, max_frames=1, text_temperature=0.0)
        wav = codec.decode(codes)
        wp = f"{a.out}/uni_{i}.wav"
        save_wav(wp, wav)
        # whisper the generated audio, WER vs the model's own reply text
        segs, _ = asr.transcribe(wp, language="en")
        hyp = " ".join(s.text for s in segs).strip()
        w = jiwer_wer(norm(text), norm(hyp)) if (jiwer_wer and text.strip() and hyp.strip()) else float("nan")
        print(f"\n[{i}] query: {q!r}", flush=True)
        print(f"    reply (real audio): {text!r}", flush=True)
        print(f"    reply (zeroed)    : {zero_text!r}   <- must differ (grounding)", flush=True)
        print(f"    whisper<-wav      : {hyp!r}", flush=True)
        print(f"    {wav.numel()/24000:4.1f}s  WER(reply vs spoken)={w:.2f}  -> {wp}", flush=True)

    print("\nDONE_DECODE", flush=True)


if __name__ == "__main__":
    main()
