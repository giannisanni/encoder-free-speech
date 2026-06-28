"""LoRA-strength sweep for the unified model (no retraining).

Question: the unified head (unified_a6000/ckpt_3000) was fit with a QLoRA that also
perturbs the frozen Gemma backbone out-of-distribution, so OOD prompts collapse to
training-style empathy templates ("It sounds like you're feeling lonely..."). Does
dialing the adapter DOWN (1.0 -> 0.0) let the pristine backbone reasoning resurface
(e.g. an actual "Congratulations!") while the head still produces intelligible audio?

For each strength s in --strengths: scale every LoRA layer's `scaling` to base*s,
run generate_unified on the spoken query, decode to wav, transcribe with
faster-whisper, and report the reply text + audio WER (transcription vs reply text).

Run (substrate):
  GEMMA_BACKBONE=google/gemma-4-12B-it ~/audioin-venv/bin/python -m scripts.sweep_lora_strength \
    --query ~/promo_q.wav --out ~/sweep_clips
"""
import argparse
import os
import re
import wave

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

from gemma_csm.codec import MimiCodec
from gemma_csm.config import GemmaCSMConfig
from gemma_csm.loader import load_gemma_csm_audio_in

CKPT_REPO = "giannisan/gemma-csm-ckpts"

_LORA_BASE = {}


def index_lora(model):
    for name, mod in model.backbone.named_modules():
        sc = getattr(mod, "scaling", None)
        if isinstance(sc, dict):
            _LORA_BASE[name] = dict(sc)


def set_lora_strength(model, s):
    s = float(s)
    for name, mod in model.backbone.named_modules():
        base = _LORA_BASE.get(name)
        if base:
            for adp, b in base.items():
                if adp in mod.scaling:
                    mod.scaling[adp] = b * s


def save_wav(path, wav, sr=24000):
    x = (wav.detach().cpu().float().clamp(-1, 1).numpy().reshape(-1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes())


def _norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).split()


def wer(ref, hyp):
    r, h = _norm(ref), _norm(hyp)
    if not r:
        return float("nan")
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--ckpt", default="unified_a6000/ckpt_3000.pt")
    ap.add_argument("--out", default=os.path.expanduser("~/sweep_clips"))
    ap.add_argument("--strengths", default="1.0,0.6,0.4,0.2,0.0")
    ap.add_argument("--samples", type=int, default=1, help="reply samples per strength")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    strengths = [float(x) for x in a.strengths.split(",")]

    cfg = GemmaCSMConfig()
    cfg.backbone_id = os.environ.get("GEMMA_BACKBONE", cfg.backbone_id)
    cfg.lora_r, cfg.lora_alpha, cfg.lora_all_linear = 32, 16, True
    model, tok, _ = load_gemma_csm_audio_in(cfg, device="cuda")
    for m in (getattr(model, "full_model", None), model.backbone):
        try:
            m.gradient_checkpointing_disable()
        except Exception:
            pass

    ckpt_path = hf_hub_download(CKPT_REPO, a.ckpt, repo_type="model")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {a.ckpt}: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    model.eval()
    index_lora(model)
    print(f"indexed {len(_LORA_BASE)} LoRA layers", flush=True)

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(cfg.backbone_id)
    codec = MimiCodec(device="cuda")

    from faster_whisper import WhisperModel
    asr = WhisperModel("base.en", device="cuda", compute_type="float16")

    audio, sr = sf.read(a.query)
    if audio.ndim > 1:
        audio = audio.mean(1)
    feat = proc(audio=audio.astype("float32"), sampling_rate=sr, return_tensors="pt")
    inf = None
    for k in ("input_features", "audio_values", "input_audio_embeds"):
        if k in feat:
            inf = feat[k]; break
    inf = inf.squeeze(0).to("cuda", torch.bfloat16)

    rows = []
    with torch.no_grad():
        for s in strengths:
            set_lora_strength(model, s)
            for j in range(a.samples):
                text, codes = model.generate_unified(
                    inf, tok, tok.eos_token_id, max_text=96, max_frames=400,
                    temperature=0.6, top_k=50, text_temperature=0.6)
                wav = codec.decode(codes)
                tag = f"s{s:.2f}_{j}"
                wpath = os.path.join(a.out, f"reply_{tag}.wav")
                save_wav(wpath, wav)
                segs, _ = asr.transcribe(wpath, language="en")
                trans = " ".join(seg.text for seg in segs).strip()
                w = wer(text, trans)
                rows.append((s, j, text, trans, w))
                print(f"\n[strength {s:.2f} #{j}]", flush=True)
                print(f"  reply text : {text!r}", flush=True)
                print(f"  audio->ASR : {trans!r}", flush=True)
                print(f"  audio WER  : {w:.2f}   -> {wpath}", flush=True)

    print("\n\n===== SWEEP SUMMARY =====", flush=True)
    print(f"{'strength':>8}  {'WER':>5}  reply text", flush=True)
    for s, j, text, trans, w in rows:
        print(f"{s:>8.2f}  {w:>5.2f}  {text[:80]!r}", flush=True)
    print("\nSWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
