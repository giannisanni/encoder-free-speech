"""Clean hybrid voicing test (v2). The first hybrid run forced the FULL ablated
reply, including the LoRA-off backbone's degenerate <turn|>/thought looping tail,
which confounds the audio WER. Here we truncate the ablated reply at its first
degeneration marker so we voice ONLY the correct clean sentence, then ask: can the
full-LoRA head voice that correct reply intelligibly (no retraining)?

Per query:
  think : LoRA OFF, greedy -> ablated reply, truncated to first clean sentence
  voice : LoRA FULL, generate_unified(force_text_ids=clean reply) -> audio
WER = faster-whisper(audio) vs the clean forced reply.

Run (substrate):
  GEMMA_BACKBONE=google/gemma-4-12B-it ~/audioin-venv/bin/python -u -m scripts.ablated_hybrid_clean \
    --queries ~/ood_wavs/q1.wav ~/ood_wavs/q2.wav ~/ood_wavs/q3.wav ~/promo_q.wav \
    --out ~/hybrid_clean_clips
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
MARKERS = ["<turn|>", "<channel|>", "<|channel>", "<end_of_turn>", "thought", "\n<", "\n\n"]


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


def clean_reply(text):
    t = text.strip()
    for m in ("<channel|>", "<|channel>", "<turn|>", "thought"):   # strip leading marker(s)
        while t.startswith(m):
            t = t[len(m):].strip()
    cut = len(t)
    for m in MARKERS:
        i = t.find(m)
        if i != -1:
            cut = min(cut, i)
    out = t[:cut].strip()
    return out if out else t.strip()


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
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--ckpt", default="unified_a6000/ckpt_3000.pt")
    ap.add_argument("--out", default=os.path.expanduser("~/hybrid_clean_clips"))
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

    ckpt_path = hf_hub_download(CKPT_REPO, a.ckpt, repo_type="model")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd, strict=False)
    model.eval()
    index_lora(model)
    print(f"indexed {len(_LORA_BASE)} LoRA layers", flush=True)

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(cfg.backbone_id)
    codec = MimiCodec(device="cuda")
    from faster_whisper import WhisperModel
    asr = WhisperModel("base.en", device="cuda", compute_type="float16")

    def transcribe(path):
        segs, _ = asr.transcribe(path, language="en")
        return " ".join(s.text for s in segs).strip()

    def feats_of(wp):
        audio, sr = sf.read(wp)
        if audio.ndim > 1:
            audio = audio.mean(1)
        feat = proc(audio=audio.astype("float32"), sampling_rate=sr, return_tensors="pt")
        for k in ("input_features", "audio_values", "input_audio_embeds"):
            if k in feat:
                return feat[k].squeeze(0).to("cuda", torch.bfloat16)
        raise KeyError("no input_features")

    rows = []
    with torch.no_grad():
        for wp in a.queries:
            name = os.path.splitext(os.path.basename(wp))[0]
            inf = feats_of(wp)
            print(f"\n\n######## QUERY {name} ########", flush=True)

            # think: LoRA off, greedy -> ablated reply
            set_lora_strength(model, 0.0)
            t_full, _ = model.generate_unified(inf, tok, tok.eos_token_id, max_text=96,
                                               max_frames=1, temperature=0.6, top_k=50,
                                               text_temperature=0.0)
            reply = clean_reply(t_full)
            print(f"  ablated reply (raw)  : {t_full[:90]!r}", flush=True)
            print(f"  clean reply (forced) : {reply!r}", flush=True)

            # voice: LoRA full, force the clean reply
            force_ids = tok(reply, add_special_tokens=False)["input_ids"]
            set_lora_strength(model, 1.0)
            _, codes = model.generate_unified(inf, tok, tok.eos_token_id, max_text=96,
                                              max_frames=400, temperature=0.6, top_k=50,
                                              force_text_ids=force_ids)
            wpath = os.path.join(a.out, f"{name}_hybrid_clean.wav")
            save_wav(wpath, codec.decode(codes))
            trans = transcribe(wpath)
            w = wer(reply, trans)
            print(f"  audio->ASR           : {trans!r}", flush=True)
            print(f"  clean-hybrid WER     : {w:.2f} -> {wpath}", flush=True)
            rows.append((name, reply, trans, w))

    print("\n\n===== CLEAN HYBRID SUMMARY =====", flush=True)
    for name, reply, trans, w in rows:
        print(f"[{name}] WER {w:4.2f}  forced={reply[:55]!r}  heard={trans[:55]!r}", flush=True)
    print("\nCLEAN_HYBRID_DONE", flush=True)


if __name__ == "__main__":
    main()
