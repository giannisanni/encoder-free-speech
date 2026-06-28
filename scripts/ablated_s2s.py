"""Speech-to-speech in the ablated state: complete the unified ablation table with
the audio column the paper never measured, and test the hybrid think/voice split.

Per spoken query, three end-to-end S2S configs (audio in -> audio out):
  A  +LoRA       : standard unified (strength 1.0). Expect OOD template content,
                   clean audio (WER ~0).
  B  -LoRA       : ablated (strength 0.0 == disable_adapter, numerically identical).
                   Expect correct content (pristine backbone), garbled audio.
  C  hybrid      : THINK with LoRA off (take B's correct reply), VOICE that reply at
                   full LoRA via generate_unified(force_text_ids=...). The one no-retrain
                   config that could give correct content AND intelligible audio.

Audio WER = faster-whisper transcript of the generated audio vs that config's reply text.

Run (substrate):
  GEMMA_BACKBONE=google/gemma-4-12B-it ~/audioin-venv/bin/python -u -m scripts.ablated_s2s \
    --queries ~/ood_wavs/q1.wav ~/ood_wavs/q2.wav ~/ood_wavs/q3.wav ~/promo_q.wav \
    --out ~/ablated_clips
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
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--ckpt", default="unified_a6000/ckpt_3000.pt",
                    help="local path OR repo-relative path in the HF ckpts repo")
    ap.add_argument("--out", default=os.path.expanduser("~/ablated_clips"))
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-coverage", default="", help="'', attn, attn_down, all (match the ckpt)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = GemmaCSMConfig()
    cfg.backbone_id = os.environ.get("GEMMA_BACKBONE", cfg.backbone_id)
    cfg.lora_r, cfg.lora_alpha, cfg.lora_all_linear = a.lora_r, a.lora_alpha, True
    cfg.lora_coverage = a.lora_coverage
    model, tok, _ = load_gemma_csm_audio_in(cfg, device="cuda")
    for m in (getattr(model, "full_model", None), model.backbone):
        try:
            m.gradient_checkpointing_disable()
        except Exception:
            pass

    ckpt_path = a.ckpt if os.path.exists(a.ckpt) else hf_hub_download(CKPT_REPO, a.ckpt, repo_type="model")
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

    def gen(inf, strength, force_text_ids=None, tt=0.6):
        set_lora_strength(model, strength)
        return model.generate_unified(inf, tok, tok.eos_token_id, max_text=96,
                                      max_frames=400, temperature=0.6, top_k=50,
                                      text_temperature=tt, force_text_ids=force_text_ids)

    rows = []
    with torch.no_grad():
        for wp in a.queries:
            name = os.path.splitext(os.path.basename(wp))[0]
            q_asr = transcribe(wp)
            inf = feats_of(wp)
            print(f"\n\n######## QUERY {name}: {q_asr!r} ########", flush=True)

            # A: standard +LoRA
            tA, cA = gen(inf, 1.0)
            wpA = os.path.join(a.out, f"{name}_A_plusLoRA.wav"); save_wav(wpA, codec.decode(cA))
            werA = wer(tA, transcribe(wpA))
            print(f"[A +LoRA   ] reply={tA!r}\n             WER={werA:.2f} -> {wpA}", flush=True)

            # B: ablated -LoRA
            tB, cB = gen(inf, 0.0)
            wpB = os.path.join(a.out, f"{name}_B_ablated.wav"); save_wav(wpB, codec.decode(cB))
            werB = wer(tB, transcribe(wpB))
            print(f"[B -LoRA   ] reply={tB!r}\n             WER={werB:.2f} -> {wpB}", flush=True)

            # C: hybrid -- think with B's (LoRA-off, correct) reply, voice at full LoRA
            force_ids = tok(tB, add_special_tokens=False)["input_ids"]
            tC, cC = gen(inf, 1.0, force_text_ids=force_ids)
            wpC = os.path.join(a.out, f"{name}_C_hybrid.wav"); save_wav(wpC, codec.decode(cC))
            werC = wer(tC, transcribe(wpC))
            print(f"[C hybrid  ] reply={tC!r}\n             WER={werC:.2f} -> {wpC}", flush=True)

            rows.append((name, q_asr, tA, werA, tB, werB, tC, werC))

    print("\n\n===== ABLATED S2S MATRIX =====", flush=True)
    for name, q, tA, wA, tB, wB, tC, wC in rows:
        print(f"\n[{name}] {q!r}", flush=True)
        print(f"  A +LoRA  WER {wA:4.2f}  {tA[:70]!r}", flush=True)
        print(f"  B -LoRA  WER {wB:4.2f}  {tB[:70]!r}", flush=True)
        print(f"  C hybrid WER {wC:4.2f}  {tC[:70]!r}", flush=True)
    print("\nABLATED_DONE", flush=True)


if __name__ == "__main__":
    main()
